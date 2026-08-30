import os
import sys
import importlib
import csv
import gymnasium as gym
import numpy as np

# Compatibility layer for numpy < 2 loading numpy >= 2 checkpoints
try:
    importlib.import_module("numpy._core")
except ImportError:
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from usv_gym_env import CurriculumUSVEnv

# Cleanup old metrics
if os.path.exists("eval_metrics_history.csv"):
    os.remove("eval_metrics_history.csv") # Assessment data to plot the graphic

def make_env(stage):
    def _init():
        env = CurriculumUSVEnv(stage=stage)
        return Monitor(env)
    return _init

class ScenarioLockedEnv(gym.Env):
    """Temporary environment locked on a single scenario for evaluation purposes."""
    def __init__(self, scenario: str, stage: str):
        super().__init__()
        self.scenario = scenario
        self.base_env = CurriculumUSVEnv(stage=stage)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

    def reset(self, seed=None, options=None):
        opts = {} if options is None else dict(options)
        opts["scenario"] = self.scenario
        obs, info = self.base_env.reset(seed=seed, options=opts)
        return obs, info

    def step(self, action):
        return self.base_env.step(action)

class AutoCurriculumCallback(BaseCallback):
    """Evaluates the model on separated scenarios and logs metrics to CSV."""
    def __init__(self, eval_freq=10000, n_eval_episodes=100, verbose=1):
        super(AutoCurriculumCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.current_phase = None 

    def _on_step(self) -> bool:
        if self.current_phase is None:
            self.current_phase = self.training_env.get_attr("stage")[0]

        if self.n_calls % self.eval_freq == 0:
            print(f"\n--- Detailed Evaluation ({self.current_phase}) at {self.num_timesteps} steps ---")
            
            pool = self.training_env.get_attr("_scenario_pool")[0]()
            unique_scenarios = list(set(pool))
            hard_passed = True
            
            for sc in unique_scenarios:
                tmp_env = DummyVecEnv([lambda: ScenarioLockedEnv(scenario=sc, stage=self.current_phase)])
                tmp_env = VecNormalize(tmp_env, training=False, norm_reward=False)
                sync_envs_normalization(self.training_env, tmp_env)
                
                successes, errors = 0, 0
                
                for _ in range(self.n_eval_episodes):
                    obs = tmp_env.reset()
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, dones, infos = tmp_env.step(action)
                        done = dones[0]
                        info = infos[0]
                        
                        if done:
                            if info.get("event") == "goal":
                                successes += 1
                            elif info.get("event") in ["out_of_bounds", "collision"]:
                                errors += 1
                                
                s_rate = successes / self.n_eval_episodes
                e_rate = errors / self.n_eval_episodes
                
                print(f" > Scenario '{sc}': Success = {s_rate:.2f} | Error = {e_rate:.2f}")

                # Save metrics to CSV
                csv_file = "eval_metrics_history.csv"
                file_exists = os.path.isfile(csv_file)
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["step", "scenario", "success_rate", "error_rate"])
                    writer.writerow([self.num_timesteps, sc, s_rate, e_rate])
                
 
                if s_rate < 0.90 or e_rate > 0.1:
                  hard_passed = False
                
            if hard_passed:
                print("\n" + "="*70)
                print("GLOBAL OBJECTIVE REACHED! Model validated all scenarios with excellence.")
                print("="*70 + "\n")
                
                save_path = f"logs_rl/usv_ppo_VALIDATED_{self.current_phase}_{self.num_timesteps}_steps"
                self.model.save(save_path)
                return False  # STOP TRAINING
                
            print("-" * 60 + "\n")
            
        return True

def main():
    log_dir = "logs_rl"
    os.makedirs(log_dir, exist_ok=True)

    input_model = os.getenv("USV_INPUT_MODEL", "usv_rl_brain")
    input_vecnorm = os.getenv("USV_INPUT_VECNORM", os.path.join(log_dir, "vecnormalize.pkl"))
    output_model = os.getenv("USV_OUTPUT_MODEL", "usv_rl_brain")
    output_vecnorm = os.getenv("USV_OUTPUT_VECNORM", os.path.join(log_dir, "vecnormalize.pkl"))

    train_stage = os.getenv("USV_TRAIN_STAGE", "phase_2")
    additional_timesteps = int(os.getenv("USV_TOTAL_TIMESTEPS", "200000"))
    device = os.getenv("USV_DEVICE", "cpu")

    print(f"Resume training | input={input_model} | stage={train_stage} | timesteps={additional_timesteps}")

    train_env = DummyVecEnv([make_env(train_stage)])
    
    try:
        train_env = VecNormalize.load(input_vecnorm, train_env)
        train_env.training = True
        train_env.norm_reward = True
        print(f"Loaded VecNormalize for training from {input_vecnorm}.")
    except Exception:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
        print("No VecNormalize found, created a fresh one.")

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=log_dir, name_prefix="usv_ppo_checkpoint")
    auto_curriculum = AutoCurriculumCallback(eval_freq=10000, n_eval_episodes=100)

    # Custom objects (e.g., dynamically changing clip range during resume)
    custom_objects = {}
    new_clip = os.getenv("USV_CLIP_RANGE")
    if new_clip is not None:
        custom_objects["clip_range"] = float(new_clip)
        print(f"Clip range overridden to {custom_objects['clip_range']} via custom_objects.")

    # Load existing model
    model = PPO.load(input_model, env=train_env, device=device, custom_objects=custom_objects)
    print(f"Model loaded from {input_model}, continuing training...")

    # Overwrite Learning Rate if specified
    new_lr = os.getenv("USV_LEARNING_RATE")
    if new_lr is not None:
        lr_val = float(new_lr)
        model.learning_rate = lr_val
        model.lr_schedule = lambda _: lr_val
        print(f"Learning rate overridden to {lr_val}.")

    model.learn(
        total_timesteps=additional_timesteps, 
        reset_num_timesteps=False, 
        callback=[checkpoint_callback, auto_curriculum]
    )

    model.save(output_model)
    train_env.save(output_vecnorm)
    print(f"Resume training finished. Saved model={output_model}, vecnorm={output_vecnorm}.")

if __name__ == "__main__":
    main()
