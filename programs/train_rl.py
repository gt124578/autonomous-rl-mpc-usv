import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

from usv_gym_env import CurriculumUSVEnv


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
    """Evaluates the model periodically and stops training if target thresholds are met."""
    def __init__(self, eval_freq=10000, n_eval_episodes=100, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.current_phase = None 

    def _on_step(self) -> bool:
        if self.current_phase is None:
            self.current_phase = self.training_env.get_attr("stage")[0]

        if self.n_calls % self.eval_freq == 0:
            print(f"\n--- Initial Evaluation ({self.current_phase}) at {self.num_timesteps} steps ---")
            
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

                # STRICT (but easy at this step) THRESHOLD (98% success, 2% error max)
                if s_rate < 0.98 or e_rate > 0.02:
                    hard_passed = False
                
            if hard_passed:
                print("\n" + "="*70)
                print("INITIAL VALIDATION PASSED (Success >= 0.98, Error <= 0.02) !")
                print("="*70 + "\n")
                return False  # Stops training early
                
            print("-" * 60 + "\n")
            
        return True


def main():
    log_dir = "logs_rl"
    os.makedirs(log_dir, exist_ok=True)

    train_stage = os.getenv("USV_TRAIN_STAGE", "free_only")
    total_timesteps = int(os.getenv("USV_TOTAL_TIMESTEPS", "1000000"))
    device = os.getenv("USV_DEVICE", "cpu")
    output_model = os.getenv("USV_OUTPUT_MODEL", "usv_rl_brain_free")
    output_vecnorm = os.getenv("USV_OUTPUT_VECNORM", os.path.join(log_dir, "vecnormalize_free.pkl"))

    # 1. Multiprocessing Environment
    num_cpu = 10
    train_env = SubprocVecEnv([make_env(train_stage) for _ in range(num_cpu)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # 2. Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=log_dir, name_prefix="usv_ppo_checkpoint")
    auto_curriculum = AutoCurriculumCallback(eval_freq=10000, n_eval_episodes=100) # Updated to 100 episodes

    # 3. PPO Model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,
        device=device,
    )

    print(f"Starting training | stage={train_stage} | timesteps={total_timesteps} | cpus={num_cpu}")
    
    # 4. Train & Save
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, auto_curriculum])
    model.save(output_model)
    train_env.save(output_vecnorm)
    print(f"Model saved successfully: model={output_model}, vecnorm={output_vecnorm}.")

if __name__ == "__main__":
    main()
