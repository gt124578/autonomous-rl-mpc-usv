import os
import sys
import importlib
import pandas as pd
import csv
import numpy as np
import gymnasium as gym

# Handle NumPy compatibility for different versions
try:
    importlib.import_module("numpy._core")
except ImportError:
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization
from usv_gym_env import CurriculumUSVEnv


def make_env(stage):
    """
    Factory function to create a monitored environment for a specific training stage.
    
    Args:
        stage: The curriculum stage to train on
        
    Returns:
        A callable that instantiates a monitored CurriculumUSVEnv
    """
    def _init():
        env = CurriculumUSVEnv(stage=stage)
        return Monitor(env)
    return _init


class ScenarioLockedEnv(gym.Env):
    """
    Wrapper environment that locks evaluation to a specific scenario.
    Allows testing the model on individual scenarios without curriculum progression.
    """
    def __init__(self, scenario: str, stage: str):
        super().__init__()
        self.scenario = scenario
        self.base_env = CurriculumUSVEnv(stage=stage)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

    def reset(self, seed=None, options=None):
        """
        Reset the environment with a specific locked scenario.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options passed to base environment
            
        Returns:
            Initial observation and info dict
        """
        opts = {} if options is None else dict(options)
        opts["scenario"] = self.scenario 
        obs, info = self.base_env.reset(seed=seed, options=opts)
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        return self.base_env.step(action)


class AutoCurriculumCallback(BaseCallback):
    """
    Callback for periodic model evaluation and high-score tracking.
    
    Evaluates the model on all scenarios at regular intervals:
    - Tests on a "boss" scenario (mixed_dynamic) for overall performance
    - Tests on "guardian" scenarios (others) to ensure skill retention
    - Only saves models that improve the high score
    
    The curriculum progresses when:
    1. All guardian scenarios maintain >90% success rate
    2. Boss scenario succeeds more than the current best, OR
    3. Same boss success but with lower error rate
    """
    def __init__(self, shared_state, attempt, train_env_to_save, eval_freq=1000, n_eval_episodes=100, verbose=1):
        """
        Initialize the callback.
        
        Args:
            shared_state: Dict tracking best_success, best_error across attempts
            attempt: Current training attempt number
            train_env_to_save: Training environment for saving normalization stats
            eval_freq: Evaluation frequency (in num_envs steps for SubprocVecEnv)
            n_eval_episodes: Number of episodes to evaluate per scenario
            verbose: Logging verbosity level
        """
        super(AutoCurriculumCallback, self).__init__(verbose)
        self.shared_state = shared_state
        self.attempt = attempt
        self.train_env_to_save = train_env_to_save  
        self.eval_freq = eval_freq  # Set to 1000 for 10 cores (1000 * 10 = 10000 steps)
        self.n_eval_episodes = n_eval_episodes
        self.current_phase = None 

    def _on_step(self) -> bool:
        """
        Called after each environment step during training.
        Triggers evaluation at regular intervals and saves high-score models.
        """
        # Initialize current phase from training environment
        if self.current_phase is None:
            self.current_phase = self.training_env.get_attr("stage")[0]

        # Check if evaluation interval has been reached
        if self.n_calls % self.eval_freq == 0:
            # Calculate actual global steps (accounting for multi-processing)
            real_steps = self.n_calls * self.training_env.num_envs
            
            print(f"\n--- Evaluation (Attempt {self.attempt}/150 | Step {real_steps}) ---")
            
            # Retrieve all available scenarios from the environment
            pool = self.training_env.get_attr("_scenario_pool")[0]()
            unique_scenarios = list(set(pool))
            
            # ===== BOSS SCENARIO STRATEGY =====
            # Boss (mixed_dynamic) is the primary performance metric
            target_boss = "mixed_dynamic"
            if target_boss in unique_scenarios:
                # Evaluate boss first, then guardians
                ordered_scenarios = [target_boss] + [s for s in unique_scenarios if s != target_boss]
            else:
                ordered_scenarios = unique_scenarios  # Fallback if stage misconfigured
            
            base_scenarios_passed = True
            boss_success = 0.0
            boss_error = 0.0
            
            best_s = self.shared_state["best_success"]
            best_e = self.shared_state["best_error"]
            
            # ===== EVALUATE EACH SCENARIO =====
            for sc in ordered_scenarios:
                # Create scenario-locked environment
                def _init():
                    return ScenarioLockedEnv(scenario=sc, stage=self.current_phase)
                
                # Sequential evaluation to avoid excessive RAM usage
                tmp_env = DummyVecEnv([_init])
                tmp_env = VecNormalize(tmp_env, training=False, norm_reward=False)
                # Sync normalization stats from training environment
                sync_envs_normalization(self.training_env, tmp_env)
                
                # Run episodes and collect success/error rates
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
                            # Track outcomes
                            if info.get("event") == "goal":
                                successes += 1
                            elif info.get("event") in ["out_of_bounds", "collision"]:
                                errors += 1
                                
                s_rate = successes / self.n_eval_episodes
                e_rate = errors / self.n_eval_episodes
                print(f" > '{sc}': Success = {s_rate:.2f} | Error = {e_rate:.2f}")
                
                # ===== BOSS EVALUATION =====
                if sc == target_boss:
                    boss_success = s_rate
                    boss_error = e_rate
                    
                    # Check for regression on boss scenario
                    if boss_success < best_s or (boss_success == best_s and boss_error > best_e):
                        print(f"  Regression on {target_boss} (Score: {boss_success:.2f} < {best_s:.2f}). Early Exit.")
                        base_scenarios_passed = False
                        break 
                        
                # ===== GUARDIAN EVALUATION =====
                else:
                    # Guardian scenarios require 90% minimum success to maintain skill
                    if s_rate < 0.90:
                        print(f"  Skill loss on {sc} (Success: {s_rate:.2f} < 0.90). Early Exit.")
                        base_scenarios_passed = False
                        break 

            # ===== SAVE HIGH SCORE MODEL =====
            out_model = self.shared_state["out_model"]
            out_vecnorm = self.shared_state["out_vecnorm"]
                
            if base_scenarios_passed:
                best_s = self.shared_state["best_success"]
                best_e = self.shared_state["best_error"]
                
                # Update high score if boss performance improved
                if boss_success > best_s or (boss_success == best_s and boss_error < best_e):
                    print(f"\n NEW HIGH SCORE FOUND : (Success: {boss_success:.2f}, Error: {boss_error:.2f})")
                    self.shared_state["best_success"] = boss_success
                    self.shared_state["best_error"] = boss_error
                    
                    # Persist the model and normalization stats
                    self.model.save(out_model)
                    self.train_env_to_save.save(out_vecnorm)
            else:
                print("\n Base requirements not met or insufficient score, model discarded.")

            print("------------------------------------------------------------\n")
            
        return True


def main():
    """
    Main training loop: 150 attempts to find a high-performing policy.
    
    For each attempt:
    1. Create a fresh multi-process training environment
    2. Load the base model and normalization stats
    3. Train for additional_timesteps with periodic evaluation
    4. Only save models that beat the current best
    
    The "casino" metaphor: each attempt is a spin of the slot machine,
    hoping to hit a jackpot (high score) before running out of attempts.
    """
    log_dir = "logs_rl"
    os.makedirs(log_dir, exist_ok=True)

    # ===== LOAD CONFIGURATION FROM ENVIRONMENT VARIABLES =====
    input_model = os.getenv("USV_INPUT_MODEL", "usv_rl_brain_phase_dynamic_unpredictable")
    input_vecnorm = os.getenv("USV_INPUT_VECNORM", os.path.join(log_dir, "vecnormalize_phase_dynamic_unpredictable.pkl"))
    output_model = os.getenv("USV_OUTPUT_MODEL", "usv_rl_brain_phase_dynamic_final")
    output_vecnorm = os.getenv("USV_OUTPUT_VECNORM", os.path.join(log_dir, "vecnormalize_phase_dynamic_final.pkl"))
    
    train_stage = os.getenv("USV_TRAIN_STAGE", "phase_dynamic_final")
    additional_timesteps = int(os.getenv("USV_TOTAL_TIMESTEPS", "150000")) 
    device = os.getenv("USV_DEVICE", "cpu")
    lr_val = float(os.getenv("USV_LEARNING_RATE", "5e-5"))
    n_cores = 10  # Number of parallel environments for SubprocVecEnv

    print(f"🎰 Fortune Casino Program Launched (150 attempts × {additional_timesteps} steps) 🎰")
    print(f"Source: {input_model} | Destination: {output_model}")

    # ===== SHARED STATE TRACKING =====
    shared_state = {
        "best_success": 0.85,  # Minimum 85% boss success required to save model
        "best_error": 0.2,
        "miracle": False,  # Could be used to stop early if miracle occurs
        "out_model": output_model,
        "out_vecnorm": output_vecnorm
    }

    # ===== MAIN TRAINING LOOP =====
    for attempt in range(1, 151):
        print("\n" + "="*80)
        print(f"▶ ATTEMPT {attempt}/150 START")
        print("="*80)

        # 1. Create fresh multi-process environment loaded with source normalization
        env_fns = [make_env(train_stage) for _ in range(n_cores)]
        train_env = SubprocVecEnv(env_fns)
        
        # Load normalization statistics from source model
        train_env = VecNormalize.load(input_vecnorm, train_env)
        train_env.training = True
        train_env.norm_reward = True

        # Custom objects to enforce safety clip range
        custom_objects = {"clip_range": 0.1}

        # 2. Load base model and reset its learning rate
        model = PPO.load(input_model, env=train_env, device=device, custom_objects=custom_objects)
        model.learning_rate = lr_val
        model.lr_schedule = lambda _: lr_val

        # 3. Set up periodic evaluation callback
        callback = AutoCurriculumCallback(
            shared_state=shared_state, 
            attempt=attempt, 
            train_env_to_save=train_env,
            eval_freq=1000,  # Evaluate every 1000 calls (10,000 steps with 10 cores)
            n_eval_episodes=100
        )

        # 4. Train the model
        model.learn(
            total_timesteps=additional_timesteps, 
            reset_num_timesteps=False, 
            callback=callback
        )

        # Early exit if miracle condition met
        if shared_state["miracle"]:
            break

        # 5. Clean up resources for next attempt
        del model
        train_env.close()  # Critical: close worker processes before deleting
        del train_env

    print(f"\n Casino Finished. Best result saved to: {output_model}")


if __name__ == "__main__":
    main()
