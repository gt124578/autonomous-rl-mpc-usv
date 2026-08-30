import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from usv_gym_env import ShieldedUSVEnv


SCENARIOS_TO_TEST = [
     "free",
     "static_single",
     "static_multi",
     "moving_cross",
     "moving_follow",
     "narrow_corridor",
     "moving_target",
     "fast_unpredictable_obstacle",
     "mixed_dynamic",
     "variable_distance_world",
]

TIMESTEPS_PER_SCENARIO = 1_500_000
NUM_CORES = 12  # Number of parallel processes


def make_env(scenario_name, rank, seed=0):
    """
    Create an environment factory for multiprocessing.

    Each worker must instantiate its own environment instance and use a distinct seed.
    """
    def _init():
        env = ShieldedUSVEnv()

        # Override reset so that the selected scenario is always enforced.
        original_reset = env.reset

        def custom_reset(seed=None, options=None):
            if options is None:
                options = {}
            options["scenario"] = scenario_name
            return original_reset(seed=seed, options=options)

        env.reset = custom_reset

        # Offset the seed by the worker rank to diversify randomness across processes.
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    os.makedirs("unit_test_models", exist_ok=True)

    for scenario in SCENARIOS_TO_TEST:
        print("\n=========================================================")
        print(f"TRAINING WITH MULTIPLE CORES (x{NUM_CORES}): {scenario.upper()}")
        print("=========================================================")

        # Create one environment factory per worker process.
        env_fns = [make_env(scenario, i) for i in range(NUM_CORES)]

        # Launch the environments in separate processes.
        vec_env = SubprocVecEnv(env_fns)

        # Initialize the PPO agent.
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
        )

        # Train the model for the current scenario.
        model.learn(total_timesteps=TIMESTEPS_PER_SCENARIO)

        # Save the trained model.
        save_path = f"unit_test_models/model_{scenario}"
        model.save(save_path)
        print(f"Model saved: {save_path}.zip")

        # Free resources before moving to the next scenario.
        vec_env.close()
        del model
        del vec_env


if __name__ == "__main__":
    # This guard is required when using SubprocVecEnv.
    main()
