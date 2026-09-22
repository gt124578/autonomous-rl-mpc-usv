import os

from stable_baselines3 import PPO

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

NUM_EPISODES = 100  # Evaluate each scenario 100 times to compute stable statistics.


def main():
    env = ShieldedUSVEnv()

    print("\n" + "=" * 50)
    print("STARTING UNIT MODEL EVALUATION")
    print("=" * 50)

    for scenario in SCENARIOS_TO_TEST:
        model_path = f"unit_test_models/model_{scenario}.zip"

        if not os.path.exists(model_path):
            print(f"\nModel not found for {scenario} ({model_path}). Skipping.")
            continue

        print(f"\nEvaluating scenario: {scenario.upper()}")
        model = PPO.load(model_path)

        # Count the terminal events observed during evaluation.
        results = {
            "goal": 0,
            "collision": 0,
            "out_of_bounds": 0,
            "timeout": 0,
        }

        for _ in range(NUM_EPISODES):
            # Force the selected scenario for this evaluation episode.
            obs, _ = env.reset(options={"scenario": scenario})
            done = False

            while not done:
                # Use deterministic actions to evaluate the learned policy without exploration noise.
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if done:
                    # Read the termination event to classify the episode outcome.
                    event = info.get("event", "unknown")
                    if event in results:
                        results[event] += 1

        # Compute the success rate.
        success_rate = results["goal"] / NUM_EPISODES

        # Display the evaluation summary.
        print(f"Results over {NUM_EPISODES} episodes:")
        print(f"  Success (Goal)       : {results['goal']} ({success_rate * 100:.1f}%)")
        print(f"  Collisions           : {results['collision']} ({results['collision'] / NUM_EPISODES * 100:.1f}%)")
        print(f"  Out of bounds        : {results['out_of_bounds']} ({results['out_of_bounds'] / NUM_EPISODES * 100:.1f}%)")
        print(f"  Timeout              : {results['timeout']} ({results['timeout'] / NUM_EPISODES * 100:.1f}%)")

        # Warn if the target success threshold is not reached.
        if scenario == "moving_cross" and success_rate < 0.85:
            print(f"WARNING: The 0.85 target was not reached for {scenario}.")
        elif scenario != "moving_cross" and success_rate < 0.98:
            print(f"WARNING: The 0.98 target was not reached for {scenario}.")


if __name__ == "__main__":
    main()
