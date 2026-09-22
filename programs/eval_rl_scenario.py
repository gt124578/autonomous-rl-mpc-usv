import argparse
import importlib
import os
import sys

import numpy as np

# Compatibility layer for models saved with NumPy 2.x when running in a NumPy < 2 environment.
# In NumPy >= 2, numpy._core exists natively and must not be overwritten.
try:
    importlib.import_module("numpy._core")
except ImportError:
    import numpy.core as _np_core

    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from usv_gym_env import CurriculumUSVEnv


DEFAULT_SCENARIOS_6 = [
    "free",
    "static_single",
    "static_multi",
    "moving_cross",
    "moving_follow",
    "narrow_corridor",
]

DEFAULT_SCENARIOS_10 = DEFAULT_SCENARIOS_6 + [
    "moving_target",
    "fast_unpredictable_obstacle",
    "mixed_dynamic",
    "variable_distance_world",
]


def infer_vecnorm_path(model_path: str) -> str:
    """Infer the VecNormalize file path from the model filename."""
    name = os.path.basename(model_path)
    if name.endswith(".zip"):
        name = name[:-4]

    if name.endswith("_all_v1"):
        return "logs_rl/vecnormalize_all_v1.pkl"
    if name.endswith("_all"):
        return "logs_rl/vecnormalize_all.pkl"
    if name.endswith("_dynamic"):
        return "logs_rl/vecnormalize_dynamic.pkl"
    return "logs_rl/vecnormalize.pkl"


class ScenarioLockedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario: str, stage: str = "all"):
        super().__init__()
        self.scenario = scenario
        self.base_env = CurriculumUSVEnv(stage=stage)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space

    def reset(self, seed=None, options=None):
        """Reset the environment while forcing the requested scenario."""
        opts = {} if options is None else dict(options)
        opts["scenario"] = self.scenario
        obs, info = self.base_env.reset(seed=seed, options=opts)

        # Ensure that the underlying environment respected the locked scenario.
        if getattr(self.base_env, "current_scenario", None) != self.scenario:
            raise RuntimeError(
                f"Scenario lock failed: expected={self.scenario}, "
                f"got={getattr(self.base_env, 'current_scenario', None)}"
            )

        return obs, info

    def step(self, action):
        return self.base_env.step(action)

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


def make_env(scenario: str, stage: str):
    """Create a factory function compatible with DummyVecEnv."""
    def _init():
        return ScenarioLockedEnv(scenario=scenario, stage=stage)

    return _init


def evaluate_scenario(model, vecnorm_path, scenario, episodes, stage):
    """Evaluate a trained policy on a single locked scenario."""
    env = DummyVecEnv([make_env(scenario, stage)])

    try:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    except Exception as e:
        print(f"[WARN] VecNormalize not loaded for {scenario}: {e}")

    metrics = {
        "successes": 0,
        "collisions": 0,
        "timeouts": 0,
        "out_of_bounds": 0,
        "unknown": 0,
    }

    total_reward = 0.0
    total_length = 0

    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])

            ep_reward += float(reward[0])
            ep_length += 1
            last_info = infos[0] if infos else {}

        total_reward += ep_reward
        total_length += ep_length

        outcome = last_info.get("outcome", "unknown")
        if outcome == "success":
            metrics["successes"] += 1
        elif outcome == "collision":
            metrics["collisions"] += 1
        elif outcome == "timeout":
            metrics["timeouts"] += 1
        elif outcome == "out_of_bounds":
            metrics["out_of_bounds"] += 1
        else:
            metrics["unknown"] += 1

    avg_reward = total_reward / max(episodes, 1)
    avg_length = total_length / max(episodes, 1)

    return {
        "scenario": scenario,
        "episodes": episodes,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vecnorm-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS_6)
    args = parser.parse_args()

    model = PPO.load(args.model_path)

    vecnorm_path = args.vecnorm_path or infer_vecnorm_path(args.model_path)

    results = []
    for scenario in args.scenarios:
        result = evaluate_scenario(
            model=model,
            vecnorm_path=vecnorm_path,
            scenario=scenario,
            episodes=args.episodes,
            stage=args.stage,
        )
        results.append(result)

        print(
            f"[{scenario}] "
            f"success={result['successes']}/{result['episodes']}, "
            f"collision={result['collisions']}, "
            f"timeout={result['timeouts']}, "
            f"oob={result['out_of_bounds']}, "
            f"unknown={result['unknown']}, "
            f"avg_reward={result['avg_reward']:.3f}, "
            f"avg_length={result['avg_length']:.2f}"
        )

    return results


if __name__ == "__main__":
    main()
