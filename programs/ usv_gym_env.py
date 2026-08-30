import math
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from USV_dynamics_model import USVParameters, USVDynamics
from mpc_shield import PredictiveSafetyFilter


@dataclass
class Obstacle:
    """
    Generic obstacle model used by scenario generation and runtime updates.

    Attributes:
        x, y: Obstacle position in world coordinates.
        vx, vy: Obstacle velocity components in world coordinates.
        radius: Physical radius used in collision checks.
        mobile: Whether obstacle moves each simulation step.
        direction_change_interval: For stochastic mobile obstacles, number of
            steps between random heading changes.
        _steps_since_change: Internal counter for direction-change logic.
    """
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 1.0
    mobile: bool = False
    direction_change_interval: int = 0
    _steps_since_change: int = 0


class ShieldedUSVEnv(gym.Env):
    """
    USV environment with a predictive safety filter (shield) between RL action
    and applied control.

    Core idea:
      1) RL proposes an action in normalized motor space.
      2) MPC shield modifies it if needed for near-term safety.
      3) Dynamics integrates the safe action into next state.
      4) Reward encourages progress to goal while penalizing unsafe/inefficient behavior.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # --- Dynamics and safety components ---
        # USVParameters controls low-level physical constants used by both
        # dynamics and shield.
        self.params = USVParameters(pwm=400.0)
        self.dynamics = USVDynamics(self.params)
        self.shield = PredictiveSafetyFilter(self.params)

        # RL action: normalized left/right motor commands.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation shape is 18.
        # Features:
        # [dx_target, dy_target, rel_target_vx, rel_target_vy, psi, u, v, r,
        #  obs1_dx, obs1_dy, obs1_dist, obs1_rel_vx, obs1_rel_vy,
        #  obs2_dx, obs2_dy, obs2_dist, obs2_rel_vx, obs2_rel_vy]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

        # --- World and goal defaults ---
        self.target_x = 50.0
        self.target_y = 50.0
        self.target_vx = 0.0  # Used in moving_target scenario
        self.target_vy = 0.0
        self.world_min_x = -15.0
        self.world_max_x = 90.0
        self.world_min_y = -30.0
        self.world_max_y = 90.0

        # --- Episode controls ---
        self.max_steps = 1200
        self.current_step = 0

        # --- Safety / termination thresholds ---
        self.goal_tolerance = 2.5
        self.collision_distance = 2.0
        self.safe_distance = 6.0
        self.front_cone_half_angle = math.radians(45.0)

        # --- State ---
        # [x, y, psi, u, v, r]
        self.state = np.zeros(6, dtype=np.float32)
        self.prev_target_dist = None

        # Current scenario and obstacle set
        self.obstacles = []
        self.current_scenario = "free"

        # Sentinel distance used in observations when obstacle slots are empty
        self.far_obstacle_distance = 100.0

        # --- Reset randomization knobs ---
        # Purpose: reduce overfitting to deterministic initial conditions.
        self.randomize = True
        self.start_pos_jitter = 2.0
        self.start_psi_jitter = math.radians(30.0)
        self.obstacle_pos_jitter = 2.0
        self.obstacle_speed_jitter = 0.15
        self.target_pos_jitter = 3.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_target_dist = None

        # Initial vessel state
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        scenario = None
        curriculum_stage = "all"

        if options is not None:
            scenario = options.get("scenario", None)
            curriculum_stage = options.get("curriculum_stage", "all")

        # Build episode content (target, world bounds, obstacles)
        self._sample_scenario(scenario=scenario, curriculum_stage=curriculum_stage)

        # Randomize initial position after scenario creation
        if self.randomize:
            self.state[0] += random.uniform(-self.start_pos_jitter, self.start_pos_jitter)
            self.state[1] += random.uniform(-self.start_pos_jitter, self.start_pos_jitter)

        # Initialize heading toward target (plus optional jitter)
        dx = self.target_x - self.state[0]
        dy = self.target_y - self.state[1]
        base_psi = math.atan2(dy, dx)
        if self.randomize:
            base_psi += random.uniform(-self.start_psi_jitter, self.start_psi_jitter)
        self.state[2] = base_psi % (2 * np.pi)

        self.prev_target_dist = self._distance_to_target()
        return self._get_obs(), {}

    def _sample_scenario(self, scenario=None, curriculum_stage="all"):
        """
        Select and instantiate one scenario for this episode.

        If scenario is None, one is sampled from a stage-dependent pool.
        Each scenario configures:
          - obstacle set (static or mobile)
          - optional target dynamics
          - optional world bounds adaptation
        """
        if scenario is None:
            if curriculum_stage == "free_only":
                pool = ["free"]
            elif curriculum_stage == "phase_2":
                pool = ["free"] * 7 + ["static_single"] * 3
            elif curriculum_stage == "phase_3":
                pool = ["free"] * 4 + ["static_single"] * 3 + ["static_multi"] * 3
            else:
                pool = [
                    "free",
                    "static_single",
                    "static_multi",
                    "moving_cross",
                    "moving_follow",
                    "narrow_corridor",
                ]
            scenario = random.choice(pool)

        self.current_scenario = scenario
        self.obstacles = []

        # Reset target and world defaults
        self.target_x = 50.0
        self.target_y = 50.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.world_min_x = -15.0
        self.world_max_x = 90.0
        self.world_min_y = -30.0
        self.world_max_y = 90.0

        if scenario == "free":
            self.obstacles = []

        elif scenario == "static_single":
            self.obstacles = [
                Obstacle(x=20.0, y=10.0, radius=1.2, mobile=False),
            ]

        elif scenario == "static_multi":
            self.obstacles = [
                Obstacle(x=18.0, y=8.0, radius=1.0, mobile=False),
                Obstacle(x=28.0, y=18.0, radius=1.2, mobile=False),
                Obstacle(x=35.0, y=25.0, radius=1.0, mobile=False),
            ]

        elif scenario == "moving_cross":
            # Randomized crossing obstacle to avoid a single exploitable geometry.
            start_y = random.uniform(-8.0, 8.0) if self.randomize else -5.0
            vy = (
                random.choice([-1.0, 1.0]) * random.uniform(0.6, 1.0)
                if self.randomize
                else 0.8
            )
            self.obstacles = [
                Obstacle(x=25.0, y=start_y, vx=0.0, vy=vy, radius=1.0, mobile=True),
            ]

        elif scenario == "moving_follow":
            self.obstacles = [
                Obstacle(x=12.0, y=2.0, vx=0.2, vy=0.0, radius=1.0, mobile=True),
            ]

        elif scenario == "narrow_corridor":
            self.obstacles = [
                Obstacle(x=15.0, y=-4.0, radius=1.0, mobile=False),
                Obstacle(x=15.0, y=4.0, radius=1.0, mobile=False),
                Obstacle(x=25.0, y=-4.0, radius=1.0, mobile=False),
                Obstacle(x=25.0, y=4.0, radius=1.0, mobile=False),
            ]

        elif scenario == "moving_target":
            # Target moves slowly and bounces on boundaries.
            self.obstacles = []
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.05, 0.15)
            self.target_vx = speed * math.cos(angle)
            self.target_vy = speed * math.sin(angle)

        elif scenario == "fast_unpredictable_obstacle":
            # Mobile obstacle that periodically changes direction.
            speed = random.uniform(1.0, 1.5)
            angle = random.uniform(0, 2 * math.pi)
            interval = random.randint(30, 50)
            self.obstacles = [
                Obstacle(
                    x=25.0,
                    y=10.0,
                    vx=speed * math.cos(angle),
                    vy=speed * math.sin(angle),
                    radius=1.0,
                    mobile=True,
                    direction_change_interval=interval,
                    _steps_since_change=0,
                ),
            ]

        elif scenario == "mixed_dynamic":
            self.obstacles = [
                Obstacle(x=15.0, y=8.0, radius=1.2, mobile=False),
                Obstacle(x=30.0, y=15.0, vx=0.0, vy=0.6, radius=1.0, mobile=True),
            ]

        elif scenario == "variable_distance_world":
            # Goal distance and heading vary; world bounds adapt to remain feasible.
            self.obstacles = []
            base_angle = math.pi / 4
            dist = random.uniform(30.0, 100.0)
            angle_offset = random.uniform(-math.pi / 8, math.pi / 8)
            angle = base_angle + angle_offset
            self.target_x = dist * math.cos(angle)
            self.target_y = dist * math.sin(angle)

            margin = 15.0
            self.world_min_x = min(-5.0, self.target_x - margin)
            self.world_max_x = max(80.0, self.target_x + margin)
            self.world_min_y = min(-20.0, self.target_y - margin)
            self.world_max_y = max(80.0, self.target_y + margin)

        else:
            self.obstacles = [
                Obstacle(x=20.0, y=10.0, radius=1.2, mobile=False),
            ]

        # Scenario-level randomization
        if self.randomize:
            # Keep explicit target dynamics untouched for moving-target scenarios.
            if self.current_scenario not in ("moving_target", "variable_distance_world"):
                self.target_x += random.uniform(-self.target_pos_jitter, self.target_pos_jitter)
                self.target_y += random.uniform(-self.target_pos_jitter, self.target_pos_jitter)

            # Jitter obstacle positions and speed magnitudes.
            for obs in self.obstacles:
                obs.x += random.uniform(-self.obstacle_pos_jitter, self.obstacle_pos_jitter)
                obs.y += random.uniform(-self.obstacle_pos_jitter, self.obstacle_pos_jitter)
                if obs.mobile:
                    jitter = 1.0 + random.uniform(
                        -self.obstacle_speed_jitter, self.obstacle_speed_jitter
                    )
                    obs.vx *= jitter
                    obs.vy *= jitter

    def _get_obs(self):
        """
        Build observation from:
          - target relative position and relative velocity
          - vessel orientation/body velocities
          - two nearest obstacles (360°) in relative coordinates
        """
        dx_target = self.target_x - self.state[0]
        dy_target = self.target_y - self.state[1]

        psi = self.state[2]
        u = self.state[3]
        v = self.state[4]
        boat_vx = u * math.cos(psi) - v * math.sin(psi)
        boat_vy = u * math.sin(psi) + v * math.cos(psi)

        rel_target_vx = self.target_vx - boat_vx
        rel_target_vy = self.target_vy - boat_vy

        features = [
            dx_target,
            dy_target,
            rel_target_vx,
            rel_target_vy,
            self.state[2],
            self.state[3],
            self.state[4],
            self.state[5],
        ]

        slot1, slot2 = self._nearest_obstacles_360(k=2)
        for obs_x, obs_y, obs_vx, obs_vy, dist_obs in (slot1, slot2):
            if obs_x is None or obs_y is None or not np.isfinite(dist_obs):
                dx_obs = self.far_obstacle_distance
                dy_obs = self.far_obstacle_distance
                dist_obs = self.far_obstacle_distance
                rel_obs_vx = 0.0
                rel_obs_vy = 0.0
            else:
                dx_obs = obs_x - self.state[0]
                dy_obs = obs_y - self.state[1]
                rel_obs_vx = obs_vx - boat_vx
                rel_obs_vy = obs_vy - boat_vy

            features.extend([dx_obs, dy_obs, dist_obs, rel_obs_vx, rel_obs_vy])

        obs = np.array(features, dtype=np.float32)
        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=self.far_obstacle_distance,
            neginf=-self.far_obstacle_distance,
        )

    def _distance_to_target(self):
        return math.hypot(self.target_x - self.state[0], self.target_y - self.state[1])

    def _advance_obstacles(self):
        """
        Integrate mobile obstacle states and bounce them on world limits.
        For stochastic obstacles, periodically randomize direction while
        preserving speed magnitude.
        """
        for obstacle in self.obstacles:
            if not obstacle.mobile:
                continue

            if obstacle.direction_change_interval > 0:
                obstacle._steps_since_change += 1
                if obstacle._steps_since_change >= obstacle.direction_change_interval:
                    obstacle._steps_since_change = 0
                    speed = math.hypot(obstacle.vx, obstacle.vy)
                    new_angle = random.uniform(0, 2 * math.pi)
                    obstacle.vx = speed * math.cos(new_angle)
                    obstacle.vy = speed * math.sin(new_angle)

            obstacle.x += obstacle.vx
            obstacle.y += obstacle.vy

            if obstacle.x < self.world_min_x or obstacle.x > self.world_max_x:
                obstacle.vx *= -1.0
                obstacle.x = np.clip(obstacle.x, self.world_min_x, self.world_max_x)

            if obstacle.y < self.world_min_y or obstacle.y > self.world_max_y:
                obstacle.vy *= -1.0
                obstacle.y = np.clip(obstacle.y, self.world_min_y, self.world_max_y)

    def _advance_target(self):
        """
        Integrate moving target and bounce it inside a margin from boundaries.
        Margin keeps shield/target interactions away from edge singularities.
        """
        if self.target_vx == 0.0 and self.target_vy == 0.0:
            return

        self.target_x += self.target_vx
        self.target_y += self.target_vy

        margin = 4.5
        if self.target_x < self.world_min_x + margin or self.target_x > self.world_max_x - margin:
            self.target_vx *= -1.0
            self.target_x = np.clip(
                self.target_x, self.world_min_x + margin, self.world_max_x - margin
            )
        if self.target_y < self.world_min_y + margin or self.target_y > self.world_max_y - margin:
            self.target_vy *= -1.0
            self.target_y = np.clip(
                self.target_y, self.world_min_y + margin, self.world_max_y - margin
            )

    def _get_threat_obstacles(self, k=2):
        """
        Return k closest obstacles inside the forward threat cone.
        Used by helper logic; filled with sentinels if fewer than k.
        """
        boat_x = self.state[0]
        boat_y = self.state[1]
        psi = self.state[2]

        dir_x = math.cos(psi)
        dir_y = math.sin(psi)

        candidates = []
        for obstacle in self.obstacles:
            vec_x = obstacle.x - boat_x
            vec_y = obstacle.y - boat_y

            dist = math.hypot(vec_x, vec_y)
            if dist <= 1e-6:
                continue

            if vec_x * dir_x + vec_y * dir_y <= 0.0:
                continue

            angle = math.atan2(vec_y, vec_x) - psi
            while angle > math.pi:
                angle -= 2.0 * math.pi
            while angle < -math.pi:
                angle += 2.0 * math.pi

            if abs(angle) > self.front_cone_half_angle:
                continue

            candidates.append((dist, obstacle))

        candidates.sort(key=lambda c: c[0])

        result = []
        for dist, o in candidates[:k]:
            result.append((o.x, o.y, o.vx, o.vy, dist))
        while len(result) < k:
            result.append((None, None, None, None, float("inf")))
        return result

    def _get_threat_obstacle(self):
        return self._get_threat_obstacles(k=1)[0]

    def _nearest_obstacle_any_direction(self):
        """
        Closest obstacle in 360° (no front-cone filter).
        Kept for diagnostics and alternative sensing strategies.
        """
        boat_x = self.state[0]
        boat_y = self.state[1]
        best = None
        best_dist = float("inf")
        for o in self.obstacles:
            d = math.hypot(o.x - boat_x, o.y - boat_y)
            if 1e-6 < d < best_dist:
                best_dist = d
                best = o
        if best is None:
            return (None, None, None, None, float("inf"))
        return (best.x, best.y, best.vx, best.vy, best_dist)

    def _nearest_obstacles_360(self, k=2):
        """
        k closest obstacles in full 360°.
        Distinct slots, sorted by increasing distance, sentinel-filled if needed.
        """
        bx, by = self.state[0], self.state[1]
        cands = []
        for o in self.obstacles:
            d = math.hypot(o.x - bx, o.y - by)
            if d > 1e-6:
                cands.append((d, o))
        cands.sort(key=lambda c: c[0])
        res = [(o.x, o.y, o.vx, o.vy, d) for d, o in cands[:k]]
        while len(res) < k:
            res.append((None, None, None, None, float("inf")))
        return res

    def _distance_to_any_obstacle(self):
        if not self.obstacles:
            return float("inf")

        boat_x = self.state[0]
        boat_y = self.state[1]

        return min(
            math.hypot(obstacle.x - boat_x, obstacle.y - boat_y)
            for obstacle in self.obstacles
        )

    def _distance_to_threatening_obstacle(self):
        """
        Distance to closest obstacle considered threatening:
          - ahead of vessel, OR
          - approaching relative to vessel velocity.

        This avoids over-penalizing obstacles already passed and diverging.
        """
        if not self.obstacles:
            return float("inf")
        bx, by = self.state[0], self.state[1]
        psi = self.state[2]
        dir_x, dir_y = math.cos(psi), math.sin(psi)
        u, v = self.state[3], self.state[4]
        boat_vx = u * math.cos(psi) - v * math.sin(psi)
        boat_vy = u * math.sin(psi) + v * math.cos(psi)
        best = float("inf")
        for o in self.obstacles:
            vec_x = o.x - bx
            vec_y = o.y - by
            d = math.hypot(vec_x, vec_y)
            if d <= 1e-6:
                return 0.0
            ahead = (vec_x * dir_x + vec_y * dir_y) > 0.0
            approaching = (vec_x * (o.vx - boat_vx) + vec_y * (o.vy - boat_vy)) < 0.0
            if ahead or approaching:
                best = min(best, d)
        return best

    def _collision_with_any_obstacle(self):
        boat_x = self.state[0]
        boat_y = self.state[1]

        for obstacle in self.obstacles:
            dist = math.hypot(obstacle.x - boat_x, obstacle.y - boat_y)
            if dist <= self.collision_distance + obstacle.radius:
                return True

        return False

    def step(self, action_rl):
        """
        One simulation step:
          1) Advance exogenous dynamics (obstacles + moving target)
          2) Build nearest threat proxy for shield (obstacle or wall)
          3) Shield action and integrate USV dynamics
          4) Compute reward and termination flags
        """
        self.current_step += 1

        action_rl = np.asarray(action_rl, dtype=np.float32)
        action_rl = np.clip(action_rl, self.action_space.low, self.action_space.high)

        self._advance_obstacles()
        self._advance_target()

        obs_x, obs_y, obs_vx, obs_vy, dist_obs = self._nearest_obstacles_360(k=1)[0]

        # Treat world boundaries as virtual solid hazards for shield input.
        dist_left = self.state[0] - self.world_min_x
        dist_right = self.world_max_x - self.state[0]
        dist_bottom = self.state[1] - self.world_min_y
        dist_top = self.world_max_y - self.state[1]

        min_wall_dist = min(dist_left, dist_right, dist_bottom, dist_top)

        if min_wall_dist < dist_obs and min_wall_dist < self.safe_distance:
            shield_obs_vx = 0.0
            shield_obs_vy = 0.0

            if min_wall_dist == dist_left:
                shield_obs_x = self.world_min_x
                shield_obs_y = self.state[1]
            elif min_wall_dist == dist_right:
                shield_obs_x = self.world_max_x
                shield_obs_y = self.state[1]
            elif min_wall_dist == dist_bottom:
                shield_obs_x = self.state[0]
                shield_obs_y = self.world_min_y
            else:
                shield_obs_x = self.state[0]
                shield_obs_y = self.world_max_y

        else:
            if obs_x is not None and obs_y is not None and dist_obs < self.safe_distance:
                shield_obs_x = obs_x
                shield_obs_y = obs_y
                shield_obs_vx = obs_vx if obs_vx is not None else 0.0
                shield_obs_vy = obs_vy if obs_vy is not None else 0.0
            else:
                # No immediate hazard: place proxy obstacle far ahead.
                shield_obs_x = self.state[0] + self.far_obstacle_distance * math.cos(self.state[2])
                shield_obs_y = self.state[1] + self.far_obstacle_distance * math.sin(self.state[2])
                shield_obs_vx = 0.0
                shield_obs_vy = 0.0

        action_safe = self.shield.get_safe_action(
            self.state,
            action_rl,
            shield_obs_x,
            shield_obs_y,
            shield_obs_vx,
            shield_obs_vy,
        )
        action_safe = np.asarray(action_safe, dtype=np.float32)
        action_safe = np.clip(action_safe, self.action_space.low, self.action_space.high)

        self.state, _ = self.dynamics.step(self.state, action_safe)
        self.state = np.asarray(self.state, dtype=np.float32)

        dist_to_target = self._distance_to_target()

        # Reward coefficients (kept identical across scenarios in final version).
        is_free = self.current_scenario == "free"
        if is_free:
            distance_coeff = 0.002
            progress_coeff = 1.0
            shield_coeff = 0.1
            success_bonus = 500.0
        else:
            distance_coeff = 0.002
            progress_coeff = 1.0
            shield_coeff = 0.1
            success_bonus = 500.0

        reward = 0.0

        # Distance-to-go penalty
        reward -= distance_coeff * dist_to_target

        # Orientation alignment bonus when progressing toward target
        desired_psi = math.atan2(self.target_y - self.state[1], self.target_x - self.state[0])
        angle_err = (desired_psi - self.state[2] + math.pi) % (2.0 * math.pi) - math.pi
        reward += 0.1 * math.cos(angle_err) * (dist_to_target < self.prev_target_dist)

        # Soft boundary penalty near world edges
        margin = 5.0
        if (
            self.state[0] < self.world_min_x + margin
            or self.state[0] > self.world_max_x - margin
            or self.state[1] < self.world_min_y + margin
            or self.state[1] > self.world_max_y - margin
        ):
            reward -= 2.0

        # Progress reward
        progress = 0.0
        if self.prev_target_dist is not None:
            progress = self.prev_target_dist - dist_to_target
            reward += progress_coeff * progress

        # Time penalty per step
        reward -= 0.1
        self.prev_target_dist = dist_to_target

        # Penalize shield intervention magnitude
        shield_intervention = np.linalg.norm(action_safe - action_rl)
        if shield_intervention > 0.1:
            reward -= shield_coeff * shield_intervention

        # Proximity penalty to threatening obstacles
        dist_threat = self._distance_to_threatening_obstacle()
        if dist_threat < 5.0:
            reward -= (5.0 - dist_threat) * 0.2

        # Terminal events
        if self._collision_with_any_obstacle():
            reward -= 250.0
            info = {"event": "collision", "scenario": self.current_scenario}
            return self._get_obs(), reward, True, False, info

        if dist_to_target < self.goal_tolerance:
            reward += success_bonus
            info = {"event": "goal", "scenario": self.current_scenario}
            return self._get_obs(), reward, True, False, info

        out_of_bounds = (
            self.state[0] < self.world_min_x
            or self.state[0] > self.world_max_x
            or self.state[1] < self.world_min_y
            or self.state[1] > self.world_max_y
        )
        if out_of_bounds:
            reward -= 400.0
            info = {"event": "out_of_bounds", "scenario": self.current_scenario}
            return self._get_obs(), reward, True, False, info

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "event": "timeout" if truncated else "running",
            "scenario": self.current_scenario,
            "progress": float(progress),
            "dist_to_target": float(dist_to_target),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass


class CurriculumUSVEnv(ShieldedUSVEnv):
    """
    Wrapper environment that controls scenario distribution across training phases.
    """
    def __init__(self, stage="free_only"):
        super().__init__()
        self.total_env_steps = 0
        self.stage = stage

    def set_stage(self, new_stage):
        """
        External hook to force curriculum progression (e.g., callback-driven).
        """
        self.stage = new_stage
        print(f"\n[ENVIRONMENT] Stage switched to: {self.stage}\n")

    def _scenario_pool(self):
        """
        Return stage-dependent weighted scenario pool used by reset() when no
        explicit scenario is provided.
        """
        if self.stage == "free_only":
            return ["free"]

        if self.stage == "phase_2":
            return ["free"] * 7 + ["static_single"] * 3

        if self.stage == "phase_3":
            return ["free"] * 5 + ["static_single"] * 4 + ["static_multi"] * 1

        if self.stage == "phase_moving_follow":
            return (
                ["free"] * 6
                + ["static_single"] * 3
                + ["static_multi"] * 3
                + ["moving_follow"] * 3
            )

        if self.stage == "phase_moving_cross":
            return (
                ["free"] * 10
                + ["static_single"] * 5
                + ["static_multi"] * 5
                + ["moving_follow"] * 5
                + ["moving_cross"] * 2
            )

        if self.stage == "all":
            return (
                ["free"] + ["static_single"] + ["static_multi"]
                + ["moving_follow"] + ["narrow_corridor"] + ["moving_cross"]
            )

        if self.stage == "phase_dynamic_1":
            return (
                ["free"] * 3 + ["static_single"] * 2 + ["static_multi"] * 4
                + ["moving_follow"] * 2 + ["narrow_corridor"] * 3
                + ["moving_cross"] * 3 + ["moving_target"] * 3
            )

        if self.stage == "phase_dynamic_2":
            return (
                ["free"] * 2 + ["static_single"] * 2 + ["static_multi"] * 4
                + ["moving_cross"] * 2 + ["moving_follow"] * 2 + ["narrow_corridor"] * 2
                + ["moving_target"] * 2 + ["fast_unpredictable_obstacle"] * 4
            )

        if self.stage == "phase_dynamic_3":
            return (
                ["free"] * 2 + ["static_single"] * 2 + ["static_multi"] * 3
                + ["moving_cross"] * 2 + ["moving_follow"] * 2 + ["narrow_corridor"] * 2
                + ["moving_target"] * 3 + ["variable_distance_world"] * 3
                + ["fast_unpredictable_obstacle"] * 3 + ["mixed_dynamic"] * 3
            )

        if self.stage == "phase_dynamic":
            return [
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

        # Automatic fallback curriculum based on total steps.
        if self.total_env_steps < 100_000:
            return ["free"]
        if self.total_env_steps < 200_000:
            return ["free", "static_single"]
        if self.total_env_steps < 300_000:
            return ["free", "static_single", "static_multi"]
        return [
            "free",
            "static_single",
            "static_multi",
            "moving_cross",
            "moving_follow",
            "narrow_corridor",
        ]

    def reset(self, seed=None, options=None):
        options = {} if options is None else dict(options)
        if "scenario" not in options:
            options["scenario"] = random.choice(self._scenario_pool())
        return super().reset(seed=seed, options=options)

    def step(self, action_rl):
        obs, reward, terminated, truncated, info = super().step(action_rl)
        self.total_env_steps += 1
        return obs, reward, terminated, truncated, info
