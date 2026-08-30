#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from std_msgs.msg import Float32
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan

from mpc_shield import PredictiveSafetyFilter
from USV_dynamics_model import USVParameters

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces


def quaternion_to_yaw(x, y, z, w):
    """Convert a quaternion orientation into a yaw angle."""
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    return np.arctan2(siny_cosp, cosy_cosp)


class USVController:
    def __init__(self):
        rospy.init_node("usv_safe_rl_controller", anonymous=True)

        # ---------------------------------------------------------------------
        # System configuration
        # ---------------------------------------------------------------------
        self.boat_name = rospy.get_param("~boat_name", "myboat")

        # Keep the same propulsion scale as the RL training environment.
        # This is essential because the MPC shield relies on the same dynamics
        # model as the policy learned during training.
        self.params = USVParameters(pwm=400.0)

        # The controller loop must follow the same timing convention as training.
        # If the loop runs too fast or too slow, the learned control policy no
        # longer acts on the same physical time scale.
        self.rate_hz = 10

        # Map boundaries used by the deployment environment.
        self.x_min, self.x_max = -540.0, -450.0
        self.y_min, self.y_max = -50.0, 50.0
        self.safe_distance = 2.0

        # Allow reverse motion because the RL policy was trained on a symmetric action space.
        self.cmd_min, self.cmd_max = -1.0, 1.0

        # ---------------------------------------------------------------------
        # State variables updated from Gazebo
        # ---------------------------------------------------------------------
        self.boat_x = 0.0
        self.boat_y = 0.0
        self.psi = 0.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0
        self.state_received = False

        # Keep the two closest perceived obstacles.
        self.closest_obstacles = [
            {"dist": 100.0, "x": 1000.0, "y": 1000.0, "vx": 0.0, "vy": 0.0}
            for _ in range(2)
        ]

        # ---------------------------------------------------------------------
        # Lidar perception pipeline
        # ---------------------------------------------------------------------
        # The controller does not rely only on raw ModelStates for obstacle
        # handling. Instead, it reconstructs obstacles from three lidar layers:
        # water, keel, and mast. This makes the perception more robust and more
        # representative of the vessel geometry.
        self._lidar_clusters = {"water": [], "keel": [], "mast": []}
        self._tracked_obstacles = []

        # Clustering and tracking parameters.
        self.lidar_cluster_gap = 1.5
        self.lidar_merge_radius = 2.5
        self.lidar_assoc_radius = 3.0
        self.lidar_obstacle_timeout = 2.0
        self.lidar_self_hit_dist = 1.5
        self.lidar_max_obstacle_speed = 2.0

        # ---------------------------------------------------------------------
        # ROS I/O
        # ---------------------------------------------------------------------
        self.pub_left = rospy.Publisher("/myboat/thrusters/left_thrust_cmd", Float32, queue_size=10)
        self.pub_right = rospy.Publisher("/myboat/thrusters/right_thrust_cmd", Float32, queue_size=10)

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_callback)
        rospy.Subscriber("/myboat/sensors/lidar/scan_water", LaserScan, self.lidar_water_callback)
        rospy.Subscriber("/myboat/sensors/lidar/scan_keel", LaserScan, self.lidar_keel_callback)
        rospy.Subscriber("/myboat/sensors/lidar/scan_mast", LaserScan, self.lidar_mast_callback)

        # ---------------------------------------------------------------------
        # MPC safety shield
        # ---------------------------------------------------------------------
        # The RL policy proposes a control action, and the MPC shield verifies that
        # the action remains safe with respect to predicted vessel-obstacle geometry.
        self.shield = PredictiveSafetyFilter(self.params)

        # Persistent waypoint used to reduce oscillatory steering behavior.
        self.current_waypoint = None
        self.waypoint_set_time = 0.0

        # ---------------------------------------------------------------------
        # RL model loading
        # ---------------------------------------------------------------------
        rospy.loginfo("Loading RL model and normalization statistics...")
        model_path = "usv_rl_brain_final.zip"
        vecnorm_path = "vecnormalize_final.pkl"

        self.model = PPO.load(model_path, device="cpu")

        # Dummy environment used only to load observation normalization statistics.
        class DummyUSVEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                )

        dummy_env = DummyVecEnv([lambda: DummyUSVEnv()])
        self.vec_env = VecNormalize.load(vecnorm_path, dummy_env)
        self.vec_env.training = False
        self.vec_env.norm_reward = False
        rospy.loginfo("RL model loaded successfully.")

        # Navigation target used by the deployment controller.
        self.target_x = -65.0
        self.target_y = 100.0

    def state_callback(self, msg: ModelStates):
        try:
            idx = msg.name.index(self.boat_name)
            self.boat_x = float(msg.pose[idx].position.x)
            self.boat_y = float(msg.pose[idx].position.y)

            # Convert Gazebo quaternion orientation to yaw.
            raw_psi = quaternion_to_yaw(
                msg.pose[idx].orientation.x,
                msg.pose[idx].orientation.y,
                msg.pose[idx].orientation.z,
                msg.pose[idx].orientation.w,
            )

            # Match the heading convention used during training.
            self.psi = raw_psi % (2 * np.pi)

            x_dot = msg.twist[idx].linear.x
            y_dot = msg.twist[idx].linear.y
            self.r = msg.twist[idx].angular.z

            # Convert world-frame velocities into body-frame velocities.
            self.u = x_dot * np.cos(self.psi) + y_dot * np.sin(self.psi)
            self.v = -x_dot * np.sin(self.psi) + y_dot * np.cos(self.psi)

            self.state_received = True

            # A large lateral velocity can indicate a mismatch between the Gazebo
            # model and the expected motion convention.
            # rospy.loginfo_throttle(1.0, f"DEBUG -> u: {self.u:.2f}, v: {self.v:.2f}, psi: {self.psi:.2f}")

        except ValueError:
            rospy.logwarn_throttle(1.0, f"Model '{self.boat_name}' not found in Gazebo.")

    def _cluster_scan(self, msg: LaserScan):
        """
        Group valid lidar rays into obstacle clusters.

        The goal is to convert dense range measurements into compact obstacle
        representations that can be reused by the RL observation builder and by
        the safety shield.
        """
        if not self.state_received:
            return []

        min_valid_dist = max(msg.range_min, self.lidar_self_hit_dist)
        n = len(msg.ranges)

        valid = [
            r if (min_valid_dist <= r <= msg.range_max and not math.isinf(r) and not math.isnan(r)) else None
            for r in msg.ranges
        ]

        clusters = []
        current = []

        for i in range(n):
            r = valid[i]
            if r is None:
                if current:
                    clusters.append(current)
                    current = []
                continue

            if current and abs(r - valid[current[-1]]) > self.lidar_cluster_gap:
                clusters.append(current)
                current = []

            current.append(i)

        if current:
            clusters.append(current)

        # Merge the first and last clusters if they correspond to the same 360-degree obstacle.
        if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == n - 1:
            if abs(valid[clusters[-1][-1]] - valid[clusters[0][0]]) <= self.lidar_cluster_gap:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        detections = []
        boat_x, boat_y, psi = self.boat_x, self.boat_y, self.psi

        for cluster in clusters:
            best_i = min(cluster, key=lambda i: valid[i])
            dist = valid[best_i]
            angle = msg.angle_min + best_i * msg.angle_increment
            global_angle = psi + angle
            obs_x = boat_x + dist * math.cos(global_angle)
            obs_y = boat_y + dist * math.sin(global_angle)
            detections.append((dist, obs_x, obs_y))

        return detections

    def lidar_water_callback(self, msg: LaserScan):
        self._lidar_clusters["water"] = self._cluster_scan(msg)
        self._update_obstacles()

    def lidar_keel_callback(self, msg: LaserScan):
        self._lidar_clusters["keel"] = self._cluster_scan(msg)
        self._update_obstacles()

    def lidar_mast_callback(self, msg: LaserScan):
        self._lidar_clusters["mast"] = self._cluster_scan(msg)
        self._update_obstacles()

    def _fuse_lidar_layers(self):
        """
        Merge detections from the three lidar layers.

        This step handles the fact that the same physical obstacle may appear
        only on one of the three scans depending on the obstacle height.
        """
        raw = []
        for layer_detections in self._lidar_clusters.values():
            raw.extend(layer_detections)

        fused = []
        used = [False] * len(raw)

        for i, (dist_i, x_i, y_i) in enumerate(raw):
            if used[i]:
                continue

            group = [(dist_i, x_i, y_i)]
            used[i] = True

            for j in range(i + 1, len(raw)):
                if used[j]:
                    continue
                dist_j, x_j, y_j = raw[j]
                if math.hypot(x_j - x_i, y_j - y_i) <= self.lidar_merge_radius:
                    group.append((dist_j, x_j, y_j))
                    used[j] = True

            fused.append(min(group, key=lambda d: d[0]))

        return fused

    def _track_obstacles(self, fused):
        """
        Track detections over time to estimate obstacle velocity.

        The lidar gives position only. Velocity is reconstructed from successive
        frames and then smoothed to reduce noise.
        """
        now = rospy.get_time()
        unmatched = list(range(len(fused)))

        for tracked in self._tracked_obstacles:
            best_j, best_dist = None, self.lidar_assoc_radius

            for j in unmatched:
                _, x_j, y_j = fused[j]
                d = math.hypot(x_j - tracked["x"], y_j - tracked["y"])
                if d < best_dist:
                    best_dist, best_j = d, j

            if best_j is not None:
                dist_j, x_j, y_j = fused[best_j]
                dt = max(now - tracked["last_seen"], 1e-3)

                # Smooth the estimated speed to avoid reacting to lidar jitter.
                raw_vx = np.clip(
                    (x_j - tracked["x"]) / dt,
                    -self.lidar_max_obstacle_speed,
                    self.lidar_max_obstacle_speed,
                )
                raw_vy = np.clip(
                    (y_j - tracked["y"]) / dt,
                    -self.lidar_max_obstacle_speed,
                    self.lidar_max_obstacle_speed,
                )

                tracked["vx"] = 0.5 * tracked["vx"] + 0.5 * raw_vx
                tracked["vy"] = 0.5 * tracked["vy"] + 0.5 * raw_vy
                tracked["x"], tracked["y"] = x_j, y_j
                tracked["dist"] = dist_j
                tracked["last_seen"] = now
                unmatched.remove(best_j)

        for j in unmatched:
            dist_j, x_j, y_j = fused[j]
            self._tracked_obstacles.append(
                {
                    "x": x_j,
                    "y": y_j,
                    "vx": 0.0,
                    "vy": 0.0,
                    "dist": dist_j,
                    "last_seen": now,
                }
            )

        # Remove stale detections that have not been seen for too long.
        self._tracked_obstacles = [
            o for o in self._tracked_obstacles
            if now - o["last_seen"] <= self.lidar_obstacle_timeout
        ]

    def _update_obstacles(self):
        """
        Convert fused lidar detections into the two closest obstacles.

        This is the compact obstacle representation used by the RL observation
        and by the MPC shield.
        """
        fused = self._fuse_lidar_layers()
        self._track_obstacles(fused)

        ordered = sorted(self._tracked_obstacles, key=lambda o: o["dist"])[:2]
        self.closest_obstacles = [
            {"dist": o["dist"], "x": o["x"], "y": o["y"], "vx": o["vx"], "vy": o["vy"]}
            for o in ordered
        ]

        while len(self.closest_obstacles) < 2:
            self.closest_obstacles.append(
                {
                    "dist": 100.0,
                    "x": self.boat_x + 1000.0,
                    "y": self.boat_y + 1000.0,
                    "vx": 0.0,
                    "vy": 0.0,
                }
            )

    def build_rl_observation(self):
        """
        Build the input observation expected by the RL policy.

        The observation is expressed in a transformed reference frame that matches
        the one used during training, which is necessary for deployment consistency.
        """
        fake_psi = math.pi / 4.0
        now = rospy.get_time()

        real_dx = self.target_x - self.boat_x
        real_dy = self.target_y - self.boat_y
        real_dist = math.hypot(real_dx, real_dy)
        real_angle = math.atan2(real_dy, real_dx)

        angle_diff = (real_angle - self.psi + math.pi) % (2 * math.pi) - math.pi

        # When the target becomes close and aligned, switch from waypoint steering
        # to direct target tracking.
        if real_dist < 15.0 and abs(angle_diff) < math.radians(50.0):
            wp_x = self.target_x
            wp_y = self.target_y
            self.current_waypoint = None
        else:
            is_behind = abs(angle_diff) > (math.pi / 2.0)

            need_new_waypoint = False

            if self.current_waypoint is None:
                need_new_waypoint = True
            else:
                dist_to_wp = math.hypot(
                    self.current_waypoint[0] - self.boat_x,
                    self.current_waypoint[1] - self.boat_y,
                )
                time_elapsed = now - self.waypoint_set_time

                if dist_to_wp < 10.0 or time_elapsed >= 3.0:
                    need_new_waypoint = True

            if need_new_waypoint:
                max_angle = math.radians(30.0)

                if is_behind:
                    # If the goal is behind the boat, create a lateral waypoint to
                    # force a turn without asking the policy to perform an unstable reversal.
                    waypoint_dist = 15.0
                    waypoint_local_angle = math.copysign(max_angle, angle_diff)
                else:
                    # If the goal is in front, keep the waypoint aligned with the target direction.
                    waypoint_dist = min(real_dist, 25.0)
                    waypoint_local_angle = np.clip(angle_diff, -max_angle, max_angle)

                waypoint_global_angle = self.psi + waypoint_local_angle
                wp_x = self.boat_x + waypoint_dist * math.cos(waypoint_global_angle)
                wp_y = self.boat_y + waypoint_dist * math.sin(waypoint_global_angle)

                self.current_waypoint = (wp_x, wp_y)
                self.waypoint_set_time = now

            wp_x, wp_y = self.current_waypoint

        # Convert the active waypoint into the same frame used by the trained RL policy.
        wp_dx = wp_x - self.boat_x
        wp_dy = wp_y - self.boat_y
        wp_dist = math.hypot(wp_dx, wp_dy)
        wp_angle = math.atan2(wp_dy, wp_dx)

        wp_local_angle = (wp_angle - self.psi + math.pi) % (2 * math.pi) - math.pi

        # Project the waypoint into the artificial training reference frame.
        fake_target_angle = fake_psi + wp_local_angle
        fake_dx_target = wp_dist * math.cos(fake_target_angle)
        fake_dy_target = wp_dist * math.sin(fake_target_angle)

        # Convert body-frame velocities to the same transformed reference frame.
        fake_boat_vx = self.u * math.cos(fake_psi) - self.v * math.sin(fake_psi)
        fake_boat_vy = self.u * math.sin(fake_psi) + self.v * math.cos(fake_psi)

        fake_rel_target_vx = 0.0 - fake_boat_vx
        fake_rel_target_vy = 0.0 - fake_boat_vy

        # Encode the two closest obstacles.
        obs_features = []
        for obs in self.closest_obstacles:
            # Hide obstacles that are too far away to be relevant for the policy.
            if obs["dist"] >= 99.0 or obs["dist"] > 5.0:
                fake_dx_obs = 100.0
                fake_dy_obs = 100.0
                dist_obs = 100.0
                fake_rel_obs_vx = 0.0
                fake_rel_obs_vy = 0.0
            else:
                obs_dx = obs["x"] - self.boat_x
                obs_dy = obs["y"] - self.boat_y
                dist_obs = obs["dist"]

                obs_angle = math.atan2(obs_dy, obs_dx)
                obs_angle_diff = (obs_angle - self.psi + math.pi) % (2 * math.pi) - math.pi

                fake_obs_angle = fake_psi + obs_angle_diff
                fake_dx_obs = dist_obs * math.cos(fake_obs_angle)
                fake_dy_obs = dist_obs * math.sin(fake_obs_angle)

                v_obs_u = obs["vx"] * math.cos(self.psi) + obs["vy"] * math.sin(self.psi)
                v_obs_v = -obs["vx"] * math.sin(self.psi) + obs["vy"] * math.cos(self.psi)

                fake_obs_vx = v_obs_u * math.cos(fake_psi) - v_obs_v * math.sin(fake_psi)
                fake_obs_vy = v_obs_u * math.sin(fake_psi) + v_obs_v * math.cos(fake_psi)

                fake_rel_obs_vx = fake_obs_vx - fake_boat_vx
                fake_rel_obs_vy = fake_obs_vy - fake_boat_vy

            obs_features.extend([
                fake_dx_obs,
                fake_dy_obs,
                dist_obs,
                fake_rel_obs_vx,
                fake_rel_obs_vy,
            ])

        # Assemble the final observation vector expected by the policy.
        obs_array = np.array(
            [
                fake_dx_target,
                fake_dy_target,
                fake_rel_target_vx,
                fake_rel_target_vy,
                fake_psi,
                self.u,
                self.v,
                self.r,
            ] + obs_features,
            dtype=np.float32,
        )

        # Remove invalid values before normalization.
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=100.0, neginf=-100.0)

        return self.vec_env.normalize_obs(obs_array)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("Hybrid control loop started.")

        while not rospy.is_shutdown():
            if not self.state_received:
                rate.sleep()
                continue

            # -----------------------------------------------------------------
            # 1. Absolute safety handling and emergency extraction logic
            # -----------------------------------------------------------------
            now = rospy.get_time()
            closest_obs = self.closest_obstacles[0]

            if not hasattr(self, "pivot_count"):
                self.pivot_count = 0
                self.last_pivot_dir = 0
                self.reverse_end_time = 0.0

            # Reset the emergency state once the boat is far enough from hazards.
            if closest_obs["dist"] >= 2.5:
                self.pivot_count = 0
                self.last_pivot_dir = 0

            # Maintain a forced reverse phase when the controller is already in recovery mode.
            if now < self.reverse_end_time:
                action_emergency = np.array([-1.0, -1.0], dtype=np.float32)
                rospy.logwarn_throttle(0.5, "Emergency reverse in progress.")
                self.pub_left.publish(Float32(float(action_emergency[1])))
                self.pub_right.publish(Float32(float(action_emergency[0])))
                rate.sleep()
                continue

            # Keep the extraction impulse active until the vessel is clearly free.
            if hasattr(self, "extraction_end_time") and now < self.extraction_end_time:
                obs0_angle = math.atan2(closest_obs["y"] - self.boat_y, closest_obs["x"] - self.boat_x)
                rel_angle_0 = (obs0_angle - self.psi + math.pi) % (2 * math.pi) - math.pi

                if closest_obs["dist"] < 2.5 and abs(rel_angle_0) < 0.78:
                    self.extraction_end_time = 0.0
                    self.pivot_count = 0
                    rospy.logwarn_throttle(0.5, "Extraction cancelled because a new obstacle appeared.")
                else:
                    self.pivot_count = 0
                    action_emergency = np.array([1.0, 1.0], dtype=np.float32)
                    rospy.loginfo_throttle(0.5, "Extraction impulse maintained.")
                    self.pub_left.publish(Float32(float(action_emergency[1])))
                    self.pub_right.publish(Float32(float(action_emergency[0])))
                    rate.sleep()
                    continue

            # Emergency collision handling when the closest obstacle becomes critical.
            if closest_obs["dist"] < 2.5:
                obs0_angle = math.atan2(closest_obs["y"] - self.boat_y, closest_obs["x"] - self.boat_x)
                rel_angle_0 = (obs0_angle - self.psi + math.pi) % (2 * math.pi) - math.pi

                rel_angle_1 = 0.0
                obs1 = self.closest_obstacles[1]
                if obs1["dist"] < 15.0:
                    obs1_angle = math.atan2(obs1["y"] - self.boat_y, obs1["x"] - self.boat_x)
                    rel_angle_1 = (obs1_angle - self.psi + math.pi) % (2 * math.pi) - math.pi

                danger_dir = rel_angle_0 + (rel_angle_1 * 0.6)

                # Step 1: break forward momentum if the vessel is still moving toward the obstacle.
                if self.u > 0.15:
                    action_emergency = np.array([-1.0, -1.0], dtype=np.float32)
                    self.last_pivot_dir = 0
                    rospy.logwarn_throttle(0.5, f"Emergency braking: obstacle at {closest_obs['dist']:.2f} m.")

                # Step 2: pivot away from the dangerous direction.
                elif abs(rel_angle_0) < 0.78:
                    current_pivot_dir = 1 if danger_dir > 0 else -1

                    if self.last_pivot_dir != 0:
                        # If the chosen pivot direction changes repeatedly, the controller is oscillating.
                        if current_pivot_dir != self.last_pivot_dir:
                            self.pivot_count += 1

                    self.last_pivot_dir = current_pivot_dir

                    if self.pivot_count >= 6:
                        rospy.logwarn("Stuck condition detected. Activating reverse recovery.")
                        self.reverse_end_time = now + 3.0
                        self.pivot_count = 0
                        action_emergency = np.array([-1.0, -1.0], dtype=np.float32)
                    else:
                        if danger_dir > 0:
                            action_emergency = np.array([-1.0, 1.0], dtype=np.float32)
                            rospy.logwarn_throttle(0.5, f"Emergency pivot right (oscillation {self.pivot_count}/6).")
                        else:
                            action_emergency = np.array([1.0, -1.0], dtype=np.float32)
                            rospy.logwarn_throttle(0.5, f"Emergency pivot left (oscillation {self.pivot_count}/6).")

                # Step 3: once the path is cleared, apply a short extraction burst.
                else:
                    self.extraction_end_time = now + 2.5
                    self.pivot_count = 0
                    self.last_pivot_dir = 0
                    action_emergency = np.array([1.0, 1.0], dtype=np.float32)
                    rospy.logwarn_throttle(0.5, "Path cleared, starting extraction impulse.")

                self.pub_left.publish(Float32(float(action_emergency[1])))
                self.pub_right.publish(Float32(float(action_emergency[0])))
                rate.sleep()
                continue

            # -----------------------------------------------------------------
            # 2. Clean stopping behavior near the target
            # -----------------------------------------------------------------
            dx_target = self.target_x - self.boat_x
            dy_target = self.target_y - self.boat_y
            dist = math.hypot(dx_target, dy_target)

            if dist < 6.0:
                # Instead of cutting the motors immediately and letting the boat drift,
                # apply an active brake when the residual forward speed is still high.
                if self.u > 0.15:
                    action_stop = np.array([-0.6, -0.6], dtype=np.float32)
                    rospy.loginfo_throttle(1.0, "Target is close, applying active braking.")
                else:
                    action_stop = np.array([0.0, 0.0], dtype=np.float32)
                    rospy.loginfo_throttle(1.0, f"Target reached and vessel stopped at {dist:.1f} m.")

                self.pub_left.publish(Float32(float(action_stop[1])))
                self.pub_right.publish(Float32(float(action_stop[0])))
                rate.sleep()
                continue

            # -----------------------------------------------------------------
            # 3. RL policy action selection
            # -----------------------------------------------------------------
            obs_norm = self.build_rl_observation()

            # Defensive check against invalid numerical values.
            if np.isnan(obs_norm).any() or np.isinf(obs_norm).any():
                rospy.logwarn_throttle(1.0, "Invalid values detected in observation.")
                obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=10.0, neginf=-10.0)

            action_rl, _ = self.model.predict(obs_norm, deterministic=True)

            # -----------------------------------------------------------------
            # 4. MPC shield filtering
            # -----------------------------------------------------------------
            current_state = np.array([self.boat_x, self.boat_y, self.psi, self.u, self.v, self.r])
            closest_obs = self.closest_obstacles[0]

            # The MPC shield takes over only when an obstacle is close enough to require
            # explicit safety checking.
            if closest_obs["dist"] < 3.0:
                shield_obs_x = closest_obs["x"]
                shield_obs_y = closest_obs["y"]
                shield_obs_vx = closest_obs["vx"]
                shield_obs_vy = closest_obs["vy"]
            else:
                # Use a far-away dummy obstacle so the RL policy can move freely.
                shield_obs_x = self.boat_x + 1000.0
                shield_obs_y = self.boat_y + 1000.0
                shield_obs_vx = 0.0
                shield_obs_vy = 0.0

            try:
                action_safe = self.shield.get_safe_action(
                    current_state,
                    action_rl,
                    shield_obs_x,
                    shield_obs_y,
                    shield_obs_vx,
                    shield_obs_vy,
                )
            except Exception as e:
                rospy.logwarn_throttle(1.0, f"MPC shield error: {e}")
                action_safe = action_rl

            # Soft-brake logic.
            # If the RL policy wants to move forward but the shield outputs a strong
            # reverse command, replace it with a neutral command to avoid harsh braking.
            if action_rl[0] > 0.0 and action_safe[0] < -0.5:
                action_safe = np.array([0.0, 0.0], dtype=np.float32)

            action_safe = np.clip(action_safe, self.cmd_min, self.cmd_max)

            # -----------------------------------------------------------------
            # 5. Thruster command publication
            # -----------------------------------------------------------------
            # The left/right thruster order is intentionally inverted because the
            # Gazebo plugin and the analytical model do not use the same convention.
            self.pub_left.publish(Float32(float(action_safe[1])))
            self.pub_right.publish(Float32(float(action_safe[0])))

            # Compact telemetry for debugging and report traceability.
            obs_log = ""
            for i, obs in enumerate(self.closest_obstacles):
                if obs["dist"] >= 99.0:
                    obs_log += f"[O{i}: EMPTY] "
                else:
                    obs_log += f"[O{i}:d={obs['dist']:.1f}m,x={obs['x']:.1f},y={obs['y']:.1f}] "

            telemetry = (
                f"T={dist:.1f}m | "
                f"Pos:({self.boat_x:.1f}, {self.boat_y:.1f}, psi={self.psi:.2f}) | "
                f"Cmd:RL[{action_rl[0]:.2f},{action_rl[1]:.2f}]->MPC[{action_safe[0]:.2f},{action_safe[1]:.2f}] | "
                f"Obs:{obs_log}"
            )
            rospy.loginfo_throttle(2.0, telemetry)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = USVController()
        node.run()
    except rospy.ROSInterruptException:
        pass
