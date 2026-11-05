"""Version 2 driving environment: complete a planned route like a real driver.

Compared with carla_env.py (v1):
- Each drive follows a planned route (through junctions) and is scored on
  how much of it the car completes, instead of how long it survives.
- The car aims for the road's speed limit and is penalised for speeding,
  jerky steering, throttle/brake flipping and hard cornering.
- Steering and pedals have realistic rates of change.
- It sees further (50 m LiDAR), the route ahead at 5/15/30 m, its own
  previous controls and whether it faces a red light.
- Waiting behind traffic or at a red light is not counted as "stuck".
"""

import math
import random

import carla
import numpy as np

from carla_env import (CarlaEnv, FIXED_DELTA, NUM_LIDAR_SECTORS, lidar_sectors,
                       wrap_angle)

# ---- Observation layout ---------------------------------------------------
LIDAR_RANGE_V2 = 50.0
LOOKAHEAD_M = (5.0, 15.0, 30.0)
NUM_INPUTS = NUM_LIDAR_SECTORS + 2 + 2 + len(LOOKAHEAD_M) + 2 + 1
NUM_OUTPUTS = 2

_DIRECTIONS = {0: "front", 90: "right", 180: "rear", 270: "left"}
INPUT_LABELS = (
    [f"ray {round(i * 360 / NUM_LIDAR_SECTORS):>3}°"
     + (f" {_DIRECTIONS[round(i * 360 / NUM_LIDAR_SECTORS)]}"
        if round(i * 360 / NUM_LIDAR_SECTORS) in _DIRECTIONS else "")
     for i in range(NUM_LIDAR_SECTORS)]
    + ["speed", "speed limit", "heading err", "lane offset"]
    + [f"route {int(d)} m" for d in LOOKAHEAD_M]
    + ["prev steer", "prev pedal", "red light"]
)

# ---- Driving / reward settings --------------------------------------------
ROUTE_STEP = 2.0            # metres between route waypoints
SPEED_NORM = 25.0           # m/s used to normalise speed inputs
MIN_TARGET, MAX_TARGET = 8.0, 25.0   # m/s clamp on the speed-limit target
MIN_AVG_SPEED = 3.0         # m/s; sets the time limit for a route (allows for red lights)
MAX_STEER_DELTA = 0.06      # per step (20 Hz): full lock from centre in ~0.8 s
MAX_PEDAL_DELTA = 0.10      # per step

COMPLETION_BONUS = 50.0
COLLISION_PENALTY = 50.0
OFF_ROUTE_PENALTY = 25.0
STUCK_PENALTY = 25.0
STEER_JERK_COST = 1.0       # per unit of steering change
PEDAL_JERK_COST = 0.3
OVERSPEED_COST = 0.02       # per m/s above 110% of the limit, per step
LATERAL_ACCEL_LIMIT = 3.0   # m/s^2, comfortable cornering for a passenger
LATERAL_ACCEL_COST = 0.02
LANE_CENTRE_COST = 0.02
SAFE_HEADWAY = 1.5          # seconds to the vehicle ahead
HEADWAY_COST = 0.05
RED_LIGHT_COST = 0.2        # per step moving towards a red light

STUCK_GRACE_STEPS = 80
STUCK_LIMIT_STEPS = 60      # 3 s not moving on a clear road
WAIT_LIMIT_STEPS = 1200     # 60 s waiting (red phases can be ~40 s) ends the drive, no penalty
BLOCKED_DISTANCE = 12.0     # m: something this close ahead counts as "waiting"


class RouteEnv(CarlaEnv):
    lidar_range = LIDAR_RANGE_V2

    def __init__(self, *args, route_length=400.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_length = route_length
        self.route = []

    # ---- Episode API ------------------------------------------------------
    def reset(self, spawn_index=None, episode_seed=None):
        self._respawn(spawn_index)
        rng = random.Random(episode_seed) if episode_seed is not None else self.rng
        start = self.map.get_waypoint(self.ego.get_location(), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
        self.route = self._build_route(start, self.route_length, rng)
        self.route_idx = 0
        self.route_dist = 0.0
        self.route_total = (len(self.route) - 1) * ROUTE_STEP
        self.episode_steps = int(self.route_total / MIN_AVG_SPEED / FIXED_DELTA) + 100
        self.prev_steer = 0.0
        self.prev_pedal = 0.0
        self.wait_steps = 0
        self.stats = {"distance": 0.0, "speed_sum": 0.0, "steer_change": 0.0, "max_lat_acc": 0.0}
        return self._observe()

    def _build_route(self, start, length, rng):
        route, wp, dist = [start], start, 0.0
        while dist < length:
            options = wp.next(ROUTE_STEP)
            if not options:
                break
            wp = rng.choice(options)  # random turn at junctions
            route.append(wp)
            dist += ROUTE_STEP
        return route

    def step(self, action):
        # Real cars can't snap the wheel or pedals instantly.
        steer = float(np.clip(action[0], -1.0, 1.0))
        pedal = float(np.clip(action[1], -1.0, 1.0))
        steer = self.prev_steer + np.clip(steer - self.prev_steer, -MAX_STEER_DELTA, MAX_STEER_DELTA)
        pedal = self.prev_pedal + np.clip(pedal - self.prev_pedal, -MAX_PEDAL_DELTA, MAX_PEDAL_DELTA)
        self.ego.apply_control(carla.VehicleControl(
            steer=float(steer), throttle=max(float(pedal), 0.0), brake=max(-float(pedal), 0.0)))
        self._tick()
        self.steps += 1

        self.steer_change = abs(steer - self.prev_steer)
        self.pedal_change = abs(pedal - self.prev_pedal)
        self.prev_steer, self.prev_pedal = float(steer), float(pedal)

        self._advance_route()
        obs = self._observe()
        reward, done, info = self._reward()
        if not done and self.steps >= self.episode_steps:
            done, info["reason"] = True, "time_limit"
        if done:
            info.update(self.summary())
        return obs, reward, done, info

    def summary(self):
        s = self.stats
        n = max(self.steps, 1)
        return {
            "route_done": self.route_idx * ROUTE_STEP / max(self.route_total, ROUTE_STEP),
            "distance": s["distance"],
            "avg_speed_kmh": s["speed_sum"] / n * 3.6,
            "steer_change_per_s": s["steer_change"] / (n * FIXED_DELTA),
            "max_lat_acc": s["max_lat_acc"],
        }

    # ---- Route tracking ---------------------------------------------------
    def _advance_route(self):
        loc = self.ego.get_location()
        best, best_d = self.route_idx, float("inf")
        for j in range(self.route_idx, min(self.route_idx + 15, len(self.route))):
            d = loc.distance(self.route[j].transform.location)
            if d < best_d:
                best, best_d = j, d
        self.route_idx = best
        self.route_dist = best_d

    def _route_state(self):
        tf = self.ego.get_transform()
        wp = self.route[self.route_idx]
        fwd = wp.transform.get_forward_vector()
        dx = tf.location.x - wp.transform.location.x
        dy = tf.location.y - wp.transform.location.y
        lateral = fwd.x * dy - fwd.y * dx
        heading_err = wrap_angle(math.radians(tf.rotation.yaw - wp.transform.rotation.yaw))
        yaw = math.radians(tf.rotation.yaw)
        looks = []
        for metres in LOOKAHEAD_M:
            j = min(self.route_idx + int(metres / ROUTE_STEP), len(self.route) - 1)
            target = self.route[j].transform.location
            ang = math.atan2(target.y - tf.location.y, target.x - tf.location.x)
            looks.append(wrap_angle(ang - yaw))
        return wp, lateral, heading_err, looks

    def _target_speed(self):
        limit = self.ego.get_speed_limit() / 3.6
        return float(np.clip(limit if limit > 0 else 50 / 3.6, MIN_TARGET, MAX_TARGET))

    def _red_light(self):
        return self.ego.is_at_traffic_light() and self.ego.get_traffic_light_state() in (
            carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)

    # ---- Observation / reward ----------------------------------------------
    def _observe(self):
        self.sectors = lidar_sectors(self._latest_points, max_range=self.lidar_range,
                                     ego_box=self.ego_box)
        wp, lateral, heading_err, looks = self._route_state()
        half_width = max(wp.lane_width / 2.0, 0.5)
        extra = [
            np.clip(self._speed() / SPEED_NORM, 0.0, 1.5),
            self._target_speed() / SPEED_NORM,
            heading_err / math.pi,
            np.clip(lateral / half_width, -2.0, 2.0),
            *[a / math.pi for a in looks],
            self.prev_steer,
            self.prev_pedal,
            1.0 if self._red_light() else 0.0,
        ]
        return np.concatenate([self.sectors, np.array(extra, dtype=np.float32)])

    def _reward(self):
        wp, lateral, heading_err, _ = self._route_state()
        speed = self._speed()
        target = self._target_speed()
        half_width = max(wp.lane_width / 2.0, 0.5)
        info = {"reason": None}

        v = self.ego.get_velocity()
        fwd = wp.transform.get_forward_vector()
        progress = max(v.x * fwd.x + v.y * fwd.y, 0.0) * FIXED_DELTA
        if speed > 1.1 * target:
            progress *= 0.5
        reward = progress

        # Comfort and safety, as a real driver would care about them.
        lat_acc = abs(speed * math.radians(self.ego.get_angular_velocity().z))
        reward -= OVERSPEED_COST * max(0.0, speed - 1.1 * target)
        reward -= STEER_JERK_COST * self.steer_change + PEDAL_JERK_COST * self.pedal_change
        reward -= LATERAL_ACCEL_COST * max(0.0, lat_acc - LATERAL_ACCEL_LIMIT)
        reward -= LANE_CENTRE_COST * abs(lateral / half_width)
        front = self.sectors[0] * self.lidar_range
        if speed > 2.0 and front / speed < SAFE_HEADWAY:
            reward -= HEADWAY_COST * (SAFE_HEADWAY - front / speed)
        red = self._red_light()
        if red and speed > 1.0:
            reward -= RED_LIGHT_COST

        s = self.stats
        s["distance"] += progress
        s["speed_sum"] += speed
        s["steer_change"] += self.steer_change
        s["max_lat_acc"] = max(s["max_lat_acc"], lat_acc)

        done = False
        if self.collided:
            reward -= COLLISION_PENALTY
            done, info["reason"] = True, "collision"
        elif self.route_dist > half_width + 1.5 or abs(heading_err) > math.radians(100):
            reward -= OFF_ROUTE_PENALTY
            done, info["reason"] = True, "off_route"
        elif self.route_idx >= len(self.route) - 3:
            reward += COMPLETION_BONUS
            done, info["reason"] = True, "completed"
        elif self.steps > STUCK_GRACE_STEPS and speed < 0.5:
            if front < BLOCKED_DISTANCE or red:
                self.wait_steps += 1  # waiting in traffic or at a light is fine
                if self.wait_steps >= WAIT_LIMIT_STEPS:
                    done, info["reason"] = True, "blocked"
            else:
                self.slow_steps += 1
                if self.slow_steps >= STUCK_LIMIT_STEPS:
                    reward -= STUCK_PENALTY
                    done, info["reason"] = True, "stuck"
        else:
            self.slow_steps = 0
        return reward, done, info
