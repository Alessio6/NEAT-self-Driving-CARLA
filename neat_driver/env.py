"""The driving task: follow a planned route like a real driver.

Each drive (episode) plans a route from a spawn point, taking random turns at
junctions. The car is rewarded for progress along it and penalised for
driving badly (see rewards.py); the drive ends when it finishes the route,
crashes, leaves the route, gets stuck, waits too long or runs out of time.

Usage follows the usual reset()/step() pattern:

    with DrivingEnv(num_npcs=0) as env:
        observation = env.reset()
        observation, reward, done, info = env.step((steer, pedal))
"""

import math
import random
from dataclasses import dataclass

import carla
import numpy as np

from . import settings
from .ego_car import EgoCar
from .lidar import lidar_sectors
from .rewards import DrivingState, step_reward
from .route import Route
from .simulator import Simulator

_DIRECTIONS = {0: "front", 90: "right", 180: "rear", 270: "left"}


def _ray_label(i):
    degrees = round(i * 360 / settings.LIDAR_SECTORS)
    direction = f" {_DIRECTIONS[degrees]}" if degrees in _DIRECTIONS else ""
    return f"ray {degrees:>3}°{direction}"


INPUT_LABELS = (
    [_ray_label(i) for i in range(settings.LIDAR_SECTORS)]
    + ["speed", "speed limit", "heading err", "lane offset"]
    + [f"route {int(metres)} m" for metres in settings.LOOKAHEAD]
    + ["prev steer", "prev pedal", "red light"]
)
OUTPUT_LABELS = ["steer", "throttle/brake"]
NUM_INPUTS = len(INPUT_LABELS)
NUM_OUTPUTS = len(OUTPUT_LABELS)


def build_observation(rays, state, lookahead_angles, steer, pedal):
    """The network's inputs, in the order of INPUT_LABELS, scaled to roughly [-1, 1]."""
    lane = [
        np.clip(state.speed / settings.SPEED_SCALE, 0.0, settings.MAX_SPEED_INPUT),
        state.target_speed / settings.SPEED_SCALE,
        state.heading_error / math.pi,
        np.clip(state.lateral_offset / state.half_lane_width,
                -settings.MAX_LANE_OFFSET_INPUT, settings.MAX_LANE_OFFSET_INPUT),
        *[angle / math.pi for angle in lookahead_angles],
        steer,
        pedal,
        1.0 if state.red_light else 0.0,
    ]
    return np.concatenate([rays, np.array(lane, dtype=np.float32)])


@dataclass
class DriveStats:
    """Running totals for one drive, reported when it ends."""

    distance: float = 0.0
    speed_sum: float = 0.0
    steer_change_sum: float = 0.0
    max_lateral_accel: float = 0.0

    def add(self, state, metres):
        self.distance += metres
        self.speed_sum += state.speed
        self.steer_change_sum += state.steer_change
        self.max_lateral_accel = max(self.max_lateral_accel, state.lateral_accel)


class DrivingEnv:
    def __init__(self, host=settings.HOST, port=settings.PORT, town=settings.TOWN, num_npcs=30,
                 no_render=False, tm_port=settings.TRAFFIC_MANAGER_PORT, seed=None,
                 route_length=settings.ROUTE_START_LENGTH):
        self.rng = random.Random(seed)
        self.sim = Simulator(host, port, town, tm_port, no_render, seed)
        self.sim.spawn_traffic(num_npcs, self.rng)
        self.sim.tick()
        self.route_length = route_length  # the curriculum changes this between generations
        self.car = None
        self.route = None
        self.points = None

    @property
    def spawn_points(self):
        return self.sim.spawn_points

    # ---- Episode API ----------------------------------------------------------
    def reset(self, spawn_index=None, episode_seed=None):
        """Start a new drive; the same spawn index and seed give the same route."""
        self._spawn_car(spawn_index)
        rng = random.Random(episode_seed) if episode_seed is not None else self.rng
        start = self.sim.map.get_waypoint(self.car.location, project_to_road=True,
                                          lane_type=carla.LaneType.Driving)
        self.route = Route.plan(start, self.route_length, rng)
        self.time_limit = (int(self.route.total_length / settings.MIN_AVERAGE_SPEED / settings.FIXED_DELTA)
                           + settings.EXTRA_STEPS)
        self.steer = self.pedal = 0.0
        self.stopped_steps = 0   # stopped on a clear road
        self.waiting_steps = 0   # stopped behind traffic or at a red light
        self.stats = DriveStats()
        state, rays, lookahead = self._measure(steer_change=0.0, pedal_change=0.0)
        return build_observation(rays, state, lookahead, self.steer, self.pedal)

    def step(self, action):
        """Apply (steer, pedal) from the network for one step.

        Returns (observation, reward, done, info); info["ending"] says why a
        drive ended and the other info keys summarise it.
        """
        # Real cars can't snap the wheel or the pedals instantly.
        steer = self._rate_limited(self.steer, action[0], settings.MAX_STEER_CHANGE)
        pedal = self._rate_limited(self.pedal, action[1], settings.MAX_PEDAL_CHANGE)
        self.car.apply_controls(steer, pedal)
        self._tick()
        self.steps += 1
        steer_change, pedal_change = abs(steer - self.steer), abs(pedal - self.pedal)
        self.steer, self.pedal = steer, pedal

        self.route.advance(self.car.location)
        state, rays, lookahead = self._measure(steer_change, pedal_change)
        observation = build_observation(rays, state, lookahead, steer, pedal)
        reward, metres = step_reward(state)
        self.stats.add(state, metres)

        ending, bonus = self._ending(state)
        reward += bonus
        if ending is None and self.steps >= self.time_limit:
            ending = "time_limit"
        info = {"ending": ending}
        if ending is not None:
            info.update(self.summary())
        return observation, reward, ending is not None, info

    def summary(self):
        steps = max(self.steps, 1)
        return {
            "route_completed": self.route.fraction_done,
            "route_length": self.route.total_length,
            "distance": self.stats.distance,
            "average_speed_kmh": self.stats.speed_sum / steps * 3.6,
            "steering_changes_per_s": self.stats.steer_change_sum / (steps * settings.FIXED_DELTA),
            "max_lateral_accel": self.stats.max_lateral_accel,
        }

    def close(self):
        try:
            if self.car is not None:
                self.car.destroy()
                self.car = None
            self.sim.close()
        except RuntimeError as error:
            print(f"Cleanup warning: {error}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- Internals ------------------------------------------------------------
    def _spawn_car(self, spawn_index):
        if self.car is not None:
            self.car.destroy()
        if spawn_index is None:
            spawn_index = self.rng.randrange(len(self.spawn_points))
        self.car = EgoCar(self.sim, spawn_index)
        self.steps = 0
        # Let the car settle onto the road before the network takes over.
        for _ in range(settings.SETTLE_STEPS):
            self.car.apply_controls(0.0, -1.0)
            self._tick()
        self.car.collided = False

    def _tick(self):
        frame = self.sim.tick()
        self.points = self.car.read_lidar(frame)
        if self.sim.rendering:
            self.sim.follow_with_camera(self.car.actor)

    @staticmethod
    def _rate_limited(previous, target, max_change):
        target = float(np.clip(target, -1.0, 1.0))
        return float(previous + np.clip(target - previous, -max_change, max_change))

    def _measure(self, steer_change, pedal_change):
        """Read the car, route and sensors after a tick: (state, rays, lookahead angles)."""
        rays = lidar_sectors(self.points, body_box=self.car.body_box)
        transform = self.car.transform
        lateral, heading_error = self.route.lane_position(transform)
        lane = self.route.current
        forward = lane.transform.get_forward_vector()
        velocity = self.car.velocity
        speed = self.car.speed()
        limit = self.car.speed_limit()
        if limit <= 0:
            limit = settings.DEFAULT_SPEED_LIMIT_KMH / 3.6
        state = DrivingState(
            speed=speed,
            target_speed=float(np.clip(limit, settings.MIN_TARGET_SPEED, settings.MAX_TARGET_SPEED)),
            speed_along_route=max(velocity.x * forward.x + velocity.y * forward.y, 0.0),
            lateral_offset=lateral,
            half_lane_width=max(lane.lane_width / 2.0, settings.MIN_HALF_LANE_WIDTH),
            heading_error=heading_error,
            lateral_accel=abs(speed * self.car.yaw_rate()),
            front_distance=float(rays[0]) * settings.LIDAR_RANGE,
            red_light=self.car.facing_red_light(),
            steer_change=steer_change,
            pedal_change=pedal_change,
        )
        return state, rays, self.route.lookahead_angles(transform)

    def _ending(self, state):
        """(why the drive ends or None, one-off reward)."""
        if self.car.collided:
            return "collision", -settings.COLLISION_PENALTY
        off_lane = self.route.distance_from_route > state.half_lane_width + settings.OFF_ROUTE_MARGIN
        if off_lane or abs(state.heading_error) > math.radians(settings.MAX_HEADING_ERROR_DEG):
            return "off_route", -settings.OFF_ROUTE_PENALTY
        if self.route.finished:
            return "completed", settings.COMPLETION_BONUS

        if self.steps > settings.STUCK_GRACE_STEPS and state.speed < settings.STOPPED_SPEED:
            if state.front_distance < settings.BLOCKED_DISTANCE or state.red_light:
                # Waiting behind traffic or at a light is what a real driver does.
                self.waiting_steps += 1
                if self.waiting_steps >= settings.WAIT_LIMIT_STEPS:
                    return "blocked", 0.0
            else:
                self.stopped_steps += 1
                if self.stopped_steps >= settings.STUCK_LIMIT_STEPS:
                    return "stuck", -settings.STUCK_PENALTY
        else:
            self.stopped_steps = 0
        return None, 0.0
