"""What a good driver is rewarded and penalised for, each step."""

from dataclasses import dataclass

from . import settings


@dataclass(frozen=True)
class DrivingState:
    """What the car is doing this step, measured after the simulator ticked."""

    speed: float                # m/s
    target_speed: float         # m/s: the road's speed limit, clamped
    speed_along_route: float    # m/s in the route's direction (never negative)
    lateral_offset: float       # metres from the lane centre, positive = right
    half_lane_width: float      # metres
    heading_error: float        # radians between the car and the lane
    lateral_accel: float        # m/s^2 sideways
    front_distance: float       # metres to the nearest obstacle straight ahead
    red_light: bool             # facing a red or yellow light
    steer_change: float         # how much steering changed this step
    pedal_change: float         # how much throttle/brake changed this step

    @property
    def speeding(self):
        return self.speed > settings.SPEEDING_MARGIN * self.target_speed


def progress(state):
    """Metres travelled along the route this step, with reduced credit while speeding."""
    metres = state.speed_along_route * settings.FIXED_DELTA
    if state.speeding:
        metres *= settings.SPEEDING_PROGRESS_FACTOR
    return metres


def step_reward(state):
    """Reward for one step: progress minus the costs of driving badly.

    Returns (reward, progress in metres).
    """
    metres = progress(state)
    reward = metres
    reward -= settings.SPEEDING_COST * max(0.0, state.speed - settings.SPEEDING_MARGIN * state.target_speed)
    reward -= (settings.STEER_CHANGE_COST * state.steer_change
               + settings.PEDAL_CHANGE_COST * state.pedal_change)
    reward -= settings.LATERAL_ACCEL_COST * max(0.0, state.lateral_accel - settings.COMFORT_LATERAL_ACCEL)
    reward -= settings.LANE_CENTRE_COST * abs(state.lateral_offset / state.half_lane_width)
    if state.speed > settings.HEADWAY_MIN_SPEED:
        headway = state.front_distance / state.speed  # seconds to the vehicle ahead
        if headway < settings.SAFE_HEADWAY:
            reward -= settings.HEADWAY_COST * (settings.SAFE_HEADWAY - headway)
    if state.red_light and state.speed > settings.RED_LIGHT_MOVING_SPEED:
        reward -= settings.RED_LIGHT_COST
    return reward, metres
