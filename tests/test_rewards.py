import math
import unittest
from dataclasses import replace

import numpy as np

from neat_driver import settings
from neat_driver.env import INPUT_LABELS, NUM_INPUTS, build_observation
from neat_driver.rewards import DrivingState, step_reward

STEP = settings.FIXED_DELTA

GOOD_DRIVING = DrivingState(speed=10.0, target_speed=13.9, speed_along_route=10.0, lateral_offset=0.0,
                            half_lane_width=1.75, heading_error=0.0, lateral_accel=1.0,
                            front_distance=50.0, red_light=False, steer_change=0.0, pedal_change=0.0)


class StepRewardTest(unittest.TestCase):
    def test_good_driving_earns_its_progress(self):
        reward, metres = step_reward(GOOD_DRIVING)
        self.assertAlmostEqual(metres, 10.0 * STEP)
        self.assertAlmostEqual(reward, 10.0 * STEP)

    def test_speeding_halves_progress_and_costs_extra(self):
        state = replace(GOOD_DRIVING, speed=20.0, speed_along_route=20.0, target_speed=10.0)
        reward, metres = step_reward(state)
        self.assertAlmostEqual(metres, 20.0 * STEP * settings.SPEEDING_PROGRESS_FACTOR)
        self.assertAlmostEqual(reward, metres - settings.SPEEDING_COST * (20.0 - 11.0))

    def test_jerky_controls_cost(self):
        reward, metres = step_reward(replace(GOOD_DRIVING, steer_change=0.06, pedal_change=0.1))
        self.assertAlmostEqual(reward, metres - 0.06 * settings.STEER_CHANGE_COST
                               - 0.1 * settings.PEDAL_CHANGE_COST)

    def test_hard_cornering_costs(self):
        reward, metres = step_reward(replace(GOOD_DRIVING, lateral_accel=5.0))
        self.assertAlmostEqual(reward, metres - settings.LATERAL_ACCEL_COST * 2.0)

    def test_off_centre_costs(self):
        reward, metres = step_reward(replace(GOOD_DRIVING, lateral_offset=-0.875))
        self.assertAlmostEqual(reward, metres - settings.LANE_CENTRE_COST * 0.5)

    def test_tailgating_costs(self):
        reward, metres = step_reward(replace(GOOD_DRIVING, front_distance=5.0))  # 0.5 s behind
        self.assertAlmostEqual(reward, metres - settings.HEADWAY_COST * 1.0)
        reward, _ = step_reward(replace(GOOD_DRIVING, speed=1.0, speed_along_route=1.0, front_distance=0.5))
        self.assertAlmostEqual(reward, 1.0 * STEP)  # crawling in a queue isn't tailgating

    def test_moving_towards_a_red_light_costs(self):
        reward, metres = step_reward(replace(GOOD_DRIVING, red_light=True))
        self.assertAlmostEqual(reward, metres - settings.RED_LIGHT_COST)
        reward, _ = step_reward(replace(GOOD_DRIVING, red_light=True, speed=0.0, speed_along_route=0.0))
        self.assertEqual(reward, 0.0)  # stopped at the light


class ObservationTest(unittest.TestCase):
    def test_layout_matches_the_labels(self):
        rays = np.full(settings.LIDAR_SECTORS, 0.5, dtype=np.float32)
        state = replace(GOOD_DRIVING, speed=50.0, lateral_offset=7.0, heading_error=math.pi / 2, red_light=True)
        obs = build_observation(rays, state, [0.1, 0.2, math.pi], steer=0.3, pedal=-0.4)
        self.assertEqual(len(obs), NUM_INPUTS)
        self.assertEqual(len(INPUT_LABELS), 26)
        values = dict(zip(INPUT_LABELS, obs))
        self.assertAlmostEqual(values["speed"], settings.MAX_SPEED_INPUT)          # clipped
        self.assertAlmostEqual(values["speed limit"], 13.9 / settings.SPEED_SCALE, places=6)
        self.assertAlmostEqual(values["heading err"], 0.5)
        self.assertAlmostEqual(values["lane offset"], settings.MAX_LANE_OFFSET_INPUT)  # clipped
        self.assertAlmostEqual(values["route 30 m"], 1.0)
        self.assertAlmostEqual(values["prev steer"], 0.3, places=6)
        self.assertAlmostEqual(values["prev pedal"], -0.4, places=6)
        self.assertEqual(values["red light"], 1.0)
        self.assertEqual(obs.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
