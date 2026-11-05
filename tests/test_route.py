import math
import random
import unittest

import carla

from neat_driver.route import Route, wrap_angle


class FakeWaypoint:
    """Just enough of carla.Waypoint: a road along the x axis, optionally forking."""

    def __init__(self, x, y=0.0, yaw=0.0, fork_at=None, lane_width=3.5):
        self.transform = carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw))
        self.lane_width = lane_width
        self.fork_at = fork_at

    def next(self, distance):
        x = self.transform.location.x + distance
        options = [FakeWaypoint(x, self.transform.location.y, fork_at=self.fork_at)]
        if self.fork_at is not None and x >= self.fork_at:
            options.append(FakeWaypoint(x, self.transform.location.y + 5, fork_at=None))
        return options


def car_at(x, y=0.0, yaw=0.0):
    return carla.Transform(carla.Location(x=x, y=y), carla.Rotation(yaw=yaw))


class RouteTest(unittest.TestCase):
    def test_plan_reaches_the_requested_length(self):
        route = Route.plan(FakeWaypoint(0), 100.0, random.Random(0), step=2.0)
        self.assertEqual(route.total_length, 100.0)
        self.assertEqual(len(route.waypoints), 51)

    def test_same_seed_same_route(self):
        plan = lambda: [w.transform.location.y for w in
                        Route.plan(FakeWaypoint(0, fork_at=10), 60.0, random.Random(7)).waypoints]
        self.assertEqual(plan(), plan())

    def test_advance_tracks_progress_and_distance(self):
        route = Route.plan(FakeWaypoint(0), 100.0, random.Random(0))
        route.advance(carla.Location(x=11.0, y=1.0))
        self.assertEqual(route.index, 5)  # waypoint at x = 10 (5 waypoints ahead is within the window)
        self.assertAlmostEqual(route.distance_from_route, math.hypot(1, 1), places=5)
        self.assertAlmostEqual(route.fraction_done, 0.1)

    def test_advance_only_searches_a_window_ahead(self):
        route = Route.plan(FakeWaypoint(0), 100.0, random.Random(0))
        route.advance(carla.Location(x=80.0))
        self.assertEqual(route.index, 14)  # can't jump further than the search window

    def test_lane_position_sign_and_heading(self):
        route = Route.plan(FakeWaypoint(0), 20.0, random.Random(0))
        lateral, heading = route.lane_position(car_at(0, 1.5, yaw=10))
        self.assertAlmostEqual(lateral, 1.5, places=5)  # +y is right of a road heading +x
        self.assertAlmostEqual(heading, math.radians(10), places=5)

    def test_lookahead_angles(self):
        route = Route.plan(FakeWaypoint(0), 40.0, random.Random(0))
        angles = route.lookahead_angles(car_at(0, 0, yaw=0), distances=(10.0,))
        self.assertAlmostEqual(angles[0], 0.0, places=5)
        angles = route.lookahead_angles(car_at(0, 0, yaw=90), distances=(10.0,))
        self.assertAlmostEqual(angles[0], -math.pi / 2, places=5)  # road is to the car's left

    def test_finished_near_the_end(self):
        route = Route.plan(FakeWaypoint(0), 20.0, random.Random(0))
        self.assertFalse(route.finished)
        for x in range(0, 21, 2):
            route.advance(carla.Location(x=float(x)))
        self.assertTrue(route.finished)

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3 * math.pi / 2), -math.pi / 2)
        self.assertAlmostEqual(wrap_angle(-3 * math.pi / 2), math.pi / 2)


if __name__ == "__main__":
    unittest.main()
