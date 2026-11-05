import unittest

import numpy as np

from neat_driver import settings
from neat_driver.lidar import lidar_sectors

GROUND = -settings.LIDAR_HEIGHT  # z of the road surface in the sensor frame


def points(*xyz):
    return np.array([[x, y, z, 1.0] for x, y, z in xyz], dtype=np.float32)


class LidarSectorsTest(unittest.TestCase):
    def test_nothing_seen_reads_clear(self):
        self.assertTrue(np.all(lidar_sectors(None) == 1.0))
        self.assertTrue(np.all(lidar_sectors(points()) == 1.0))

    def test_obstacle_ahead_and_to_the_right(self):
        rays = lidar_sectors(points((10, 0, 0), (0, 20, 0)))
        self.assertAlmostEqual(rays[0], 10 / settings.LIDAR_RANGE, places=6)  # ahead
        self.assertAlmostEqual(rays[4], 20 / settings.LIDAR_RANGE, places=6)  # 90 degrees, right
        self.assertEqual(int(np.sum(rays < 1)), 2)

    def test_nearest_hit_wins_within_a_sector(self):
        rays = lidar_sectors(points((30, 0, 0), (12, 1, 0)))
        self.assertAlmostEqual(rays[0], np.hypot(12, 1) / settings.LIDAR_RANGE, places=6)

    def test_road_surface_is_ignored(self):
        rays = lidar_sectors(points((5, 0, GROUND), (5, 0, GROUND + 0.1)))
        self.assertTrue(np.all(rays == 1.0))

    def test_own_body_is_ignored(self):
        body = (0.0, 2.7, 1.2)  # centre x, half length, half width
        rays = lidar_sectors(points((-2.0, 0.0, -1.0)), body_box=body)
        self.assertTrue(np.all(rays == 1.0))
        rays = lidar_sectors(points((-6.0, 0.0, -1.0)), body_box=body)
        self.assertLess(rays[settings.LIDAR_SECTORS // 2], 1.0)  # a car behind is still seen

    def test_far_hits_are_capped(self):
        rays = lidar_sectors(points((500, 0, 0)))
        self.assertEqual(rays[0], 1.0)


if __name__ == "__main__":
    unittest.main()
