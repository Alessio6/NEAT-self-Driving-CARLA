"""Turn a LiDAR point cloud into a few distance "rays" a small network can use."""

import math

import numpy as np

from . import settings


def lidar_sectors(points, body_box=None, num_sectors=settings.LIDAR_SECTORS,
                  max_range=settings.LIDAR_RANGE, sensor_height=settings.LIDAR_HEIGHT,
                  ground_clearance=settings.GROUND_CLEARANCE):
    """Reduce an (N, 4) LiDAR point array to the nearest obstacle in each sector.

    Returns `num_sectors` values in [0, 1]: 1 means nothing within range and
    0 means touching. Sector 0 points straight ahead and the sectors go
    clockwise, because CARLA's y axis points right.

    `body_box` is (centre_x, half_length, half_width) in the sensor frame;
    hits inside it are the car's own body and are ignored.
    """
    result = np.ones(num_sectors, dtype=np.float32)
    if points is None or len(points) == 0:
        return result

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = z > ground_clearance - sensor_height  # above the road surface
    if body_box is not None:
        centre_x, half_length, half_width = body_box
        keep &= ~((np.abs(x - centre_x) < half_length) & (np.abs(y) < half_width))
    if not np.any(keep):
        return result
    x, y = x[keep], y[keep]

    distance = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)  # 0 = ahead, +pi/2 = right
    width = 2 * math.pi / num_sectors
    sector = (((angle + width / 2) % (2 * math.pi)) // width).astype(int)  # sector 0 centred ahead
    sector = np.clip(sector, 0, num_sectors - 1)

    np.minimum.at(result, sector, np.clip(distance / max_range, 0.0, 1.0))
    return result


def points_from_measurement(measurement):
    """CARLA LiDAR measurement -> (N, 4) array of x, y, z, intensity."""
    return np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
