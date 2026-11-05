"""A planned route through the road network, and the car's progress along it."""

import math

from . import settings


def wrap_angle(angle):
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Route:
    """Waypoints `step` metres apart, from a start point through random turns."""

    def __init__(self, waypoints, step=settings.ROUTE_STEP):
        self.waypoints = waypoints
        self.step = step
        self.index = 0                     # waypoint the car is closest to
        self.distance_from_route = 0.0     # metres from that waypoint

    @classmethod
    def plan(cls, start, length, rng, step=settings.ROUTE_STEP):
        """Follow the road from `start` for `length` metres, turning randomly at junctions."""
        waypoints, waypoint, distance = [start], start, 0.0
        while distance < length:
            options = waypoint.next(step)
            if not options:
                break
            waypoint = rng.choice(options)
            waypoints.append(waypoint)
            distance += step
        return cls(waypoints, step)

    @property
    def total_length(self):
        return (len(self.waypoints) - 1) * self.step

    @property
    def current(self):
        return self.waypoints[self.index]

    @property
    def fraction_done(self):
        return self.index * self.step / max(self.total_length, self.step)

    @property
    def finished(self):
        return self.index >= len(self.waypoints) - settings.FINISH_MARGIN

    def advance(self, location):
        """Move on to the closest waypoint within the search window ahead."""
        end = min(self.index + settings.ROUTE_SEARCH_WINDOW, len(self.waypoints))
        best, best_distance = self.index, float("inf")
        for i in range(self.index, end):
            distance = location.distance(self.waypoints[i].transform.location)
            if distance < best_distance:
                best, best_distance = i, distance
        self.index, self.distance_from_route = best, best_distance

    def lane_position(self, transform):
        """(lateral offset in metres, positive = right of centre; heading error in radians)."""
        lane = self.current.transform
        forward = lane.get_forward_vector()
        dx = transform.location.x - lane.location.x
        dy = transform.location.y - lane.location.y
        lateral = forward.x * dy - forward.y * dx
        heading_error = wrap_angle(math.radians(transform.rotation.yaw - lane.rotation.yaw))
        return lateral, heading_error

    def lookahead_angles(self, transform, distances=settings.LOOKAHEAD):
        """Angle from the car's heading to the route point at each distance ahead."""
        yaw = math.radians(transform.rotation.yaw)
        angles = []
        for metres in distances:
            i = min(self.index + int(metres / self.step), len(self.waypoints) - 1)
            target = self.waypoints[i].transform.location
            bearing = math.atan2(target.y - transform.location.y, target.x - transform.location.x)
            angles.append(wrap_angle(bearing - yaw))
        return angles
