"""The car being driven: a Tesla Model 3 with a roof LiDAR and a collision sensor."""

import math
import queue

import carla

from . import settings
from .lidar import points_from_measurement


class EgoCar:
    def __init__(self, sim, spawn_index):
        self.actor = self._spawn(sim, spawn_index)
        self.lidar = self._attach_lidar(sim)
        self.collision_sensor = sim.world.spawn_actor(
            sim.blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=self.actor)
        self.collided = False
        self.collision_sensor.listen(self._on_collision)

        box = self.actor.bounding_box
        # The car's body as seen from the LiDAR, so its own hits can be ignored.
        self.body_box = (box.location.x, box.extent.x + settings.BODY_MARGIN,
                         box.extent.y + settings.BODY_MARGIN)

    def _spawn(self, sim, spawn_index):
        blueprint = sim.blueprints.find(settings.EGO_BLUEPRINT)
        blueprint.set_attribute("role_name", "hero")
        # Try the requested spawn point first, then the others if it's occupied.
        order = [spawn_index] + [i for i in range(len(sim.spawn_points)) if i != spawn_index]
        for i in order:
            actor = sim.world.try_spawn_actor(blueprint, sim.spawn_points[i])
            if actor is not None:
                return actor
        raise RuntimeError("Could not spawn the ego vehicle at any spawn point")

    def _attach_lidar(self, sim):
        blueprint = sim.blueprints.find("sensor.lidar.ray_cast")
        for name, value in {
            "channels": settings.LIDAR_CHANNELS,
            "range": settings.LIDAR_RANGE,
            "points_per_second": settings.LIDAR_POINTS_PER_SECOND,
            "rotation_frequency": 1.0 / settings.FIXED_DELTA,  # one full sweep per step
            "upper_fov": settings.LIDAR_UPPER_FOV,
            "lower_fov": settings.LIDAR_LOWER_FOV,
            "dropoff_general_rate": 0.0,
        }.items():
            blueprint.set_attribute(name, str(value))
        mount = carla.Transform(carla.Location(x=0.0, z=settings.LIDAR_HEIGHT))
        lidar = sim.world.spawn_actor(blueprint, mount, attach_to=self.actor)
        self._measurements = queue.Queue()
        lidar.listen(self._measurements.put)
        return lidar

    def _on_collision(self, event):
        self.collided = True

    def read_lidar(self, frame):
        """The LiDAR points for `frame`, skipping older measurements (None if none arrive)."""
        while True:
            try:
                measurement = self._measurements.get(timeout=settings.SENSOR_TIMEOUT)
            except queue.Empty:
                return None
            if measurement.frame >= frame:
                return points_from_measurement(measurement)

    def apply_controls(self, steer, pedal):
        """Steer in [-1, 1]; pedal in [-1, 1] with positive = throttle, negative = brake."""
        self.actor.apply_control(carla.VehicleControl(
            steer=float(steer), throttle=max(float(pedal), 0.0), brake=max(-float(pedal), 0.0)))

    @property
    def transform(self):
        return self.actor.get_transform()

    @property
    def location(self):
        return self.actor.get_location()

    @property
    def velocity(self):
        return self.actor.get_velocity()

    def speed(self):
        v = self.velocity
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def yaw_rate(self):
        """Radians per second."""
        return math.radians(self.actor.get_angular_velocity().z)

    def speed_limit(self):
        """The current road's limit in m/s (0 if CARLA reports none)."""
        return self.actor.get_speed_limit() / 3.6

    def facing_red_light(self):
        return self.actor.is_at_traffic_light() and self.actor.get_traffic_light_state() in (
            carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)

    def destroy(self):
        for sensor in (self.lidar, self.collision_sensor):
            sensor.stop()
            sensor.destroy()
        self.actor.destroy()
