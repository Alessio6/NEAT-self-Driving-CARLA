"""Connection to the CARLA server: synchronous stepping and background traffic."""

import carla

from . import settings


class Simulator:
    """The CARLA world, stepped one tick at a time so every run is repeatable."""

    def __init__(self, host=settings.HOST, port=settings.PORT, town=settings.TOWN,
                 tm_port=settings.TRAFFIC_MANAGER_PORT, no_render=False, seed=None):
        self.client = carla.Client(host, port)
        self.client.set_timeout(settings.CLIENT_TIMEOUT)
        if town:
            self._load_town(town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprints = self.world.get_blueprint_library()
        self.rendering = not no_render

        self._original_settings = self.world.get_settings()
        world_settings = self.world.get_settings()
        world_settings.synchronous_mode = True
        world_settings.fixed_delta_seconds = settings.FIXED_DELTA
        world_settings.no_rendering_mode = no_render
        self.world.apply_settings(world_settings)

        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(settings.TRAFFIC_GAP)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(settings.HYBRID_PHYSICS_RADIUS)
        if seed is not None:
            self.traffic_manager.set_random_device_seed(seed)

        self.traffic_ids = []
        self._remove_leftovers()

    def _load_town(self, town):
        current = self.client.get_world().get_map().name
        if not current.endswith(town):
            print(f"Loading {town} (currently {current})...")
            self.client.load_world(town)

    def _remove_leftovers(self):
        """Destroy cars and sensors left behind by a run that crashed."""
        leftovers = [actor.id for actor in self.world.get_actors()
                     if actor.type_id.startswith(("vehicle.", "sensor.lidar", "sensor.other.collision"))]
        if leftovers:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in leftovers], True)
            print(f"Removed {len(leftovers)} vehicles/sensors left over from an earlier run")

    def spawn_traffic(self, count, rng):
        """Spawn `count` random cars driven by CARLA's traffic manager."""
        if count <= 0:
            return
        cars = [bp for bp in self.blueprints.filter("vehicle.*")
                if int(bp.get_attribute("number_of_wheels")) == 4]
        points = list(self.spawn_points)
        rng.shuffle(points)
        batch = []
        for transform in points[:count]:
            blueprint = rng.choice(cars)
            if blueprint.has_attribute("color"):
                blueprint.set_attribute("color", rng.choice(
                    blueprint.get_attribute("color").recommended_values))
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True,
                                           self.traffic_manager.get_port())))
        for response in self.client.apply_batch_sync(batch, True):
            if not response.error:
                self.traffic_ids.append(response.actor_id)
        print(f"Spawned {len(self.traffic_ids)} NPC vehicles")

    def tick(self):
        """Advance the world by one step; returns the new frame number."""
        return self.world.tick()

    def follow_with_camera(self, actor):
        """Put the spectator camera behind and above `actor`."""
        transform = actor.get_transform()
        back = transform.get_forward_vector() * -settings.CAMERA_DISTANCE
        location = transform.location + carla.Location(x=back.x, y=back.y, z=settings.CAMERA_HEIGHT)
        rotation = carla.Rotation(pitch=settings.CAMERA_PITCH, yaw=transform.rotation.yaw)
        self.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def close(self):
        """Remove the traffic and give CARLA back its normal (asynchronous) settings."""
        if self.traffic_ids:
            self.client.apply_batch([carla.command.DestroyActor(i) for i in self.traffic_ids])
            self.traffic_ids = []
        self.traffic_manager.set_synchronous_mode(False)
        self.world.apply_settings(self._original_settings)
