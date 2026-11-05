"""CARLA driving environment for NEAT.

The ego car sees the world through a LiDAR sensor (binned into distance "rays")
plus a few lane features, and is controlled by two network outputs:
steering and a combined throttle/brake value.
"""

import math
import queue
import random

try:
    import carla
except ImportError:
    raise SystemExit(
        "The 'carla' package isn't installed for this Python ("
        + __import__("sys").executable + ").\n"
        "Use the conda env's Python instead, e.g. .\\play.bat, or:\n"
        "  & \"$env:USERPROFILE\\anaconda3\\envs\\carla\\python.exe\" play.py")
import numpy as np

# ---- Observation layout ---------------------------------------------------
NUM_LIDAR_SECTORS = 16          # rays spread evenly over 360 degrees
LIDAR_RANGE = 30.0              # metres
LIDAR_HEIGHT = 2.4              # mount height above the car's origin
GROUND_CLEARANCE = 0.3          # ignore LiDAR hits lower than this (road surface)
LANE_FEATURES = 4               # speed, heading error, lateral offset, look-ahead angle
NUM_INPUTS = NUM_LIDAR_SECTORS + LANE_FEATURES
NUM_OUTPUTS = 2                 # steer, throttle(+)/brake(-)

# ---- Episode / reward settings --------------------------------------------
FIXED_DELTA = 0.05              # 20 Hz simulation
MAX_SPEED = 12.0                # m/s (~43 km/h) - progress above this is discounted
SPEED_NORM = 20.0               # m/s used to normalise the speed input
COLLISION_PENALTY = 25.0
OFFROAD_PENALTY = 10.0
STUCK_GRACE_STEPS = 80          # steps before "stuck" checks start
STUCK_LIMIT_STEPS = 60          # consecutive slow steps that end the episode
STUCK_SPEED = 0.5               # m/s


def lidar_sectors(points, num_sectors=NUM_LIDAR_SECTORS, max_range=LIDAR_RANGE,
                  sensor_height=LIDAR_HEIGHT, ground_clearance=GROUND_CLEARANCE,
                  ego_box=None):
    """Reduce an (N, 4) LiDAR point array to per-sector normalised distances.

    Returns values in [0, 1]: 1.0 means nothing within range, 0.0 means touching.
    Sector 0 points straight ahead and sectors go clockwise (CARLA's y is right).
    ego_box is (centre_x, half_length, half_width) in the sensor frame; hits
    inside it are the car's own body and are ignored.
    """
    result = np.ones(num_sectors, dtype=np.float32)
    if points is None or len(points) == 0:
        return result

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Keep only obstacles above the road surface.
    mask = z > (-sensor_height + ground_clearance)
    if ego_box is not None:
        cx, half_len, half_wid = ego_box
        mask &= ~((np.abs(x - cx) < half_len) & (np.abs(y) < half_wid))
    if not np.any(mask):
        return result
    x, y = x[mask], y[mask]

    dist = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)  # 0 = forward, +pi/2 = right
    sector_width = 2 * math.pi / num_sectors
    # Centre sector 0 on the forward direction.
    idx = (((angle + sector_width / 2) % (2 * math.pi)) // sector_width).astype(int)
    idx = np.clip(idx, 0, num_sectors - 1)

    np.minimum.at(result, idx, np.clip(dist / max_range, 0.0, 1.0))
    return result


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CarlaEnv:
    lidar_range = LIDAR_RANGE

    def __init__(self, host="localhost", port=2000, town="Town04", num_npcs=30,
                 no_render=False, tm_port=8000, seed=None):
        self.rng = random.Random(seed)
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)

        if town:
            current = self.client.get_world().get_map().name
            if not current.endswith(town):
                print(f"Loading {town} (currently {current})...")
                self.client.load_world(town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.blueprints = self.world.get_blueprint_library()

        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA
        settings.no_rendering_mode = no_render
        self.world.apply_settings(settings)
        self.no_render = no_render

        self.tm = self.client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_hybrid_physics_radius(70.0)
        if seed is not None:
            self.tm.set_random_device_seed(seed)

        self.npcs = []
        self.ego = None
        self.lidar = None
        self.collision_sensor = None
        self.lidar_queue = queue.Queue()
        self._latest_points = None
        self.collided = False

        self._spawn_npcs(num_npcs)
        self.world.tick()

    # ---- Actor management -------------------------------------------------
    def _spawn_npcs(self, count):
        if count <= 0:
            return
        vehicle_bps = [bp for bp in self.blueprints.filter("vehicle.*")
                       if int(bp.get_attribute("number_of_wheels")) == 4]
        points = list(self.spawn_points)
        self.rng.shuffle(points)
        batch = []
        for transform in points[:count]:
            bp = self.rng.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", self.rng.choice(
                    bp.get_attribute("color").recommended_values))
            bp.set_attribute("role_name", "autopilot")
            batch.append(carla.command.SpawnActor(bp, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm.get_port())))
        for response in self.client.apply_batch_sync(batch, True):
            if not response.error:
                self.npcs.append(response.actor_id)
        print(f"Spawned {len(self.npcs)} NPC vehicles")

    def _destroy_ego(self):
        for actor in (self.lidar, self.collision_sensor):
            if actor is not None:
                actor.stop()
                actor.destroy()
        if self.ego is not None:
            self.ego.destroy()
        self.ego = self.lidar = self.collision_sensor = None

    def _spawn_ego(self, spawn_index):
        bp = self.blueprints.find("vehicle.tesla.model3")
        bp.set_attribute("role_name", "hero")
        # Try the requested spawn point first, then others if it is occupied.
        order = [spawn_index] + [i for i in range(len(self.spawn_points)) if i != spawn_index]
        for i in order:
            self.ego = self.world.try_spawn_actor(bp, self.spawn_points[i])
            if self.ego is not None:
                break
        if self.ego is None:
            raise RuntimeError("Could not spawn the ego vehicle at any spawn point")

        lidar_bp = self.blueprints.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "32")
        lidar_bp.set_attribute("range", str(self.lidar_range))
        lidar_bp.set_attribute("points_per_second", "200000")
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / FIXED_DELTA))
        lidar_bp.set_attribute("upper_fov", "10")
        lidar_bp.set_attribute("lower_fov", "-30")
        lidar_bp.set_attribute("dropoff_general_rate", "0.0")
        lidar_tf = carla.Transform(carla.Location(x=0.0, z=LIDAR_HEIGHT))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.ego)
        bb = self.ego.bounding_box
        self.ego_box = (bb.location.x, bb.extent.x + 0.3, bb.extent.y + 0.3)
        self.lidar_queue = queue.Queue()
        self.lidar.listen(self.lidar_queue.put)

        col_bp = self.blueprints.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.ego)
        self.collided = False
        self.collision_sensor.listen(self._on_collision)

    def _on_collision(self, event):
        self.collided = True

    # ---- Episode API ------------------------------------------------------
    def reset(self, spawn_index=None, episode_seed=None):
        self._respawn(spawn_index)
        return self._observe()

    def _respawn(self, spawn_index):
        self._destroy_ego()
        if spawn_index is None:
            spawn_index = self.rng.randrange(len(self.spawn_points))
        self._spawn_ego(spawn_index)
        self.steps = 0
        self.slow_steps = 0
        # Let the car settle onto the road before the network takes over.
        for _ in range(5):
            self.ego.apply_control(carla.VehicleControl(brake=1.0))
            self._tick()
        self.collided = False

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        pedal = float(np.clip(action[1], -1.0, 1.0))
        control = carla.VehicleControl(
            steer=steer,
            throttle=max(pedal, 0.0),
            brake=max(-pedal, 0.0),
        )
        self.ego.apply_control(control)
        self._tick()
        self.steps += 1

        obs = self._observe()
        reward, done, info = self._reward()
        return obs, reward, done, info

    def _tick(self):
        frame = self.world.tick()
        # Drain the LiDAR queue down to the measurement for this frame.
        self._latest_points = None
        while True:
            try:
                data = self.lidar_queue.get(timeout=2.0)
            except queue.Empty:
                break
            if data.frame >= frame:
                pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
                self._latest_points = pts
                break
        if not self.no_render:
            self._follow_camera()

    def _follow_camera(self):
        tf = self.ego.get_transform()
        back = tf.get_forward_vector() * -8.0
        loc = tf.location + carla.Location(x=back.x, y=back.y, z=4.0)
        rot = carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw)
        self.world.get_spectator().set_transform(carla.Transform(loc, rot))

    def _lane_state(self):
        tf = self.ego.get_transform()
        wp = self.map.get_waypoint(tf.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        wp_fwd = wp.transform.get_forward_vector()
        dx = tf.location.x - wp.transform.location.x
        dy = tf.location.y - wp.transform.location.y
        # Signed distance from lane centre (positive = right of centre).
        lateral = wp_fwd.x * dy - wp_fwd.y * dx
        heading_err = wrap_angle(math.radians(tf.rotation.yaw - wp.transform.rotation.yaw))

        ahead = wp.next(10.0)
        target = ahead[0].transform.location if ahead else wp.transform.location
        to_target = math.atan2(target.y - tf.location.y, target.x - tf.location.x)
        lookahead = wrap_angle(to_target - math.radians(tf.rotation.yaw))
        return wp, lateral, heading_err, lookahead

    def _speed(self):
        v = self.ego.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

    def _observe(self):
        sectors = lidar_sectors(self._latest_points, ego_box=self.ego_box)
        wp, lateral, heading_err, lookahead = self._lane_state()
        half_width = max(wp.lane_width / 2.0, 0.5)
        lane = np.array([
            np.clip(self._speed() / SPEED_NORM, 0.0, 1.5),
            heading_err / math.pi,
            np.clip(lateral / half_width, -2.0, 2.0),
            lookahead / math.pi,
        ], dtype=np.float32)
        return np.concatenate([sectors, lane])

    def _reward(self):
        wp, lateral, heading_err, _ = self._lane_state()
        speed = self._speed()
        info = {"reason": None}

        # Progress along the lane direction, discounted above the speed cap.
        v = self.ego.get_velocity()
        fwd = wp.transform.get_forward_vector()
        progress = max(v.x * fwd.x + v.y * fwd.y, 0.0) * FIXED_DELTA
        if speed > MAX_SPEED:
            progress *= 0.5
        # Small penalty for drifting from the lane centre.
        half_width = max(wp.lane_width / 2.0, 0.5)
        reward = progress - 0.02 * abs(lateral / half_width)

        done = False
        if self.collided:
            reward -= COLLISION_PENALTY
            done, info["reason"] = True, "collision"
        elif abs(lateral) > half_width + 1.0 or abs(heading_err) > math.radians(100):
            reward -= OFFROAD_PENALTY
            done, info["reason"] = True, "off_lane"
        elif self.steps > STUCK_GRACE_STEPS:
            self.slow_steps = self.slow_steps + 1 if speed < STUCK_SPEED else 0
            if self.slow_steps >= STUCK_LIMIT_STEPS:
                # As costly as leaving the lane, so standing still is never "safe".
                reward -= OFFROAD_PENALTY
                done, info["reason"] = True, "stuck"
        return reward, done, info

    def close(self):
        try:
            self._destroy_ego()
            if self.npcs:
                self.client.apply_batch([carla.command.DestroyActor(i) for i in self.npcs])
                self.npcs = []
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self._original_settings)
        except RuntimeError as e:
            print(f"Cleanup warning: {e}")
