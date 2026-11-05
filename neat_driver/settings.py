"""Every tunable number and file location in one place.

Changing a value that affects driving (inputs, rewards, endings) changes what
the cars are trained for, so networks trained before the change may score
differently afterwards.
"""

from pathlib import Path

# ---- Files ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEAT_CONFIG = PROJECT_ROOT / "config-neat.txt"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_GENOME = MODELS_DIR / "best_genome.pkl"
WINNER_GENOME = MODELS_DIR / "winner_genome.pkl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PREFIX = "neat-checkpoint-"
CURRICULUM_FILE = CHECKPOINT_DIR / "curriculum.json"
TRAINING_LOG = PROJECT_ROOT / "logs" / "training_log.txt"

# ---- Simulator --------------------------------------------------------------
HOST = "localhost"
PORT = 2000
TRAFFIC_MANAGER_PORT = 8000
TOWN = "Town04"
CLIENT_TIMEOUT = 30.0            # seconds to wait for the CARLA server
FIXED_DELTA = 0.05               # simulated seconds per step (20 Hz)
TRAFFIC_GAP = 2.5                # metres traffic keeps to the vehicle ahead
HYBRID_PHYSICS_RADIUS = 70.0     # traffic further away uses cheaper physics
CAMERA_DISTANCE = 8.0            # spectator camera: metres behind the car
CAMERA_HEIGHT = 4.0              # ... metres above it
CAMERA_PITCH = -15.0             # ... looking down this many degrees

# ---- Ego car and sensors ----------------------------------------------------
EGO_BLUEPRINT = "vehicle.tesla.model3"
SETTLE_STEPS = 5                 # braked steps after spawning, before the network drives
LIDAR_SECTORS = 16               # distance "rays" spread evenly over 360 degrees
LIDAR_RANGE = 50.0               # metres
LIDAR_HEIGHT = 2.4               # mount height above the car's origin
LIDAR_CHANNELS = 32
LIDAR_POINTS_PER_SECOND = 200_000
LIDAR_UPPER_FOV = 10.0           # degrees
LIDAR_LOWER_FOV = -30.0          # degrees
GROUND_CLEARANCE = 0.3           # hits lower than this above the road are the road
BODY_MARGIN = 0.3                # hits this close around the car's body are the car
SENSOR_TIMEOUT = 2.0             # seconds to wait for a LiDAR measurement

# ---- Route ------------------------------------------------------------------
ROUTE_STEP = 2.0                 # metres between route waypoints
ROUTE_SEARCH_WINDOW = 15         # waypoints ahead searched when tracking progress
LOOKAHEAD = (5.0, 15.0, 30.0)    # metres ahead at which the route direction is an input
FINISH_MARGIN = 3                # waypoints from the end that count as finished
MIN_AVERAGE_SPEED = 3.0          # m/s; sets a route's time limit (allows for red lights)
EXTRA_STEPS = 100                # added to every route's time limit

# ---- Controls ---------------------------------------------------------------
MAX_STEER_CHANGE = 0.06          # per step: full lock from centre takes ~0.8 s
MAX_PEDAL_CHANGE = 0.10          # per step

# ---- Inputs -----------------------------------------------------------------
SPEED_SCALE = 25.0               # m/s that maps to 1.0 in the speed inputs
MAX_SPEED_INPUT = 1.5
MAX_LANE_OFFSET_INPUT = 2.0      # lane half-widths
MIN_HALF_LANE_WIDTH = 0.5        # metres; guards against tiny or missing lane widths

# ---- Speed limit ------------------------------------------------------------
MIN_TARGET_SPEED = 8.0           # m/s; the road's limit is clamped to this range
MAX_TARGET_SPEED = 25.0
DEFAULT_SPEED_LIMIT_KMH = 50.0   # used when CARLA reports no limit
SPEEDING_MARGIN = 1.1            # above 110% of the limit counts as speeding

# ---- Rewards, per step ------------------------------------------------------
SPEEDING_PROGRESS_FACTOR = 0.5   # share of progress credited while speeding
SPEEDING_COST = 0.02             # per m/s over the speeding margin
STEER_CHANGE_COST = 1.0          # per unit of steering change
PEDAL_CHANGE_COST = 0.3          # per unit of throttle/brake change
COMFORT_LATERAL_ACCEL = 3.0      # m/s^2: comfortable cornering for a passenger
LATERAL_ACCEL_COST = 0.02        # per m/s^2 above that
LANE_CENTRE_COST = 0.02          # per lane half-width away from the centre
SAFE_HEADWAY = 1.5               # seconds behind the vehicle ahead
HEADWAY_MIN_SPEED = 2.0          # m/s; slower than this the gap isn't judged
HEADWAY_COST = 0.05              # per second short of the safe headway
RED_LIGHT_COST = 0.2             # per step moving towards a red or yellow light
RED_LIGHT_MOVING_SPEED = 1.0     # m/s

# ---- How drives end ---------------------------------------------------------
COMPLETION_BONUS = 50.0
COLLISION_PENALTY = 50.0
OFF_ROUTE_PENALTY = 25.0
STUCK_PENALTY = 25.0
OFF_ROUTE_MARGIN = 1.5           # metres beyond the lane edge
MAX_HEADING_ERROR_DEG = 100.0
STOPPED_SPEED = 0.5              # m/s
STUCK_GRACE_STEPS = 80           # steps after a start before "stuck" is checked
STUCK_LIMIT_STEPS = 60           # 3 s stopped on a clear road
WAIT_LIMIT_STEPS = 1200          # 60 s waiting in traffic (red phases can last ~40 s)
BLOCKED_DISTANCE = 12.0          # metres: something this close ahead means waiting

# ---- Training ---------------------------------------------------------------
EPISODES_PER_GENOME = 3          # routes each car drives per generation
ROUTE_START_LENGTH = 400.0       # metres
ROUTE_MAX_LENGTH = 2000.0
ROUTE_GROWTH = 1.5               # when the best car completes all its routes
