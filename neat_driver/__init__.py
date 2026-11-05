"""Evolve self-driving cars for the CARLA simulator with NEAT.

Modules:
    settings      every tunable number and file location
    simulator     connection to CARLA, synchronous stepping and background traffic
    ego_car       the car being driven, with its LiDAR and collision sensors
    lidar         turns a LiDAR point cloud into distance "rays"
    route         a planned route through the road network and progress along it
    rewards       what a good driver is rewarded and penalised for
    env           the driving task: observations, rewards and how drives end
    networks      NEAT config, saved genomes and networks
    evaluation    scoring genomes, the route-length curriculum
    netviz        matplotlib view of a network and its live signals
    training_log  reading the training log for the progress bar
    healthcheck   does the CARLA server answer? (used by run_training.ps1)
"""

try:
    import carla  # noqa: F401  (checked here so every entry point gets a clear message)
except ImportError as error:
    import sys

    raise SystemExit(
        f"The 'carla' package isn't installed for this Python ({sys.executable}).\n"
        "Use the conda env's Python instead, e.g. .\\play.bat or .\\run_training.ps1, or:\n"
        '  & "$env:USERPROFILE\\anaconda3\\envs\\carla\\python.exe" train.py'
    ) from error
