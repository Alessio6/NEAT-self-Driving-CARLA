"""Exit with code 0 if the CARLA server answers, 1 if it doesn't.

An open port isn't enough: a frozen CARLA still accepts connections but never
replies. run_training.ps1 uses this to decide whether to restart CARLA:
    python -m neat_driver.healthcheck
"""

import sys

import carla

from . import settings


def carla_responds(timeout=20.0):
    client = carla.Client(settings.HOST, settings.PORT)
    client.set_timeout(timeout)
    try:
        client.get_world()
        return True
    except RuntimeError:
        return False


if __name__ == "__main__":
    sys.exit(0 if carla_responds() else 1)
