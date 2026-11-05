"""Live progress bar for a training run.

Only reads logs/training_log.txt and the checkpoint times, so it can be
started or closed at any time without affecting training:
    python progress.py --total 100     # training stops at generation 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

from neat_driver import settings, training_log
from neat_driver.networks import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total", type=int, default=50, help="generation number training stops at")
    parser.add_argument("--log", type=Path, default=settings.TRAINING_LOG)
    parser.add_argument("--refresh", type=float, default=2.0, help="seconds between updates")
    return parser.parse_args()


def main():
    args = parse_args()
    population = load_config().pop_size
    os.system("")  # enables ANSI cursor movement in Windows consoles
    printed = 0
    try:
        while True:
            if args.log.exists():
                progress = training_log.parse(training_log.read_lines(args.log))
                lines = training_log.render(progress, args.total, population,
                                            training_log.seconds_per_generation(),
                                            training_log.log_age(args.log))
                done = progress.finished or (progress.generation or 0) >= args.total
            else:
                lines, done = [f"Waiting for {args.log} ..."], False
            if printed:
                sys.stdout.write(f"\x1b[{printed}F")  # back to the first line
            for line in lines:
                sys.stdout.write("\x1b[2K" + line + "\n")
            sys.stdout.flush()
            printed = len(lines)
            if done:
                break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
