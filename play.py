"""Watch a trained car drive in CARLA (start CARLA first).

    play.bat                              # the best car, 2 km routes, real time
    play.bat --npcs 0 --episodes 5        # no traffic, 5 drives
    play.bat --viz --speed 2              # network window, twice as fast
    play.bat --genome models\\winner_genome.pkl
"""

import argparse
import time
from pathlib import Path

from neat_driver import settings
from neat_driver.env import DrivingEnv
from neat_driver.evaluation import drive
from neat_driver.networks import load_config, load_genome, make_network


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome", type=Path, default=settings.BEST_GENOME, help="saved car to drive")
    parser.add_argument("--episodes", type=int, default=3, help="number of drives")
    parser.add_argument("--npcs", type=int, default=30, help="number of traffic vehicles")
    parser.add_argument("--route-length", type=float, default=settings.ROUTE_MAX_LENGTH,
                        help="route length in metres")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed: 1 = real time, 2 = twice as fast, 0 = as fast as possible")
    parser.add_argument("--viz", action="store_true", help="show the network and its live signals")
    parser.add_argument("--viz-every", type=int, default=2, help="steps between --viz updates")
    parser.add_argument("--town", default=settings.TOWN)
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--tm-port", type=int, default=settings.TRAFFIC_MANAGER_PORT)
    return parser.parse_args()


def live_view(genome, net, config, path, every):
    """Open the network window; returns the per-step callback that updates it."""
    from neat_driver.netviz import NetworkView

    view = NetworkView("NEAT network - play")
    fitness = f" (training fitness {genome.fitness:.0f})" if genome.fitness is not None else ""
    view.show_genome(genome, config,
                     info=f"{path.name}{fitness}   node colour: blue = negative, red = positive value")

    def on_step(step):
        if step % every == 0 and view.is_open():
            view.show_activations(net.values, config)
    return on_step


def main():
    args = parse_args()
    if not args.genome.exists():
        raise SystemExit(f"{args.genome} not found - train first, or pick a car with --genome")
    config = load_config()
    genome = load_genome(args.genome)
    net = make_network(genome, config)
    on_step = live_view(genome, net, config, args.genome, args.viz_every) if args.viz else None

    with DrivingEnv(args.host, args.port, args.town, args.npcs, tm_port=args.tm_port,
                    route_length=args.route_length) as env:
        try:
            for episode in range(1, args.episodes + 1):
                started = time.perf_counter()
                result = drive(env, net, on_step=on_step, playback_speed=args.speed)
                s = result.summary
                print(f"Episode {episode}: score={result.score:.2f}  ended by {result.ending}  "
                      f"({time.perf_counter() - started:.0f}s)\n"
                      f"    route {s['route_completed'] * 100:.0f}% of {s['route_length']:.0f} m, "
                      f"avg {s['average_speed_kmh']:.0f} km/h, "
                      f"steering changes {s['steering_changes_per_s']:.2f}/s, "
                      f"hardest cornering {s['max_lateral_accel']:.1f} m/s²")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
