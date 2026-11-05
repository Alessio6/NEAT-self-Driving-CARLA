"""Watch a trained genome drive in CARLA.

    play.bat                            # v2 best car (best_genome_v2.pkl), real-time speed
    play.bat --version 1                # v1 best car (best_genome.pkl)
    play.bat --genome winner_genome_v2.pkl --episodes 5
    play.bat --speed 2 --viz            # twice as fast, with the network window
"""

import argparse
import os
import pickle
import time

import neat

from carla_env import FIXED_DELTA
from train import HERE, VERSIONS, load_version, output_path


def run_episode_paced(env, net, max_steps, speed, on_step=None):
    """Like train.run_episode, but slowed to `speed` x real time (0 = unlimited)."""
    obs = env.reset()
    total, info = 0.0, {"reason": "time_limit"}
    next_tick = time.perf_counter()
    for step in range(max_steps):
        obs, reward, done, step_info = env.step(net.activate(obs))
        total += reward
        if on_step is not None:
            on_step(step)
        if done:
            info = step_info
            break
        if speed > 0:
            next_tick += FIXED_DELTA / speed
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()  # running behind; don't try to catch up
    return total, info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, choices=sorted(VERSIONS), default=2)
    parser.add_argument("--genome", help="saved genome (default: best_genome[_v2].pkl)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--npcs", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6000, help="v1 only; v2 drives end with the route")
    parser.add_argument("--route-length", type=float, default=2000.0, help="v2: route length in metres")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed: 1 = real time, 2 = twice as fast, 0 = as fast as possible")
    parser.add_argument("--viz", action="store_true", help="show the network and its live signals")
    parser.add_argument("--viz-every", type=int, default=2, help="steps between --viz updates")
    args = parser.parse_args()

    spec, env_class, config, input_labels = load_version(args.version)
    genome_path = args.genome or output_path(spec, "best_genome")
    if not os.path.exists(genome_path):
        raise SystemExit(f"{os.path.basename(genome_path)} not found - train version {args.version} "
                         f"first, or pick a file with --genome")
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    on_step = None
    if args.viz:
        from netviz import NetworkView
        view = NetworkView("NEAT network - play", input_labels=input_labels)
        fitness = f" (training fitness {genome.fitness:.0f})" if genome.fitness is not None else ""
        view.show_genome(genome, config, info=f"{os.path.basename(genome_path)}{fitness}"
                         "   node colour: blue = negative, red = positive value")

        def on_step(step):
            if step % args.viz_every == 0 and view.is_open():
                view.show_activations(net.values, config)

    env_kwargs = {"route_length": args.route_length} if args.version == 2 else {}
    max_steps = args.max_steps if args.version == 1 else 10 ** 9
    env = env_class(args.host, args.port, args.town, args.npcs, no_render=False,
                    tm_port=args.tm_port, **env_kwargs)
    try:
        for ep in range(args.episodes):
            start = time.perf_counter()
            score, info = run_episode_paced(env, net, max_steps, args.speed, on_step)
            line = (f"Episode {ep + 1}: score={score:.2f}  ended by {info['reason']}  "
                    f"({time.perf_counter() - start:.0f}s)")
            if "route_done" in info:
                line += (f"\n    route {info['route_done'] * 100:.0f}% of {env.route_total:.0f} m, "
                         f"avg {info['avg_speed_kmh']:.0f} km/h, "
                         f"steering changes {info['steer_change_per_s']:.2f}/s, "
                         f"hardest cornering {info['max_lat_acc']:.1f} m/s²")
            print(line)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
