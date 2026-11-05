"""Watch a trained genome drive in CARLA.

    python play.py                      # uses best_genome.pkl
    python play.py --genome winner_genome.pkl --episodes 5
"""

import argparse
import os
import pickle

import neat

from carla_env import CarlaEnv
from train import HERE, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome", default=os.path.join(HERE, "best_genome.pkl"))
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--npcs", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(HERE, "config-neat.txt"))
    with open(args.genome, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    env = CarlaEnv(args.host, args.port, args.town, args.npcs, no_render=False, tm_port=args.tm_port)
    try:
        for ep in range(args.episodes):
            score, reason = run_episode(env, net, None, args.max_steps)
            print(f"Episode {ep + 1}: score={score:.2f}  ended by {reason}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
