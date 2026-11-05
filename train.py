"""Evolve a self-driving car in CARLA with NEAT.

Start the CARLA server first (CarlaUE4.exe), then run:
    python train.py --generations 50
"""

import argparse
import os
import pickle
import random

import neat

from carla_env import CarlaEnv, NUM_INPUTS

HERE = os.path.dirname(os.path.abspath(__file__))


def run_episode(env, net, spawn_index, max_steps):
    obs = env.reset(spawn_index)
    total = 0.0
    reason = "time_limit"
    for _ in range(max_steps):
        obs, reward, done, info = env.step(net.activate(obs))
        total += reward
        if done:
            reason = info["reason"]
            break
    return total, reason


def make_eval_genomes(env, args, best_path, start_generation=0):
    state = {"generation": start_generation, "best_fitness": float("-inf")}
    # When resuming, keep the saved best unless something beats it.
    if args.resume and os.path.exists(best_path):
        with open(best_path, "rb") as f:
            saved = pickle.load(f)
        if saved.fitness is not None:
            state["best_fitness"] = saved.fitness

    def eval_genomes(genomes, config):
        # Every genome in a generation drives from the same start points, so
        # fitness is comparable; the start points change each generation.
        rng = random.Random(args.seed + state["generation"])
        spawns = [rng.randrange(len(env.spawn_points)) for _ in range(args.episodes)]

        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            scores, reasons = [], []
            for spawn in spawns:
                score, reason = run_episode(env, net, spawn, args.max_steps)
                scores.append(score)
                reasons.append(reason)
            genome.fitness = sum(scores) / len(scores)
            print(f"  gen {state['generation']:3d} genome {i + 1:3d}/{len(genomes)} "
                  f"id={genome_id:<5d} fitness={genome.fitness:8.2f}  end={','.join(reasons)}")

            if genome.fitness > state["best_fitness"]:
                state["best_fitness"] = genome.fitness
                with open(best_path, "wb") as f:
                    pickle.dump(genome, f)
        state["generation"] += 1

    return eval_genomes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town04", help="map to load ('' keeps the current one)")
    parser.add_argument("--npcs", type=int, default=30, help="number of traffic vehicles")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=1, help="drives per genome per generation")
    parser.add_argument("--max-steps", type=int, default=1200, help="steps per drive (20 steps = 1 s)")
    parser.add_argument("--no-render", action="store_true", help="disable rendering for faster training")
    parser.add_argument("--resume", help="path to a neat-checkpoint-N file to continue from")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(HERE, "config-neat.txt"))
    assert config.genome_config.num_inputs == NUM_INPUTS, \
        f"config-neat.txt num_inputs must be {NUM_INPUTS}"

    checkpoint_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(HERE, "best_genome.pkl")

    if args.resume:
        population = neat.Checkpointer.restore_checkpoint(args.resume)
    else:
        population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(
        1, filename_prefix=os.path.join(checkpoint_dir, "neat-checkpoint-")))

    env = CarlaEnv(args.host, args.port, args.town, args.npcs,
                   no_render=args.no_render, tm_port=args.tm_port, seed=args.seed)
    try:
        eval_genomes = make_eval_genomes(env, args, best_path, population.generation)
        winner = population.run(eval_genomes, args.generations)
        with open(os.path.join(HERE, "winner_genome.pkl"), "wb") as f:
            pickle.dump(winner, f)
        print(f"\nTraining finished. Winner fitness: {winner.fitness:.2f}")
    except KeyboardInterrupt:
        print("\nStopped by user. Best genome so far is in best_genome.pkl")
    finally:
        env.close()


if __name__ == "__main__":
    main()
