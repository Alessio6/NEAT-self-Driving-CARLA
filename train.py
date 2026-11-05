"""Train self-driving cars in CARLA with NEAT.

Start CARLA first. run_training.ps1 runs this with the right Python, logs the
output and restarts after crashes:
    powershell -ExecutionPolicy Bypass -File .\\run_training.ps1 -Generations 100 -Npcs 0 -NoRender

Or directly:
    python train.py --generations 100 --npcs 0 --no-render
    python train.py --resume checkpoints/neat-checkpoint-40 --npcs 30
"""

import argparse
from pathlib import Path

import neat

from neat_driver import settings
from neat_driver.env import DrivingEnv
from neat_driver.evaluation import Curriculum, GenomeEvaluator
from neat_driver.networks import load_config, load_genome, save_genome


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=50, help="how many generations to run")
    parser.add_argument("--resume", type=Path, help="checkpoint to continue from, e.g. checkpoints/neat-checkpoint-40")
    parser.add_argument("--npcs", type=int, default=30, help="number of traffic vehicles")
    parser.add_argument("--episodes", type=int, default=settings.EPISODES_PER_GENOME,
                        help="routes each car drives per generation")
    parser.add_argument("--route-length", type=float,
                        help="route length in metres (default: carry on the curriculum)")
    parser.add_argument("--max-route-length", type=float, default=settings.ROUTE_MAX_LENGTH,
                        help="longest route the curriculum grows to")
    parser.add_argument("--no-render", action="store_true", help="no graphics: faster training")
    parser.add_argument("--viz", action="store_true",
                        help="live window of each car's network while it drives (slows training)")
    parser.add_argument("--viz-every", type=int, default=5, help="steps between --viz updates")
    parser.add_argument("--viz-save", type=Path, metavar="DIR",
                        help="save a picture of each generation's best network into DIR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--town", default=settings.TOWN, help="map to load ('' keeps the current one)")
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--tm-port", type=int, default=settings.TRAFFIC_MANAGER_PORT)
    return parser.parse_args()


def make_population(config, resume):
    if resume:
        population = neat.Checkpointer.restore_checkpoint(str(resume))
    else:
        population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    settings.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    population.add_reporter(neat.Checkpointer(
        1, filename_prefix=str(settings.CHECKPOINT_DIR / settings.CHECKPOINT_PREFIX)))
    return population


def main():
    args = parse_args()
    config = load_config()
    population = make_population(config, args.resume)

    # When resuming, the saved best car is only replaced by a better one.
    best_fitness = float("-inf")
    if args.resume and settings.BEST_GENOME.exists():
        saved = load_genome(settings.BEST_GENOME)
        if saved.fitness is not None:
            best_fitness = saved.fitness

    view = None
    if args.viz or args.viz_save:
        from neat_driver.netviz import NetworkView
        view = NetworkView("NEAT training", interactive=args.viz, show_history=True)

    curriculum = Curriculum(maximum=args.max_route_length, override=args.route_length)
    print(f"Training: {args.episodes} routes per car, routes are {curriculum.length:.0f} m")

    with DrivingEnv(args.host, args.port, args.town, args.npcs, no_render=args.no_render,
                    tm_port=args.tm_port, seed=args.seed, route_length=curriculum.length) as env:
        evaluator = GenomeEvaluator(env, curriculum, episodes=args.episodes, seed=args.seed,
                                    start_generation=population.generation, best_fitness=best_fitness,
                                    view=view, viz_every=args.viz_every, picture_dir=args.viz_save)
        try:
            winner = population.run(evaluator, args.generations)
        except KeyboardInterrupt:
            print(f"\nStopped by user. Best car so far is in {settings.BEST_GENOME.relative_to(settings.PROJECT_ROOT)}")
            return
    save_genome(winner, settings.WINNER_GENOME)
    print(f"\nTraining finished. Winner fitness: {winner.fitness:.2f}")


if __name__ == "__main__":
    main()
