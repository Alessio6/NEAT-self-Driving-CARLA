"""Evolve a self-driving car in CARLA with NEAT.

Start the CARLA server first (CarlaUE4.exe), then run:
    python train.py --generations 50              # v2: complete routes like a real driver
    python train.py --version 1 --generations 50  # v1: original survive-60-seconds setup
"""

import argparse
import importlib
import json
import os
import pickle
import random

import neat

HERE = os.path.dirname(os.path.abspath(__file__))

# Each version has its own environment, NEAT config and output files, so
# networks from one version never get mixed with the other.
VERSIONS = {
    1: {"env": ("carla_env", "CarlaEnv"), "config": "config-neat.txt", "suffix": "",
        "checkpoints": "checkpoints", "episodes": 1, "max_steps": 1200},
    2: {"env": ("carla_env_v2", "RouteEnv"), "config": "config-neat-v2.txt", "suffix": "_v2",
        "checkpoints": "checkpoints_v2", "episodes": 3, "max_steps": None},
}


def load_version(version):
    """Return (spec, env class, NEAT config, input labels) for a version."""
    spec = VERSIONS[version]
    module = importlib.import_module(spec["env"][0])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(HERE, spec["config"]))
    assert config.genome_config.num_inputs == module.NUM_INPUTS, \
        f"{spec['config']} num_inputs must be {module.NUM_INPUTS}"
    return spec, getattr(module, spec["env"][1]), config, getattr(module, "INPUT_LABELS", None)


def output_path(spec, name):
    return os.path.join(HERE, f"{name}{spec['suffix']}.pkl")


def run_episode(env, net, spawn_index, max_steps, on_step=None, episode_seed=None):
    obs = env.reset(spawn_index, episode_seed)
    total = 0.0
    info = {"reason": "time_limit"}
    for step in range(max_steps):
        obs, reward, done, step_info = env.step(net.activate(obs))
        total += reward
        if on_step is not None:
            on_step(step)
        if done:
            info = step_info
            break
    return total, info


class Curriculum:
    """Route length for v2, grown as the best cars start completing their routes."""

    def __init__(self, path, start, maximum, override=None):
        self.path, self.maximum = path, maximum
        self.length = start
        if override is not None:
            self.length = override
        elif os.path.exists(path):
            with open(path) as f:
                self.length = json.load(f)["route_length"]

    def update(self, best_completed_all):
        if best_completed_all and self.length < self.maximum:
            self.length = min(self.length * 1.5, self.maximum)
            print(f"  >> best car completed every route - routes are now {self.length:.0f} m")
        with open(self.path, "w") as f:
            json.dump({"route_length": self.length}, f)


def make_eval_genomes(env, args, best_path, start_generation=0, view=None, curriculum=None):
    state = {"generation": start_generation, "best_fitness": float("-inf"), "history": []}
    # When resuming, keep the saved best unless something beats it.
    if args.resume and os.path.exists(best_path):
        with open(best_path, "rb") as f:
            saved = pickle.load(f)
        if saved.fitness is not None:
            state["best_fitness"] = saved.fitness

    def eval_genomes(genomes, config):
        # Every genome in a generation drives the same start points and routes,
        # so fitness is comparable; they change each generation.
        rng = random.Random(args.seed + state["generation"])
        starts = [(rng.randrange(len(env.spawn_points)), rng.randrange(1 << 30))
                  for _ in range(args.episodes)]
        if curriculum is not None:
            env.route_length = curriculum.length

        gen = state["generation"]
        live = view is not None and args.viz
        reasons_by_genome = {}
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            on_step = None
            if live and view.is_open():
                best = state["best_fitness"]
                view.show_genome(genome, config, history=state["history"],
                                 info=f"Generation {gen} - car {i + 1}/{len(genomes)} driving"
                                      + (f"   (best so far {best:.0f})" if best > float("-inf") else ""))

                def on_step(step, net=net):
                    if step % args.viz_every == 0 and view.is_open():
                        view.show_activations(net.values, config)
            scores, infos = [], []
            for spawn, episode_seed in starts:
                score, info = run_episode(env, net, spawn, args.max_steps, on_step, episode_seed)
                scores.append(score)
                infos.append(info)
            genome.fitness = sum(scores) / len(scores)
            reasons = [info["reason"] for info in infos]
            reasons_by_genome[genome_id] = reasons
            extra = ""
            if "route_done" in infos[0]:
                route = sum(x["route_done"] for x in infos) / len(infos)
                speed = sum(x["avg_speed_kmh"] for x in infos) / len(infos)
                extra = f"  route={route * 100:3.0f}% speed={speed:4.1f}km/h"
            print(f"  gen {gen:3d} genome {i + 1:3d}/{len(genomes)} "
                  f"id={genome_id:<5d} fitness={genome.fitness:8.2f}  end={','.join(reasons)}{extra}")

            if genome.fitness > state["best_fitness"]:
                state["best_fitness"] = genome.fitness
                with open(best_path, "wb") as f:
                    pickle.dump(genome, f)

        gen_best_id, gen_best = max(genomes, key=lambda item: item[1].fitness)
        state["history"].append((gen, gen_best.fitness))
        if curriculum is not None:
            curriculum.update(all(r == "completed" for r in reasons_by_genome[gen_best_id]))
        if args.viz_save and view is not None and view.is_open():
            view.show_genome(gen_best, config, history=state["history"],
                             info=f"Generation {gen} best car - fitness {gen_best.fitness:.1f}")
            view.save(os.path.join(args.viz_save, f"gen_{gen:03d}.png"))
        state["generation"] += 1

    return eval_genomes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, choices=sorted(VERSIONS), default=2,
                        help="2 = route driving (default), 1 = original setup")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--town", default="Town04", help="map to load ('' keeps the current one)")
    parser.add_argument("--npcs", type=int, default=30, help="number of traffic vehicles")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--episodes", type=int, help="drives per genome per generation (v1: 1, v2: 3)")
    parser.add_argument("--max-steps", type=int,
                        help="steps per drive, 20 steps = 1 s (v1: 1200; v2: set by route length)")
    parser.add_argument("--route-length", type=float,
                        help="v2: route length in metres (default: continue the curriculum, starting at 400)")
    parser.add_argument("--max-route-length", type=float, default=2000.0,
                        help="v2: longest route the curriculum grows to")
    parser.add_argument("--no-render", action="store_true", help="disable rendering for faster training")
    parser.add_argument("--resume", help="path to a neat-checkpoint-N file to continue from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz", action="store_true",
                        help="show a live window of each car's network while it drives (slows training)")
    parser.add_argument("--viz-every", type=int, default=5, help="steps between --viz updates")
    parser.add_argument("--viz-save", metavar="DIR",
                        help="save a picture of each generation's best network into DIR")
    args = parser.parse_args()

    spec, env_class, config, input_labels = load_version(args.version)
    if args.episodes is None:
        args.episodes = spec["episodes"]
    if args.max_steps is None:
        args.max_steps = spec["max_steps"] or 10 ** 9  # v2 drives end via the route time limit

    checkpoint_dir = os.path.join(HERE, spec["checkpoints"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = output_path(spec, "best_genome")

    if args.resume:
        population = neat.Checkpointer.restore_checkpoint(args.resume)
    else:
        population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(
        1, filename_prefix=os.path.join(checkpoint_dir, "neat-checkpoint-")))

    view = None
    if args.viz or args.viz_save:
        from netviz import NetworkView
        if args.viz_save:
            os.makedirs(args.viz_save, exist_ok=True)
        view = NetworkView("NEAT training", interactive=args.viz, show_history=True,
                           input_labels=input_labels)

    curriculum = None
    env_kwargs = {}
    if args.version == 2:
        curriculum = Curriculum(os.path.join(checkpoint_dir, "curriculum.json"),
                                400.0, args.max_route_length, args.route_length)
        env_kwargs["route_length"] = curriculum.length
        print(f"Version 2: {args.episodes} routes per car, routes are {curriculum.length:.0f} m")

    env = env_class(args.host, args.port, args.town, args.npcs, no_render=args.no_render,
                    tm_port=args.tm_port, seed=args.seed, **env_kwargs)
    try:
        eval_genomes = make_eval_genomes(env, args, best_path, population.generation, view, curriculum)
        winner = population.run(eval_genomes, args.generations)
        with open(output_path(spec, "winner_genome"), "wb") as f:
            pickle.dump(winner, f)
        print(f"\nTraining finished. Winner fitness: {winner.fitness:.2f}")
    except KeyboardInterrupt:
        print(f"\nStopped by user. Best genome so far is in {os.path.basename(best_path)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
