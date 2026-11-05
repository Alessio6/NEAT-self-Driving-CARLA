"""Scoring genomes by letting them drive, and the route-length curriculum."""

import json
import random
import time
from dataclasses import dataclass

from . import settings
from .networks import make_network, save_genome


@dataclass
class DriveResult:
    score: float
    ending: str
    summary: dict


def drive(env, net, spawn_index=None, episode_seed=None, on_step=None, playback_speed=0.0):
    """Let `net` drive one route in `env`.

    `on_step(step)` is called after every step (used for the live network view).
    `playback_speed` slows the drive to that multiple of real time; 0 runs as
    fast as possible, which is what training uses.
    """
    observation = env.reset(spawn_index, episode_seed)
    score, step = 0.0, 0
    next_tick = time.perf_counter()
    while True:
        observation, reward, done, info = env.step(net.activate(observation))
        score += reward
        if on_step is not None:
            on_step(step)
        if done:
            return DriveResult(score, info["ending"], info)
        step += 1
        if playback_speed > 0:
            next_tick += settings.FIXED_DELTA / playback_speed
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()  # running behind; don't try to catch up


class Curriculum:
    """Route length, grown each time a generation's best car completes all its routes.

    Saved after every generation so a resumed run carries on where it stopped.
    """

    def __init__(self, path=settings.CURRICULUM_FILE, start=settings.ROUTE_START_LENGTH,
                 maximum=settings.ROUTE_MAX_LENGTH, override=None):
        self.path, self.maximum = path, maximum
        self.length = start
        if override is not None:
            self.length = override
        elif path.exists():
            self.length = json.loads(path.read_text())["route_length"]

    def update(self, best_completed_every_route):
        if best_completed_every_route and self.length < self.maximum:
            self.length = min(self.length * settings.ROUTE_GROWTH, self.maximum)
            print(f"  >> best car completed every route - routes are now {self.length:.0f} m")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"route_length": self.length}))


class GenomeEvaluator:
    """Passed to neat.Population.run: lets every genome in a generation drive.

    All genomes in a generation drive the same start points and routes, so
    their fitness is comparable; the routes change every generation.
    """

    def __init__(self, env, curriculum, *, episodes=settings.EPISODES_PER_GENOME, seed=0,
                 start_generation=0, best_path=settings.BEST_GENOME, best_fitness=float("-inf"),
                 view=None, viz_every=5, picture_dir=None):
        self.env = env
        self.curriculum = curriculum
        self.episodes = episodes
        self.seed = seed
        self.generation = start_generation
        self.best_path = best_path
        self.best_fitness = best_fitness
        self.view = view
        self.viz_every = viz_every
        self.picture_dir = picture_dir
        self.history = []  # (generation, best fitness), for the live view's graph

    def __call__(self, genomes, config):
        rng = random.Random(self.seed + self.generation)
        starts = [(rng.randrange(len(self.env.spawn_points)), rng.randrange(1 << 30))
                  for _ in range(self.episodes)]
        self.env.route_length = self.curriculum.length

        endings = {}
        for i, (genome_id, genome) in enumerate(genomes):
            net = make_network(genome, config)
            on_step = self._live_view(genome, net, config, i, len(genomes))
            results = [drive(self.env, net, spawn, seed, on_step) for spawn, seed in starts]
            genome.fitness = sum(r.score for r in results) / len(results)
            endings[genome_id] = [r.ending for r in results]
            self._report(i, len(genomes), genome_id, genome.fitness, results)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                save_genome(genome, self.best_path)

        best_id, best = max(genomes, key=lambda item: item[1].fitness)
        self.history.append((self.generation, best.fitness))
        self.curriculum.update(all(e == "completed" for e in endings[best_id]))
        self._save_picture(best, config)
        self.generation += 1

    def _report(self, i, count, genome_id, fitness, results):
        # progress.py reads these lines, so keep their format.
        route = sum(r.summary["route_completed"] for r in results) / len(results)
        speed = sum(r.summary["average_speed_kmh"] for r in results) / len(results)
        print(f"  gen {self.generation:3d} genome {i + 1:3d}/{count} id={genome_id:<5d} "
              f"fitness={fitness:8.2f}  end={','.join(r.ending for r in results)}"
              f"  route={route * 100:3.0f}% speed={speed:4.1f}km/h")

    def _live_view(self, genome, net, config, i, count):
        if self.view is None or not self.view.interactive or not self.view.is_open():
            return None
        best = f"   (best so far {self.best_fitness:.0f})" if self.best_fitness > float("-inf") else ""
        self.view.show_genome(genome, config, history=self.history,
                              info=f"Generation {self.generation} - car {i + 1}/{count} driving{best}")

        def on_step(step):
            if step % self.viz_every == 0 and self.view.is_open():
                self.view.show_activations(net.values, config)
        return on_step

    def _save_picture(self, genome, config):
        if self.picture_dir is None or self.view is None or not self.view.is_open():
            return
        self.view.show_genome(genome, config, history=self.history,
                              info=f"Generation {self.generation} best car - fitness {genome.fitness:.1f}")
        self.view.save(self.picture_dir / f"gen_{self.generation:03d}.png")
