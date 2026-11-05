"""Reading the training log, for the progress bar."""

import re
import time
from dataclasses import dataclass

from . import settings

GENERATION_LINE = re.compile(r"Running generation (\d+)")
GENOME_LINE = re.compile(r"gen\s+(\d+) genome\s+(\d+)/(\d+) .*fitness=\s*(-?[\d.]+)\s+end=(\S+)")
RUN_START = ("===== STARTED", "===== RESUMED")
FINISHED = ("Training finished", "Stopped by user")
SUCCESSFUL_ENDINGS = {"completed", "time_limit"}  # time_limit: v1 logs, where surviving was the goal
BAR_WIDTH = 30


@dataclass
class Progress:
    generation: int | None = None
    cars_done: int = 0
    population: int | None = None
    best: float | None = None             # best fitness this run
    generation_best: float | None = None
    finished_every_drive: int = 0         # cars this generation that completed all their routes
    finished: bool = False


def read_lines(path):
    """The log's lines; tolerates the mix of encodings PowerShell can write."""
    data = path.read_bytes().replace(b"\x00", b"")
    return data.decode("utf-8", errors="ignore").lstrip("﻿").splitlines()


def parse(lines):
    progress = Progress()
    for line in lines:
        if line.startswith(RUN_START):
            progress.finished = False
            if line.startswith("===== STARTED"):
                progress.best = None  # a new run, not a continuation
            continue
        match = GENERATION_LINE.search(line)
        if match:
            progress.generation = int(match.group(1))
            progress.cars_done, progress.generation_best, progress.finished_every_drive = 0, None, 0
            continue
        match = GENOME_LINE.search(line)
        if match:
            generation, car, population = int(match.group(1)), int(match.group(2)), int(match.group(3))
            fitness, endings = float(match.group(4)), match.group(5).split(",")
            if generation == progress.generation:
                progress.cars_done, progress.population = car, population
                progress.generation_best = _max(progress.generation_best, fitness)
                progress.finished_every_drive += all(e in SUCCESSFUL_ENDINGS for e in endings)
            progress.best = _max(progress.best, fitness)
            continue
        if line.startswith(FINISHED):
            progress.finished = True
    return progress


def _max(current, value):
    return value if current is None else max(current, value)


def seconds_per_generation(checkpoint_dir=settings.CHECKPOINT_DIR, window=5):
    """Average time between the last few checkpoints, ignoring long pauses."""
    files = sorted(checkpoint_dir.glob(f"{settings.CHECKPOINT_PREFIX}*"), key=lambda p: p.stat().st_mtime)
    times = [p.stat().st_mtime for p in files[-(window + 1):]]
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a < 1800]
    return sum(gaps) / len(gaps) if gaps else None


def bar(fraction):
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * BAR_WIDTH))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {fraction * 100:5.1f}%"


def duration(seconds):
    if seconds is None:
        return "?"
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def render(progress, total, population, sec_per_generation, log_age):
    """Four lines of text: overall bar, generation bar, best fitness, status."""
    population = progress.population or population
    generation = progress.generation or 0
    within = progress.cars_done / population if population else 0.0
    overall = (generation + within) / total
    remaining = None if sec_per_generation is None else max(total - generation - within, 0) * sec_per_generation
    best = "-" if progress.best is None else f"{progress.best:.1f}"
    generation_best = "-" if progress.generation_best is None else f"{progress.generation_best:.1f}"

    if progress.finished or generation >= total:
        status = "Training finished."
    elif log_age > 300:
        status = f"No updates for {duration(log_age)} - is training still running?"
    else:
        status = f"ETA {duration(remaining)}  ({duration(sec_per_generation)} per generation)"
    return [
        f"Overall     {bar(overall)}  generation {generation}/{total}",
        f"Generation  {bar(within)}  car {progress.cars_done}/{population}, "
        f"{progress.finished_every_drive} finished every drive",
        f"Best fitness: {best} this run, {generation_best} this generation",
        status,
    ]


def log_age(path):
    return time.time() - path.stat().st_mtime
