"""Live progress bar for a training run.

Reads training_log.txt (written by train.py) and the checkpoint timestamps, so it
can be started or stopped at any time without affecting training:
    python progress.py                 # assumes training goes up to generation 50
    python progress.py --total 100
"""

import argparse
import configparser
import glob
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GEN_RE = re.compile(r"Running generation (\d+)")
GENOME_RE = re.compile(r"gen\s+(\d+) genome\s+(\d+)/(\d+) .*fitness=\s*(-?[\d.]+)\s+end=(\S+)")
BAR_WIDTH = 30


def read_log(path):
    with open(path, "rb") as f:
        text = f.read().replace(b"\x00", b"").decode("utf-8", errors="ignore")
    return text.lstrip("﻿").splitlines()


def parse(lines):
    state = {"gen": None, "done": 0, "pop": None, "best": None, "gen_best": None,
             "finished": False, "survived": 0}
    for line in lines:
        if "RESUMED" in line:
            state["finished"] = False
        m = GEN_RE.search(line)
        if m:
            state.update(gen=int(m.group(1)), done=0, gen_best=None, survived=0)
            continue
        m = GENOME_RE.search(line)
        if m:
            gen, i, pop, fit, end = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)), m.group(5)
            if gen == state["gen"]:
                state["done"], state["pop"] = i, pop
                state["gen_best"] = fit if state["gen_best"] is None else max(state["gen_best"], fit)
                # v1: survived the time limit; v2: completed every route
                state["survived"] += all(e in ("time_limit", "completed") for e in end.split(","))
            state["best"] = fit if state["best"] is None else max(state["best"], fit)
            continue
        if "Training finished" in line or "Stopped by user" in line:
            state["finished"] = True
    return state


def seconds_per_generation(checkpoint_dir, window=5):
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "neat-checkpoint-*")), key=os.path.getmtime)
    times = [os.path.getmtime(f) for f in files[-(window + 1):]]
    # Ignore gaps where training was stopped (e.g. overnight).
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a < 1800]
    return sum(gaps) / len(gaps) if gaps else None


def bar(fraction):
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * BAR_WIDTH))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {fraction * 100:5.1f}%"


def fmt_duration(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def render(state, total, pop_size, sec_per_gen, log_age):
    pop = state["pop"] or pop_size
    gen = state["gen"] or 0
    within = state["done"] / pop if pop else 0.0
    overall = (gen + within) / total
    remaining = None if sec_per_gen is None else max(total - gen - within, 0) * sec_per_gen
    best = "-" if state["best"] is None else f"{state['best']:.1f}"
    gen_best = "-" if state["gen_best"] is None else f"{state['gen_best']:.1f}"

    if state["finished"] or gen >= total:
        status = "Training finished."
    elif log_age > 300:
        status = f"No updates for {fmt_duration(log_age)} - is training still running?"
    else:
        status = f"ETA {fmt_duration(remaining)}  ({fmt_duration(sec_per_gen)} per generation)"

    return [
        f"Overall     {bar(overall)}  generation {gen}/{total}",
        f"Generation  {bar(within)}  car {state['done']}/{pop}, {state['survived']} finished every drive",
        f"Best fitness: {best} overall, {gen_best} this generation",
        status,
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=os.path.join(HERE, "training_log.txt"))
    parser.add_argument("--total", type=int, default=50, help="generation number training stops at")
    parser.add_argument("--refresh", type=float, default=2.0, help="seconds between updates")
    args = parser.parse_args()

    config = configparser.ConfigParser(inline_comment_prefixes=("#",))
    config.read(os.path.join(HERE, "config-neat.txt"))
    pop_size = config.getint("NEAT", "pop_size", fallback=40)
    checkpoint_dir = os.path.join(HERE, "checkpoints")

    os.system("")  # enables ANSI cursor movement in Windows consoles
    printed = 0
    try:
        while True:
            if not os.path.exists(args.log):
                lines = [f"Waiting for {args.log} ..."]
                state = {"finished": False}
            else:
                state = parse(read_log(args.log))
                log_age = time.time() - os.path.getmtime(args.log)
                lines = render(state, args.total, pop_size,
                               seconds_per_generation(checkpoint_dir), log_age)
            if printed:
                sys.stdout.write(f"\x1b[{printed}F")  # move back to the first line
            for line in lines:
                sys.stdout.write("\x1b[2K" + line + "\n")
            sys.stdout.flush()
            printed = len(lines)
            if state.get("finished") or (state.get("gen") or 0) >= args.total:
                break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
