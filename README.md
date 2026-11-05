# NEAT self-driving car in CARLA

Evolves neural networks with [NEAT](https://neat-python.readthedocs.io/)
that drive a car through Town04 in the [CARLA](https://carla.org/) simulator
like a real driver: follow a planned route through junctions and traffic
lights, at the speed limit, smoothly, keeping a safe gap to the car ahead.

## Quick start

### 1. Install (once)

1. **CARLA 0.9.15** (Windows build) from the
   [release page](https://github.com/carla-simulator/carla/releases/tag/0.9.15):
   unzip `CARLA_0.9.15.zip` to e.g. `C:\CARLA_0.9.15`. Town04 is included.
   Needs a GPU with about 6 GB of video memory.
2. **Python 3.10 environment** called `carla` (CARLA has no packages for
   newer Pythons):
   ```bash
   conda create -n carla python=3.10 -y
   ```
   ```bash
   conda activate carla
   ```
   ```bash
   pip install -r requirements.txt
   ```
   The launchers below use this environment's Python directly, so plain
   `python` on your PC doesn't need CARLA installed.

### 2. Train

Start the simulator, then training (in this folder):
```bash
C:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe -quality-level=Low
```
```bash
powershell -ExecutionPolicy Bypass -File .\run_training.ps1 -Generations 100 -Npcs 0 -NoRender
```

- `-NoRender` skips graphics, which is faster; `-Npcs` sets the amount of traffic.
- Continue a run with `-Resume checkpoints\neat-checkpoint-N` (training
  stops at generation N + `-Generations`).
- If training crashes, the script restarts CARLA if it has frozen and
  resumes from the latest checkpoint by itself (up to 20 times). If CARLA
  isn't in `C:\CARLA_0.9.15`, pass `-CarlaPath <path to CarlaUE4.exe>`.
- Ctrl+C stops it; the best car so far is always in `models\best_genome.pkl`.

A live progress bar with an ETA, in a second terminal (it only reads the log):
```bash
& "$env:USERPROFILE\anaconda3\envs\carla\python.exe" progress.py --total 100
```

**Suggested plan:** stage 1 without traffic (`-Npcs 0`) until routes have
grown to about 2 km, then stage 2 resuming with traffic (`-Npcs 30`).

### 3. Watch

```bash
.\play.bat
```
Options: `--npcs 0` (no traffic), `--episodes 5`, `--speed 2` (faster
playback), `--route-length 1000`, `--viz` (network window),
`--genome models\winner_genome.pkl`. After each drive it prints the share of
the route completed, average speed, steering changes per second and the
hardest cornering.

## How the car learns

**The course.** Each drive follows a route planned from a spawn point,
taking random turns at junctions. Routes start at 400 m and grow by 50% every
time a generation's best car completes all of its routes, up to 2 km. The
current length is kept in `checkpoints\curriculum.json`, so a resumed run
carries on from there.

**What it sees (26 inputs):**

| Inputs | |
|---|---|
| 16 LiDAR rays | distance to the nearest obstacle in each 22.5° sector around the car, up to 50 m |
| speed, speed limit | m/s, scaled by 25 m/s |
| heading error, lane offset | relative to the route's lane |
| route 5 / 15 / 30 m | direction of the route ahead, so it can slow for bends and turn at junctions |
| prev steer, prev pedal | its own last controls |
| red light | facing a red or yellow light |

**What it controls (2 outputs):** steering, and throttle (positive) / brake
(negative). Both change at realistic rates: the wheel takes ~0.8 s to turn
from centre to full lock.

**What it's rewarded for, each step:**

| | |
|---|---|
| progress along the route | +1 per metre (half credit above 110% of the speed limit) |
| speeding | −0.02 per m/s over 110% of the limit |
| jerky steering / pedals | −1.0 / −0.3 per unit of change |
| hard cornering | −0.02 per m/s² of sideways acceleration above 3 m/s² |
| drifting from the lane centre | −0.02 per lane half-width |
| tailgating | −0.05 per second short of a 1.5 s gap |
| moving towards a red light | −0.2 |

**How a drive ends:** finishing the route (+50), a collision (−50), leaving
the route (−25), being stopped on a clear road for 3 s (−25), waiting more
than 60 s behind traffic or at a light, or running out of time (an average
of 3 m/s). Waiting in traffic or at a red light doesn't count as stuck.

**Reliability.** Each car drives 3 routes per generation and its fitness is
the average, so a lucky start can't carry a bad driver. All cars in a
generation drive the same routes, so their scores are comparable.

Every number above lives in `neat_driver\settings.py`; NEAT's own settings
(population, mutation rates) are in `config-neat.txt`.

## Network visualisation

- `play.bat --viz` opens a window with the trained network. Nodes are
  coloured by their current value (blue = negative, red = positive) and
  connections by the signal flowing through them, so you can see which rays
  and lane inputs are steering the car.
- `-Viz` on `run_training.ps1` shows each car's network while it drives, with
  the best fitness per generation underneath. It slows training; closing the
  window turns it off.
- `-VizSave viz` saves the best network of each generation as
  `viz\gen_000.png`, `gen_001.png`, ... at almost no cost.

In the structure view, green connections have positive weights and red ones
negative; thicker means a larger weight. Grey hidden nodes don't affect the
outputs yet.

## Project layout

```
train.py, play.py, progress.py   command-line entry points
run_training.ps1, play.bat       launchers that use the carla env's Python
config-neat.txt                  NEAT settings
neat_driver/
    settings.py                  every tunable number and file location
    simulator.py                 CARLA connection, synchronous stepping, traffic
    ego_car.py                   the driven car with its LiDAR and collision sensor
    lidar.py                     point cloud -> 16 distance rays
    route.py                     route planning and progress along it
    rewards.py                   per-step reward
    env.py                       the driving task: observations, rewards, endings
    networks.py                  NEAT config, saving and loading cars
    evaluation.py                driving genomes, the route-length curriculum
    netviz.py                    network window
    training_log.py              log reading for the progress bar
    healthcheck.py               does CARLA answer? (used by run_training.ps1)
models/                          trained cars (models/v1: cars from the first version)
logs/training_log.txt            output of every training run
checkpoints/                     NEAT checkpoints (not in git)
tests/                           unit tests
```

Run the tests (no simulator needed):
```bash
& "$env:USERPROFILE\anaconda3\envs\carla\python.exe" -m unittest discover -s tests -t .
```

## Troubleshooting

- **`No module named 'carla'`** — that Python isn't the `carla` env. Use
  `play.bat` / `run_training.ps1`, or the env's `python.exe` directly.
- **Training or play can't connect / times out** — CARLA isn't running, is
  still loading, or has frozen (it can after a graphics driver hiccup).
  Restart `CarlaUE4.exe`; `run_training.ps1` does this by itself.
- **CARLA keeps freezing** (Windows logs graphics-driver "LiveKernelEvent
  193" errors) — try training without `-NoRender`, and update the graphics
  driver. Long runs with graphics on have been stable.
- **`.ps1` scripts are blocked** — run them through
  `powershell -ExecutionPolicy Bypass -File ...` as shown above.
- Don't run `play.bat` while training is running: both would drive the same
  simulator.

## History

The first version (v1) only had to survive 60 seconds in its lane, with 20
inputs and no route. Its cars are kept in `models/v1/` but need the v1 code
to drive; that code is in the git history before the clean rewrite.
