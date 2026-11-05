# NEAT self-driving car in CARLA

Evolves neural networks with NEAT (neat-python) that drive a car in CARLA
using LiDAR. There are two versions:

- **v2 (default)** — drive a planned route through Town04 (junctions, traffic
  lights, traffic) like a real driver: complete the course at the speed
  limit, smoothly, keeping a safe gap. See [Version 2](#version-2-driving-like-a-real-driver).
- **v1** — the original setup: survive 60 seconds in lane. Its trained cars
  (`best_genome.pkl`, `best_genome_no_traffic.pkl`) still work with `--version 1`.

## Files

| File | Purpose |
|---|---|
| `carla_env.py` | v1 CARLA wrapper: spawns the car, LiDAR, collision sensor and NPC traffic; builds observations and rewards |
| `carla_env_v2.py` | v2 route-driving environment (builds on `carla_env.py`) |
| `config-neat.txt`, `config-neat-v2.txt` | NEAT hyper-parameters for v1 / v2 |
| `train.py` | Runs evolution; saves `best_genome[_v2].pkl` and checkpoints in `checkpoints[_v2]/` |
| `play.py` | Replays a saved genome with the camera following the car |
| `play.bat` | Runs `play.py` with the conda env's Python (no `conda activate` needed) |
| `run_training.ps1` | Starts `train.py` in the conda env and saves its output to `training_log.txt` |
| `progress.py` | Live progress bar and ETA for a training run, read from `training_log.txt` |
| `netviz.py` | Draws the NEAT network and its live signals (used by `--viz` in training and play) |

## Setup (one time)

1. **Install the CARLA simulator 0.9.15** (Windows build) from
   https://github.com/carla-simulator/carla/releases/tag/0.9.15 —
   download `CARLA_0.9.15.zip` (Town01–05 and Town10 are included; the
   AdditionalMaps zip is only needed for other towns) and unzip it, e.g. to
   `C:\CARLA_0.9.15`. Needs a GPU with ~6 GB+ VRAM.
2. The Python env already exists: conda env **`carla`** (Python 3.10) with
   `carla 0.9.15`, `neat-python 2.0.0`, `numpy`, `matplotlib`. To recreate it:
   ```bash
   conda create -n carla python=3.10 -y
   ```
   ```bash
   conda activate carla
   ```
   ```bash
   pip install -r requirements.txt
   ```

## Run

1. Start the simulator (low quality is faster):
   ```bash
   C:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe -quality-level=Low
   ```
2. Train with the launcher script, which uses the `carla` env's Python and
   saves the output to `training_log.txt`:
   ```bash
   powershell -ExecutionPolicy Bypass -File .\run_training.ps1 -Generations 100 -Npcs 0 -NoRender
   ```
   Options: `-Version 1` for the original setup, `-Resume checkpoints_v2\neat-checkpoint-N`
   to continue, `-NoRender` for faster training without graphics, `-Viz` /
   `-VizSave viz` for the network visualisation.
   Press Ctrl+C at any time; the best genome so far is kept in `best_genome_v2.pkl`.
   To see a live progress bar with an ETA, run this in a second terminal
   (it only reads `training_log.txt`, so it can be started or closed any time;
   use `--total N` to match the generation training stops at):
   ```bash
   & "$env:USERPROFILE\anaconda3\envs\carla\python.exe" progress.py --total 100
   ```
3. Watch the best driver (`play.bat` uses the `carla` env's Python, so plain
   `python` - which may be a different Python without CARLA - isn't needed):
   ```bash
   .\play.bat
   ```
   Options: `--npcs 0` for no traffic, `--episodes 5`, `--speed 2` for
   faster playback, `--route-length 1000`, or `--version 1` (optionally with
   `--genome best_genome_no_traffic.pkl`) for the v1 cars.

## Version 2: driving like a real driver

**The course.** Each drive follows a route planned through the road network,
taking random turns at junctions. A car is scored on how much of the route it
completes (+50 bonus for finishing). Routes start at 400 m and grow by 50%
each time a generation's best car completes all of its routes, up to 2 km
(`--max-route-length`). The current length is kept in
`checkpoints_v2/curriculum.json`, so resuming carries on from there.

**What it sees (26 inputs).** 16 LiDAR rays (range 50 m), speed, the road's
speed limit, heading error and offset from the route lane, the direction of
the route 5, 15 and 30 m ahead (so it can slow for bends and turn at
junctions), its own previous steering and pedal, and whether it faces a red
light.

**What it is rewarded for**, per step:

| | |
|---|---|
| progress along the route | +1 per metre (halved above 110% of the speed limit) |
| speeding | −0.02 per m/s over 110% of the limit |
| jerky steering / pedal changes | −1.0 / −0.3 per unit of change |
| hard cornering | −0.02 per m/s² of sideways acceleration above 3 m/s² |
| drifting from the lane centre | small penalty |
| tailgating | penalty when less than 1.5 s behind the vehicle ahead |
| moving towards a red light | −0.2 per step |

A drive ends on finishing (+50), a collision (−50), leaving the route (−25),
being stuck on a clear road for 3 s (−25), waiting over 60 s, or running out
of time (an average of 3 m/s). Waiting behind traffic or at a red light is
not counted as stuck. Steering and pedals have realistic rates of change, so
the wheel can't snap from one side to the other.

**Reliability.** Each car drives 3 different routes per generation and its
fitness is the average, so a lucky start can't carry a bad driver.

**Suggested plan.** Stage 1 without traffic until routes reach ~2 km, then
stage 2 with traffic, resuming from the latest checkpoint:
```bash
powershell -ExecutionPolicy Bypass -File .\run_training.ps1 -Resume checkpoints_v2\neat-checkpoint-N -Generations 100 -Npcs 30 -NoRender
```
`play.bat` prints driver-style stats after each drive: % of the route
completed, average speed, steering changes per second and hardest cornering.

## Network visualisation

Add `--viz` to see the NEAT network in a separate window.

- **Play mode** (`.\play.bat --viz`): the trained network with live signals.
  Nodes are coloured by their current value (blue = negative, red =
  positive) and connections by the signal flowing through them, so you can
  see which LiDAR rays and lane inputs are driving the steering and
  throttle/brake values shown on the right.
- **Training** (`-Viz` on `run_training.ps1`, or `--viz` on `train.py`): each
  car's network while it drives, with a graph of the best fitness per
  generation underneath. Watch hidden nodes and connections appear as NEAT
  adds structure. This slows training noticeably; raise `--viz-every`
  (default 5 steps) to reduce the cost, or close the window to turn it off.
- **Pictures per generation** (`-VizSave viz` / `--viz-save viz`): saves the
  best network of each generation as `viz/gen_000.png`, `gen_001.png`, ... —
  works without the live window and costs almost nothing.

In the structure view, green connections have positive weights and red
negative; thicker means a larger weight. Inputs are on the left, hidden nodes
in the middle (grey ones don't affect the outputs yet), outputs on the right.

## How v1 works

**Inputs (20):** 16 LiDAR "rays" — the LiDAR point cloud is split into 16
sectors around the car, and each input is the distance to the nearest obstacle
in that sector (0 = touching, 1 = clear for 30 m; road-surface hits are
ignored). Plus 4 lane features: speed, heading error to the lane, offset from
lane centre and the angle to a point 10 m ahead on the lane. LiDAR can't see
lane markings, so these lane features are what make lane keeping possible.

**Outputs (2):** steering and throttle/brake (positive = throttle, negative = brake).

**Fitness:** metres travelled along the lane direction, minus a small penalty
for leaving the lane centre. An episode ends on a collision (−25), leaving the
lane (−10), being stuck for 3 s (−10), or the step limit (60 s by default). Speed
above ~43 km/h earns half credit. All genomes in a generation start from the
same spawn points so their scores are comparable.

## Tuning tips

- Start with `--npcs 0` so it learns to stay on the road, then resume from a
  checkpoint with traffic (`--npcs 30`).
- `--episodes 2` gives more reliable fitness but takes twice as long.
- Reward constants are at the top of `carla_env.py`; population size and
  mutation rates are in `config-neat.txt`.
- One generation of 40 genomes takes roughly 5–40 minutes, depending on how
  long they survive and on your GPU. Expect useful driving after 20–50+ generations.
