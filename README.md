# NEAT self-driving car in CARLA

Evolves neural networks with NEAT (neat-python) that drive a car in CARLA,
keeping lane and avoiding traffic, using LiDAR.

## Files

| File | Purpose |
|---|---|
| `carla_env.py` | CARLA wrapper: spawns the car, LiDAR, collision sensor and NPC traffic; builds observations and rewards |
| `config-neat.txt` | NEAT hyper-parameters (population, mutation rates, inputs/outputs) |
| `train.py` | Runs evolution; saves `best_genome.pkl` and checkpoints in `checkpoints/` |
| `play.py` | Replays a saved genome with the camera following the car |
| `run_training.ps1` | Starts `train.py` in the conda env and saves its output to `training_log.txt` |
| `progress.py` | Live progress bar and ETA for a training run, read from `training_log.txt` |

## Setup (one time)

1. **Install the CARLA simulator 0.9.15** (Windows build) from
   https://github.com/carla-simulator/carla/releases/tag/0.9.15 —
   download `CARLA_0.9.15.zip` (Town01–05 and Town10 are included; the
   AdditionalMaps zip is only needed for other towns) and unzip it, e.g. to
   `C:\CARLA_0.9.15`. Needs a GPU with ~6 GB+ VRAM.
2. The Python env already exists: conda env **`carla`** (Python 3.10) with
   `carla 0.9.15`, `neat-python 2.0.0`, `numpy`. To recreate it:
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
2. Train (in this folder, with `conda activate carla`):
   ```bash
   python train.py --generations 50
   ```
   Faster, with no graphics: `python train.py --no-render`.
   Continue later: `python train.py --resume checkpoints/neat-checkpoint-12`.
   Press Ctrl+C at any time; the best genome so far is kept in `best_genome.pkl`.
   To see a live progress bar with an ETA, run this in a second terminal
   (it only reads `training_log.txt`, so it can be started or closed any time;
   use `--total N` if training stops at a generation other than 50):
   ```bash
   python progress.py
   ```
   This needs the training output saved to `training_log.txt`. The easiest
   way is to start training with the launcher script, which does that for you:
   ```bash
   powershell -ExecutionPolicy Bypass -File .\run_training.ps1 -Resume checkpoints\neat-checkpoint-52 -Generations 28 -Npcs 30
   ```
3. Watch the best driver:
   ```bash
   python play.py
   ```

## How it works

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
