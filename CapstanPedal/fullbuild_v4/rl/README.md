# RL walking controller for the v4 quadruped

A learned walking policy for the v4 URDF (`../urdf/capstan_quad.urdf`, through `../mujoco/model.py`). It tracks a
commanded forward speed, sideways speed and yaw rate. It is trained on the GPU with thousands of MuJoCo Warp worlds,
then checked and driven in the full CPU model.

| File | |
|---|---|
| `quad.py` | The robot for learning: output-space PD at 500 Hz, the motor model (31.5 N·m, torque-speed line, output A's 25 N·m push limit), the software joint limits, the 50-D observation. CPU, full model |
| `mjx_walk.py` | The same task on the GPU (MJX, Warp backend). Visual meshes dropped, rotors welded (their inertia is already the armature). Episodes are configurations from a bank |
| `train_gpu.py` | PPO in JAX with Success-Guided Sampling of the configurations. Writes `policies/<name>.npz` / `.pkl`, `logs/<name>.csv` |
| `policy.py` | A trained policy in numpy only (no JAX/torch), for `play.py` and `teleop.py` |
| `play.py` | Evaluate on the full CPU model (sim-to-sim), or film a scripted drive |
| `teleop.py` | Drive the policy with the keyboard, real time |
| `envs.py`, `train.py` | CPU fallback: gymnasium env + stable-baselines3 PPO. About 100× slower; for machines without an NVIDIA GPU |

## The task

- **Action:** 12 output targets (ab/ad, output B = hip, output A = crank) as offsets from the standing pose, at 50 Hz.
  Targets are clamped to the firmware's limits (ab/ad ±12°, outputs ±88°, knee ±52°).
- **Observation (actor):** body velocity, gyro, gravity direction, command, output angles and speeds, last action,
  and a trot clock (sin/cos, 0.3 s period). The critic also sees the configuration (friction, added mass,
  motor strength, gain scales), the body height and foot contacts.
- **Reward:** velocity and yaw tracking, following the trot clock (diagonal pairs, as `gait.py`), feet air time; penalties
  on body bounce, tilt and height, torque, action rate, foot slip and posture when standing. −10 for a fall, +10 for a successful episode.
- **Episode = one configuration:** two 4 s commands (12 % stand still, 30 % one axis only), floor friction, torso mass,
  motor strength, PD gain scales and a shove, drawn from a bank (`quad.BANKS`, `--bank=`):

  | | commands fwd / side / yaw | friction | mass | strength | gains | shove |
  |---|---|---|---|---|---|---|
  | `base` | −0.5…1.2, ±0.4 m/s, ±1.2 rad/s | 0.4–1.25 | −0.5…+2 kg | 0.8–1.05 | ±15–20 % | ≤ 0.8 m/s |
  | `hard` | −0.8…1.6, ±0.6 m/s, ±2.0 rad/s | 0.25–1.25 | −1…+4 kg | 0.65–1.05 | ±25–30 % | ≤ 1.6 m/s |

  **Success** = no fall and rms errors under 0.2 m/s and 0.35 rad/s.

### Success-Guided Sampling

From Zhang et al., *A Balanced Data Diet* ([arXiv 2610.12465](https://arxiv.org/abs/2610.12465)). There is a fixed
bank of 16,384 configurations, and each keeps its last 16 outcomes. Each reset draws the next configuration from
softmax(log w / T), with w = (p+ε)^(κt) (1−p+ε)^(κ(1−t)), t = 0.66, κ = 5, T = 2. Most weight goes to configurations
the policy solves about two times in three; mastered and hopeless ones get little, and none gets zero.
`--sampler=uniform` is the baseline.

## Results (cake, RTX 4090, 2026-10-23)

About **420k steps/s** with 8192 worlds: 300 M steps in 12 minutes (the CPU path does ~2–4k steps/s). Every policy
is checked on the full CPU model (rotors on their joints, every mesh, real contacts), which it never trained on:
`play.py eval <name> --n=300 --bank=…`, held-out episodes.

| Run | Trained on | Bank success (training) | Full CPU model, `base` bank | Full CPU model, `hard` bank |
|---|---|---|---|---|
| `walk_gpu` (SGS) | `base`, 300 M | 0.98 | 100 %, 0 % falls, 0.051 m/s | 69 %, 7.0 % falls, 0.148 m/s |
| `walk_uniform` | `base`, 300 M | 0.98 | 100 % (200 eps, old eval) | – |
| `hard_sgs` | `hard`, 400 M | 0.83 | 100 %, 0 % falls, 0.053 m/s | 86 %, 0.7 % falls, 0.117 m/s |
| `hard_uniform` | `hard`, 400 M | 0.83 | 100 %, 0 % falls, 0.053 m/s | 87 %, 0.0 % falls, 0.115 m/s |

- Training on `hard` costs nothing on `base` and takes the hard-episode success from 69 % to 86 %, falls from 7 % to ~0.
- SGS and uniform sampling come out the same on both banks (one seed each; curves within a few % throughout).
  On `base` nearly everything is learnable; on `hard` both plateau at 83 % with ~17 % of the bank out of reach
  (fast commands on weak motors and ice). SGS correctly stops sampling those, but there is no learnable-but-rare
  middle band for it to find. That needs a different kind of hardness: rough terrain, steps, slopes.

`../video/v4_rl_walk_gpu.mp4` is the scripted drive (`play.py film`).

## Run it

GPU training (Linux, NVIDIA):

```
cd CapstanPedal/fullbuild_v4/rl
conda env create -f environment.yml && conda activate capstan-gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_gpu.py --steps=300 --name=walk_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_gpu.py --steps=400 --bank=hard --name=hard_sgs
python train_gpu.py --steps=100 --name=walk_gpu --init=policies/walk_gpu.pkl     # carry on training
```

The first iteration compiles the Warp kernels (a few minutes). Options: `--envs`, `--cfgs`, `--bank=base|hard`, `--sampler=sgs|uniform`,
`--sgs_t`, `--sgs_k`, `--sgs_temp`, `--lr`, `--ent`, `--seed`.

Check and watch (the capstan-sim env from `../mujoco/SETUP.md` is enough: numpy, mujoco, glfw):

```
python play.py eval walk_gpu --n=300 [--bank=base|hard]   # success rate on the full CPU model
MUJOCO_GL=egl python play.py film walk_gpu        # ../video/v4_rl_walk_gpu.mp4 (drop MUJOCO_GL on a Mac)
python teleop.py walk_gpu                         # WASD / QE drive, Shift run, K shove, X reset
```

CPU fallback: `pip install torch stable-baselines3 gymnasium tensorboard`, then `python train.py 200`.
