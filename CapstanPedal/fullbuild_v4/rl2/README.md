# rl2: learned control for the v4 quadruped with the robot's real sensors

The second version of `../rl`. Policies here see what the built robot will see (`hw.py`): the pi3hat's IMU, the
moteus encoders, and a new optical-flow + range sensor pair for ground speed (`../BOM.md`, "Sensing"), with their
rates, noise, biases, dropouts and a 4–20 ms control delay. No policy input comes free from the simulator.

| File | |
|---|---|
| `hw.py` | The sensors and timing: rates, noise, bias, delay, battery sag. Replace the assumed numbers with measurements |
| `core.py` | The robot: models (GPU "physics" and CPU "full"), PD + motor model, firmware limits, `sense()` (one observation frame), terrain height |
| `tasks.py` | The tasks, their scenario banks, reward terms and success tests: `walk`, `terrain`, `rear`, `handstand`, `balance` |
| `env.py` | A task on thousands of GPU worlds (`GpuEnv`) or one CPU world in the full model (`CpuEnv`) |
| `train.py` | PPO + SGS / uniform sampling, per-term reward logging, held-out evaluation every 50 iterations; `--teacher=1` trains a teacher that also reads the privileged state |
| `train_sac.py` | FastSAC: off-policy soft actor-critic (replay buffer, twin distributional critics, auto-tuned entropy); same tasks and outputs as `train.py`. Terrain 88 % vs PPO's 71 % |
| `distill.py` | Teacher → blind student: the student reads 1 s of its own sensors and imitates the teacher (DAgger), no reward |
| `terms.py` | Reward-term report for a run: which term dominates, how it moved |
| `play.py` | Held-out evaluation and films on the full CPU model |
| `progress.py` | Training-progress film: one scenario played by snapshots from across a run, with their statistics |
| `audit.py` | Is it good enough for the robot? Quality under worst-case sensors, weak motors, slippery floor, dead flow sensor |
| `pick_best.py` | Choose the deployable snapshot of a run on a separate CPU selection set (training swings; the last iterate is often not the best) |
| `run_queue.sh` | Train a batch on cake: `sh run_queue.sh experiments/<batch>.list`; after each run: films, held-out eval, audit, best snapshot |
| `pick.sh` | Best snapshot of finished runs: `sh pick.sh <name> ...` |
| `experiments/` | Every batch we ran, one run per line, and what each tested |
| `check_robot.py` | Does a build's robot fit rl2? Run it first on a new build version |
| `showreel.py` | One film of all five skills in one yard, five policies, an orbiting camera that visits each |
| `teleop.py` | Drive a policy with the keyboard (needs a display) |
| `wandb_sync.py` | Send runs to Weights & Biases from their logs (`--follow` while they train); training itself needs no account |
| `sync.sh` | Copy the code to cake (code only; policies, logs and films stay there) |
| `codex-review.md` | The independent Codex review and what changed because of it |
| `sgs-lessons.md` | The project record: what we did, what worked and what didn't, deployable policies, lessons from SGS |

## The policy's input: 3 stacked frames of 51 numbers

| Group | n | Source on the robot |
|---|---|---|
| Rotation rate | 3 | pi3hat IMU gyro, with a per-episode bias |
| Gravity direction | 3 | pi3hat attitude filter, with a per-episode tilt error |
| Ground speed (fwd, side) | 2 | NEW: optical flow × range, 20 ms old, scale error, random dropouts (held, flagged) |
| Flow valid | 1 | NEW: the flow sensor's quality flag |
| Height | 1 | NEW: time-of-flight range to the ground |
| Command | 3 | joystick / operator (rear, handstand: command[0] = up) |
| Joint angles, speeds | 24 | moteus encoders: 14-bit quantised, calibration offsets, velocity noise |
| Last action | 12 | the policy's own previous output |
| Trot clock | 2 | sin / cos of a 0.3 s cycle (walking tasks; zeros otherwise) |

Output: 12 joint targets around the standing pose, clamped to the firmware's limits, applied after the control delay.

A distilled student (`distill.py`) reads the last 50 frames (1 s) instead of 3: the robot's sensors are the same,
it just remembers longer, to work out from how its legs landed what the ground is like.

## Tasks

| Task | Episode | Success | Bank (difficulty) |
|---|---|---|---|
| `walk` | 8 s, two commands, a shove | no fall, rms errors < 0.2 m/s, 0.35 rad/s | commands, friction, mass, motors, gains, battery |
| `terrain` | up to 8 s down a 1.2 m-wide course | reach 85 % of speed × 8 s down the course | 8 terrain types × 10 levels |
| `rear` | 6 s; "up" at 0.5–1.5 s | body ≥ 50° nose-up for 80 % of the last 2 s | shoves and weaker motors by level |
| `handstand` | 6 s | body ≥ 50° nose-down likewise | as rear |
| `balance` | 8 s on a board on a 0.2 m ball | board never touches the floor, no fall | ball rolling damping (high → none) by level, shoves |

## Run

Training (cake, conda env capstan-gpu):

```
sh run_queue.sh experiments/queue4.list                         # a batch, in order; logs/<name>.*
python train.py --task=terrain --sampler=sgs --seed=0 --steps=400
python train.py --task=walk --w=slip=-0.5,air=2.0 --name=walk_try   # change reward weights
python train.py --task=terrain --sampler=uniform --teacher=1 --name=terrain_teacher_s0   # a teacher that sees the course
python distill.py --teacher=terrain_teacher_s0                # its blind student: terrain_student_s0
python terms.py walk walk_try                                   # compare reward terms
python play.py eval terrain_uniform_s0 --n=400                  # full CPU model, held-out
MUJOCO_GL=egl python play.py film rear --level=5
```

Teleop (a machine with a display; copy `policies/<name>.npz` from cake, ~1 MB): `python teleop.py walk`.

## Deployable policies (in git, ~1 MB each)

`policies/` holds only these; every other run's policies, checkpoints (`.pkl`), snapshots and logs stay on cake.

| Policy | Task | Held-out success (full CPU model) |
|---|---|---|
| `walk2_best` | walk | 100 % |
| `rear_best` | rear | 100 % |
| `handstand2_best` | handstand | 82 % |
| `balance2_best` | balance | 39.5 % (100 % at level 0) |
| `terrain_uniform_s0_best` | terrain | 75.5 % |
| `terrain_student_s0_best` | terrain (blind student of a teacher that saw the course; reads 1 s of history) | 77.0 % |

## A new build version

rl2 reads the robot from the build folder it sits in (`core.BUILD`, default `..`): `../mujoco/model.py` builds the
MuJoCo model from `../urdf/capstan_quad.urdf`, `../mujoco/gait.py` gives the leg geometry and limits, films go to
`../video/<tag>_rl2_*.mp4` (`fullbuild_v5` -> `v5`). Nothing in rl2 names v4. For a new build:

1. Make the build as before (`src/` -> meshes -> `urdf/build_urdf.py`), so `urdf/` and `mujoco/` exist in
   `fullbuild_v5/`. Then copy the code: `rsync -a --exclude policies --exclude logs fullbuild_v4/rl2/ fullbuild_v5/rl2/`
   and copy the deployable policies too if you want to try them on the new robot.
   (Or, without copying: `CAPSTAN_BUILD=../../fullbuild_v5 python ...` from here. Policies and logs then still
   land in this folder, so prefer the copy.)
2. `JAX_PLATFORMS=cpu python check_robot.py --policy=walk2_best` in the new folder. It checks what rl2 needs from
   the build:
   - `model.py`: `build_spec()`, `PODS` (four pods, in the order LF, LR, RF, RR), `TAU`, `W0`, `A_PUSH`
   - `gait.py`: `Leg` (`ja`, `jv`, `act`, `ik`, `nominal`, `hip_z`), `FOOT_R`, `H_STAND`, `ABAD_LIM`, `OUT_LIM`,
     `KNEE_SOFT`
   - URDF names: body `torso` with a collision box (the sensors sit under it), geoms `foot_<pod>`, bodies
     `shin_<pod>`, rotor joints named `*_rotor*` (dropped for the GPU model), 12 actuators
   - that the robot stands on PD alone, and that an old policy still walks (if it falls: retrain).
3. Revisit the numbers tuned to v4 if the robot changed much: `core.py` `KP`, `KD`, `ACT_SCALE` (PD gains and action
   range), `HIND` / `FRONT`; `hw.py` (sensors: replace assumptions with measurements); the fall heights and pose
   targets in `tasks.py`. If the mass or leg length changed, the standing height and sensor distances follow from the
   model on their own.
4. `sh sync.sh` (set `REMOTE` to the new folder on cake), then `sh run_queue.sh experiments/<batch>.list` on cake,
   starting from `queue2.list` and `queue3.list` (the settings that produced the deployable policies).

## Results

See `sgs-lessons.md` for the summary of every skill and the sampler comparison; per-run curves in `logs/` on cake.
