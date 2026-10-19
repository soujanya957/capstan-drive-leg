# The quadruped in MuJoCo

| Script | What |
|---|---|
| `teleop.py` | **Drive it.** The URDF model (every CAD part on its real link) under a model-based trot controller, real time, ~60 fps |
| `watch_assembly.py` | The full CAD assembly playing the build film's timeline (every part and fastener in order), slowed down, free camera |

```
pip install -r requirements.txt
cd fullbuild_v3/mujoco
python teleop.py
python watch_assembly.py --speed=0.25
```

## teleop.py

| Key | |
|---|---|
| W / S | forward (0.6 m/s; Shift 1.0) / back (0.3 m/s) |
| A / D | side-step left / right (0.3 m/s) |
| Q / E | turn left / right (0.8 rad/s; Shift 1.3) |
| R / F | body up / down (hold): hip 26 to 38.5 cm above the feet |
| 1 – 5 | gait: trot, walk, pace, bound, pronk (changes once all four feet are down, then carries on) |
| Space | jump |
| [ / ] | slower / faster time (0.05× to 2×); P pause; X reset |
| C | camera follow; G shadows (about half the frame rate); V collision shapes |
| mouse | left drag orbit, right drag pan, scroll zoom; Esc quits |

**The model** (`model.py`) is built from `../urdf/capstan_quad.urdf` (33 links). Each leg has:
- **Driven links:** the pod (ab/ad, motor C), the thigh (output B, motor B) and the crank (output A, motor A,
  coaxial with the hip).
- **The 4-bar:** the coupler on the crank pin and the shin on the knee. The URDF can't close the loop, so
  the MuJoCo model pins the coupler's far end to the shin's lever pin (it holds to 0.003 mm).
- **Rotors:** every motor's rotor and small drum turns at −9× its output (URDF `<mimic>`, MuJoCo joint
  equalities). That includes motor C's rotor on the body.

Motors: 31.5 N·m, falling off from half the no-load speed, output A limited to 25 N·m when it pushes the
coupler (the same motor model as `rl/`). Contacts are primitives (body box, pod boxes, link capsules, 23 mm
ball feet). Mass 19.2 kg from the CAD.

**Inverse kinematics** (`gait.py`, `Leg.ik`):
- **Method:** analytic (ab/ad, then the 2-link leg in its plane, on the rest pose's knee branch).
- **Limits:** where a target breaks a joint limit (ab/ad ±19°, both outputs ±88°, the knee's soft ±52°) or is
  out of reach, damped least squares projected onto the limits returns the nearest reachable foot position.
- **Checked:** exact round trips over about 14,000 poses; no limit broken for 8,000 out-of-reach targets;
  FK matches MuJoCo to 0.01 mm.
- **Used everywhere:** walking, swing, crouch, push and landing.

**The controller** (`gait.py`) uses no learning:
- **Feet:** targets are set in a level heading frame, so the body stays level. Stance feet hold their spot
  on the ground; swing feet land half a stance ahead (Raibert).
- **Legs:** PD on the drive outputs, and each stance leg carries its share of the weight plus a roll/pitch
  correction.
- **Leaping gaits** (bound, pronk): each stance pushes a half-sine impulse that carries the weight over the
  time the feet are down.

Five gaits, each tuned by a sweep in simulation. Peak body tilt and velocity error (rms) over 0.4 / 0.8 /
−0.25 m/s, 0.25 m/s sideways and 0.6 rad/s turning:

| Gait | Footfalls | Cycle | Tilt | Error |
|---|---|---|---|---|
| trot | diagonal pairs | 0.22 s | 3.1° | 0.07 m/s |
| walk | one foot at a time (hind-left, fore-left, hind-right, fore-right) | 0.45 s | 3.3° | 0.12 |
| pace | same-side pairs | 0.22 s | 4.2° | 0.16 (turns at about half the command) |
| bound | fore pair, then hind pair | 0.24 s | 6.4° | 0.11 (turns at about 70 %) |
| pronk | all four together, with flight | 0.28 s | 7.7° | 0.07 |

**The jump** is the scripted jump: crouch, force-controlled push at 65° with a fixed front/rear split and
pitch feedback, swing to a landing pose fixed in the world, soft landing. Its parameters
(`jump_params.json`) were tuned by random search on this model with mass ±10 % and friction 0.6–1.25. On 30
settings it was not tuned on:
- 28 of 30 land and stand; 1 falls and 1 doesn't settle in time.
- It travels 0.51–0.86 m (mean 0.67).
- A shin brushes the floor on 1 landing.
- Peak foot force is 630 N, about 1.5× the RL model's landings. Treat it as a demo, not a load case the legs
  are checked for.

## watch_assembly.py

| Key | |
|---|---|
| Space | pause / play; [ / ] slower / faster |
| N / B | next / previous build step; Left / Right ±5 s (Shift ±1 s); `,` / `.` one frame; Home restart |
| C | the film's camera / free camera (dragging the mouse frees it) |

It replays `src/make_quad_film.py`'s timeline: the same steps and captions as `video/quadruped.mp4`. It poses
1,150 separate parts every frame, so it is much heavier than `teleop.py`.

## Regenerating

After a CAD change:
1. `python ../src/generate_quad.py <mesh_dir> <leg_mesh_dir>`
2. Copy the meshes to `meshes/` (`quad.json` paths are relative).
3. `python ../urdf/build_urdf.py`, which rewrites the URDF, its meshes and `frames.json`.

The URDF's visual meshes are decimated by vertex clustering (1 mm on hardware, 0.6 mm elsewhere: 0.69 M
triangles from 1.4 M). Inertias come from the full CAD.

Files:
- `viewer.py`: the window (keys free, mouse camera, overlay text).
- `model.py`, `gait.py`, `teleop.py`: the simulation.
- `cad.py`, `watch_assembly.py`: the assembly viewer.
