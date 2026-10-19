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
| Space | jump |
| [ / ] | slower / faster time (0.05× to 2×); P pause; X reset |
| C | camera follow; G shadows (about half the frame rate); V collision shapes |
| mouse | left drag orbit, right drag pan, scroll zoom; Esc quits |

**The model** (`model.py`) is built from `../urdf/capstan_quad.urdf`. Its links are the parts that move
together: torso, and per leg the pod (ab/ad), thigh (output B), crank (output A), coupler and shin. The 4-bar's
crank and coupler follow the knee through joint equalities (the URDF's `<mimic>`). Motor A drives output A =
hip + knee. Motors: 31.5 N·m, falling off from half the no-load speed, output A limited to 25 N·m when it
pushes the coupler (the same motor model as `rl/`). Contacts are primitives (body box, pod boxes, link
capsules, 23 mm ball feet). Mass 19.2 kg from the CAD.

**The controller** (`gait.py`) uses no learning. It trots with a 0.25 s cycle, each foot down 60 % of it.
- **Feet:** targets are set in a level heading frame, so the body stays level. Stance feet hold their spot
  on the ground; swing feet land half a stance ahead (Raibert).
- **Legs:** full 3-D inverse kinematics (ab/ad, then the 2-link leg), PD on the drive outputs, and each
  stance leg carries its share of the weight plus a roll/pitch correction.

It stands still with no command. In simulation it tracks 0.5, 1.0 and −0.3 m/s, 0.3 m/s sideways and
0.8 rad/s turning to within a few hundredths, with under 3° of body tilt (under 10° at the height extremes).

**The jump** is the scripted jump (crouch, force-controlled push at 66°, swing to a landing pose fixed in
the world, soft landing). Its parameters (`jump_params.json`) were tuned by random search on this model with
mass ±10 % and friction 0.6–1.25. On 30 settings it was not tuned on:
- 28 of 30 land and stand; 2 fall.
- It travels about 0.6 m.
- A shin brushes the floor on 11 of the landings.
- Peak foot force is about 1,000 N, more than twice the RL model's landings. Treat it as a demo, not a
  load case the legs are checked for.

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
