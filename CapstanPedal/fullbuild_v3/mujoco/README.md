# Opening the quadruped in MuJoCo

Two windows, both on this machine's display (GLFW; plain `python` works on macOS, no `mjpython` needed):

| Script | What you see | Needs |
|---|---|---|
| `teleop_dog.py` | the robot in physics (MuJoCo, `rl/quad_model.py`), driven from the keyboard | mujoco, glfw, stable-baselines3 |
| `teleop_dog.py --cad` | the same, with the full CAD robot drawn on the physics | + build123d (the CAD code) |
| `watch_assembly.py` | the full CAD assembly playing the build film's timeline (every part and fastener in order, then stand, trot, jump), slowed down, free camera | + build123d |

```
pip install -r requirements.txt          # versions these were run with
cd fullbuild_v3/mujoco
python teleop_dog.py                     # or: python teleop_dog.py --cad
python watch_assembly.py --speed=0.25    # optional --at=<s> to start partway
```

## teleop_dog.py

| Key | |
|---|---|
| W / S | forward / back (up to 0.9 m/s forward, 0.2 m/s back) |
| A / D | side-step left / right (0.3 m/s) |
| Q / E | turn left / right (0.8 rad/s) |
| R / F | body up / down (hold; −5 to +1 cm) |
| Space | jump (the scripted jump, below), then back to walking |
| [ / ] | slower / faster time (0.05× to 2×); P pause; X reset; C camera follow |
| mouse | left drag orbit, right drag pan, scroll zoom; Esc quits |

Walking and turning are the trained walk policy (`rl/policies/walk`); the commands ramp at 1.5 per second
and are clipped to the range it was trained on. Height is an offset added to the policy's leg targets: the
policy was never trained for it, and in `dog.py`'s self-test it copes from −5 to +1 cm (at +2 cm it fell).
The robot is reset automatically if it falls.

## watch_assembly.py

| Key | |
|---|---|
| Space | pause / play |
| [ / ] | slower / faster (0.05× to 2×; starts at `--speed`, default 0.5) |
| N / B | next / previous build step |
| Left / Right | back / forward 5 s (Shift: 1 s); `,` / `.` one frame |
| Home | restart |
| C | the film's camera / free camera (dragging the mouse frees it) |

It replays `src/make_quad_film.py`'s timeline exactly: the same steps, captions and motion as
`video/quadruped.mp4`, but you choose the speed and the view. The walk and jump at its end are the film's
kinematic animation, not physics; for physics use `teleop_dog.py --cad`.

## Meshes

`meshes/` is a copy of the film meshes `generate_quad.py` writes (47 MB; `quad.json` lists them with paths
relative to it). After changing the CAD, regenerate them and point the scripts at the new folder:
`python ../src/generate_quad.py <mesh_dir> <leg_mesh_dir>`, then `--meshes=<mesh_dir>`.

Files: `viewer.py` (the window: mouse camera, keys, text), `dog.py` (physics + walk policy + height + jump;
`python dog.py` runs its self-test without a window), `cad.py` (the CAD scene, posed from physics or from
the film).
