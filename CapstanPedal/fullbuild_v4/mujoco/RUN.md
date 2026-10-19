# Running the simulator

Set up first: [SETUP.md](SETUP.md). Every command below runs from `fullbuild_v4/mujoco` with the env active:

```
cd CapstanPedal/fullbuild_v4/mujoco
conda activate capstan-sim
```

| Command | What it does |
|---|---|
| `python teleop.py` | Drive the robot with the keyboard, real time |
| `python make_videos.py [clip ...]` | Render the short films to `../video/v4_<clip>.mp4` (no window needed) |
| `python watch_assembly.py` | Watch the full CAD assembly being built, slowed down |

## Drive it: `teleop.py`

```
python teleop.py
```

The robot starts standing. Hold keys to move: it starts stepping in the selected gait (trot by default) and
stands still again once you let go and it has slowed to a stop.

| Key | |
|---|---|
| W / S | forward (0.6 m/s) / back |
| A / D | side-step left / right (0.3 m/s) |
| Q / E | turn left / right (0.8 rad/s) |
| Shift | run: forward and turning ×1.6 |
| R / F | body up / down (hold): hip 26 to 38.5 cm above the feet |
| 1 – 5 | gait: trot, walk, pace, bound, pronk (switches once all four feet are down) |
| Space | jump (crouch, push, land); best from standing still |
| [ / ] | slower / faster time (0.05× to 2×) |
| P | pause |
| X | reset the robot |
| C | camera follows the robot on / off |
| G | shadows on / off (off is about twice as fast) |
| V | show the collision shapes |
| Mouse | left drag orbit, right drag pan, scroll zoom |
| Esc | quit |

The overlay (top right) shows the time speed and frame rate, the commanded and actual speed, the hip height,
roll and pitch, and the gait or jump phase.

Tips:
- Bound and pronk are lively; give them a moment at zero speed after switching before pushing W.
- Use `[` to slow time down and watch the feet, the 4-bar and the motors' rotors (they spin at 9× the joints).
- If it falls over it stands itself back up; X resets it to the start.

## Make the videos: `make_videos.py`

```
python make_videos.py            # all eight clips, a few minutes
python make_videos.py trot jump  # just these
```

Clips: `trot`, `turn`, `spin`, `sidestep`, `height`, `gaits`, `jump` (half speed), `orbit`. Each is
1280×720 at 30 fps with a caption, written to `../video/v4_<clip>.mp4`. It renders offscreen, so it works
over SSH; on a Linux machine without a display, prefix it with `MUJOCO_GL=egl`. It prints whether the robot
finished each clip upright, and exits non-zero if any fell.

To change a clip (its speeds, timing, camera), edit the `CLIPS` table at the top of `make_videos.py`.

## Watch the build: `watch_assembly.py`

```
python watch_assembly.py                 # half speed
python watch_assembly.py --speed=0.25    # slower
python watch_assembly.py --at=120        # start 120 s in
```

| Key | |
|---|---|
| Space | pause / play |
| [ / ] | slower / faster |
| N / B | next / previous build step |
| Left / Right | ±5 s (Shift ±1 s); `,` / `.` one frame |
| Home | restart |
| C | the film's camera / free camera (dragging the mouse frees it) |

It poses about 1,150 separate parts every frame, so it is much heavier than the teleop. It takes a while to
load the CAD meshes the first time.

## Using the model in your own code

```python
import mujoco, numpy as np
import model as M, gait as G

m = M.build()                    # MuJoCo model from ../urdf/capstan_quad.urdf
d = mujoco.MjData(m)
c = G.Controller(m, d)
c.reset_pose()
c.request_gait("trot")           # trot, walk, pace, bound, pronk
c.cmd = np.array([0.5, 0.0, 0.0])  # forward m/s, sideways m/s, yaw rad/s
for _ in range(int(5 / m.opt.timestep)):
    c.step(m.opt.timestep)       # sets d.ctrl from the gait and IK
    mujoco.mj_step(m, d)
```

`c.h` sets the body height (0.26–0.385 m) and `c.jump()` starts a jump. `G.Leg` has the leg's forward
kinematics and IK.

## After a CAD change

Rebuild the URDF, then everything above uses it:

```
cd ../src && python generate_quad.py <mesh_dir> <leg_mesh_dir>
# copy <mesh_dir> to ../mujoco/meshes
cd ../urdf && python build_urdf.py      # capstan_quad.urdf, its meshes, frames.json
```
