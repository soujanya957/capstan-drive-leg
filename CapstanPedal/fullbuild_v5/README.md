# Capstan quadruped v5

[fullbuild_v4](../fullbuild_v4) redesigned for **range** and **vision**:
- a 2.8× battery, slung low in a belly pod that also stiffens the body's middle and is its skid plate;
- a head on the face end with a Jetson and a tilting stereo camera.

The four [v9.1](../v9_1_buildable) leg pods, the bulkheads, the cassette and the controller are v4's.

## What changed from v4

| | v4 | v5 |
|---|---|---|
| Battery | 6S 1800 mAh (40 Wh) in the cassette's top tray, under the lid | **6S 5000 mAh (111 Wh) in a belly pod** under the bottom plate; swapped from below with four thumb screws |
| Range, trotting at 0.6 m/s (model, `mujoco/energy.py`) | 206 W: **9 min, 0.3 km** | 228 W with the Jetson: **23 min, 0.8 km** |
| Body's middle span (between the inner bulkheads) | 3 mm plate on four bulkheads | plate + the saddle's 4 mm flange and 3 mm walls, a 54 mm deep box, clamped by the bulkheads' bottom screws |
| What it lands and lies on | the bottom plate's edges and screw heads | the pod's 5 mm cover, edges rounded (a skid): motors off, it rests on it at 0.20 m |
| Face (-X: it walks toward -X) | a flat end frame, as on Stanley | **a head**: a printed box with a Jetson Orin Nano and a stereo camera on a cradle, 0 to 90° down |
| Vision | none | ZED Mini, RealSense D435/D435i or D455 (one cradle drilled for all three) |
| Compute | Pi 5 + pi3hat (control) | the same, plus the Jetson for vision, on Ethernet |
| Length × width × height | 690 × 400 mm | 780 × 400 mm; 59 mm deeper below the plate (ground clearance standing: 281 → 223 mm) |
| Mass (model) | 18.7 kg | 19.6 kg (+0.47 head, +0.47 battery, +0.18 pod) |
| Jump on 30 untuned settings (`mujoco/jump_test.py`) | 29 of 30, torso rise 0.16 m | **30 of 30**, 0.14 m (re-tuned for v5's centre of mass, `mujoco/jump_tune.py`) |
| Stress cases | three bulkheads | the same, plus a face-first fall on the head and a belly landing on the pod |

Range is what v4 lacked most: 9 minutes of trotting. The pack goes low, not in the tray, for three reasons:
- **Size.** A 5000 mAh pack is 165 mm long; the tray takes 108.
- **Low mass.** It puts 0.75 kg 63 mm below the plate. That offsets the head's 0.47 kg up front: v5's centre of mass is
  within 2 mm of v4's in height.
- **The box.** The pod's saddle bolts through the plate into the four middle bulkheads, so it adds depth (a box section)
  where the body carries the pods' loads into each other.

The cost of all this:
- 0.9 kg more mass;
- 59 mm less ground clearance;
- a 13 % lower jump.

## Belly pod (`generate_quad.py`: `belly_saddle`, `belly_cover`)

- **Belly_Saddle** (printed): a 4 mm flange under the bottom plate with a 3 mm walled cavity for the pack and a 17 mm
  lane beside it for the leads. The four bottom-plate screws at the middle of the centre and inner bulkheads are
  M3 × 12 here (M3 × 8 in v4). They go through the flange and the plate into the same inserts, so the saddle, the
  plate and the bulkheads are one clamp.
- **Belly_Cover** (printed): 5 mm, edges rounded, vents under the pack's ends. Four M3 thumb screws into inserts in
  lugs at the saddle's ends.
- **Battery swap:** cover off, unplug the XT90 in the lane, the pack drops out. The power board's pigtail comes down
  through a slot in the plate (inside the dock's wiring window) and in the saddle's flange.
- The cassette's top tray is now free: a second pack or a payload. The build stand is two rails, the pod between
  them.

## The head (`src/head.py`)

- **Head_Mount**, one printed part. Its back plate meets the end frame: 4× M3 × 10 into inserts in the frame's new
  pads, driven from inside the head. Two side walls rise into the cradle's cheeks, and a top web ties them. It is open
  at the bottom (for the Jetson's ports, the leads and the air) and at the front.
- **Jetson Orin Nano** dev kit without its plastic base: the carrier board on four posts through its 86 × 58 mm
  corner holes, fan forward, ports down. The leads run up behind the board and through the back plate's window into
  the body. A 12 V converter (Pololu D36V50F12, from the 6S bus) is tied to the left wall.
- **Head_Face**: a vented plate 5 mm in front of the fan, its screws countersunk flush.
- **Camera_Cradle**: a shelf and a back plate, pinned between the cheeks 6 mm in front of the face. A second M3 per
  side goes through one of seven detent holes and locks the tilt: 0° (forward) to 90° (straight down) in 15° steps.
  With the pivot in front of the face, the camera clears the head at every tilt. At 90° it looks at the ground ahead
  of the fore feet. The CAD and the simulation use 30°.
- **Three cameras, one cradle**, drilled from the makers' drawings and CAD:

  | Camera | Rear threads | Spacing | Height above its base | Depth / screws |
  |---|---|---|---|---|
  | ZED Mini (Stereolabs `ZEDM.step`) | 2× M2 | 56 mm | 6.5 mm | 2 mm: M2 × 4, heads in 1.7 mm counterbores |
  | RealSense D435 / D435i (D400 datasheet, fig. 10-9) | 2× M3 | 45 mm | 12.5 mm | 3 mm: M3 × 6 |
  | RealSense D455 (fig. 10-11) | 2× M4 | 95 mm | 13 mm | 4 mm: M4 × 8 + washer |

  A 1/4-20 slot in the shelf takes anything with a tripod thread. The camera's ends stay open for a right-angle
  USB-C lead, which runs down the outside of a wall to the Jetson.

The ZED SDK needs an NVIDIA GPU, hence the Jetson; a RealSense also runs on the Pi 5. Parts and hardware:
[BOM.md](BOM.md).

## Checks (v5 geometry)

- **Head** (`src/check_head.py`): **pass**. It checks the head's parts and hardware against each other, the end
  frame, the plates and the face end's pods (ab/ad 0, ±12°). The cradle, camera and right-angle USB-C plugs are
  checked at all seven detents. The first run found four faults, all now fixed:
  - the cradle's lock bosses hit the top web at 60–75°;
  - the shelf hit the face screws' heads at 90°;
  - the end frame's top boss hit the back plate;
  - the Jetson's port envelope overlapped a screw.
- **Assembly** (`src/check_assembly_quad.py`): **pass**. Every screw, insert and nut, the belly's and the head's
  included, can be reached along its tool path when the film fits it.
- **Clash** (`src/check_quad.py`): **pass**. The pods clear the body, the belly pod and the head at ab/ad 0 and ±12°. The first run's
  only three hits were designed overlaps: the ZED's screws in its threads, and the camera model's body overlapping its
  own lens model. `check_head.py` already allowed both, and now `check_quad.py` does too.
- **Stress** (`src/stress_test.py`, `stress_test/results.md`): **pass**. PETG safety factors under twice a landing's load (1170 N): bulkheads 5.8 or more, as in v4; the head
  mount 3.0 in a face-first fall onto the cradle (14.9 MPa); the belly saddle 6.0 landing on the pod (7.5 MPa, 0.13 mm).
- **Review**: a Codex agent read the head's CAD and the build order. Its findings changed three things:
  - the ZED's screws: an M2 × 6 would have bottomed in its 2 mm threads;
  - the cradle's print orientation;
  - strain relief for the leads.

  It rated the tilt lock adequate: about 2.5 N per lock under a 5 g shock with a D455.

## Videos (`video/`, on cake)

- `quadruped.mp4`: the whole build, every screw, insert and nut. The belly pod and the head come last, then it
  stands up, trots and jumps (CAD, kinematics only).
- `v5_trot`, `v5_turn`, `v5_spin`, `v5_sidestep`, `v5_height`, `v5_gaits`, `v5_jump`, `v5_orbit`: the URDF model in
  MuJoCo under v4's model-based controller (`mujoco/gait.py`). Every clip ends upright.
- `v5_camera`: trotting and turning past some blocks. Insets show the ZED's left-camera colour image and depth
  (0.2–3 m) from the model's `zed` camera.

## Simulation

As v4 (`mujoco/`, `urdf/`), with three differences:
- The head and the belly pod are fixed to the torso, each with its own collision box.
- The feet are booked as TPU (1.2 g/cm³). v4 booked them at aluminium's 2.7, 60 g too heavy each.
- `mujoco/model.py` adds a camera `zed` at the ZED's left lens: 60° vertical field of view, at the cradle's tilt
  (`urdf/frames.json`, `head`). Render it with `mujoco.Renderer.update_scene(data, camera="zed")`.

The range figures come from `mujoco/energy.py` and rest on stated assumptions:
- the motors' power is the positive mechanical power over a 0.85 efficiency, plus copper loss at 0.12 Ω a phase;
- the electronics draw 19 W, plus 15 W for the Jetson;
- the packs are used to 80 %.

They compare v4 and v5; they are not a measured runtime.

## Regenerating

Generated on cake, where the meshes, printed STL/STEP and videos stay:

```
cd fullbuild_v5 && sh gen_v5.sh      # body, belly and head meshes; v4's pod meshes reused (mujoco/meshes)
sh post_v5.sh                        # checks, stress, URDF, gait videos, the build film (env: capstan-cad)
cd mujoco && python energy.py && python jump_test.py
```

Not built or tested on hardware.
