# Capstan quadruped v4

[fullbuild_v3](../fullbuild_v3), made narrower, with full X-truss plates and a cleaner simulation model. The
leg pods are unchanged [v9.1](../v9_1_buildable) modules. Everything v3's README says about the cassette,
the build order, the bearings and the review history still applies; this file lists what changed.

## What changed from v3

| | v3 | v4 |
|---|---|---|
| Width (outside of everything, at rest) | 430 mm | **400 mm** |
| Pods' ab/ad axes | ±100 mm | ±86 mm |
| ab/ad range | ±20° | **±12°** (software limit) |
| Bulkheads / top plate / bottom plate half-widths | 140 / 130 / 110 mm | 126 / 116 / 96 mm |
| Top and bottom plates | X-truss with half its windows missing (two of four per bay) | **all four windows of each X**: a double X on top; on the bottom, clear of the drum notches |
| End frames (front and back plates) | flat, one piece | unchanged: flat, one piece |
| Mass (CAD) | 19.2 kg | 18.9 kg |

**Why ±12°.** The left and right pods face each other across the middle. Their inboard faces (motor plate,
moteus boards, post screws) are what touch when both tilt inward. At ±20° that limits the ab/ad axes to
±94.9 mm, at ±15° to ±87.5, at ±12° to ±82.7. v4 puts them at ±86, with about 6 mm to spare at ±12°.
Narrowing further would mean a thinner leg module. The gaits use only a few degrees of ab/ad.

**Checked, not changed:**
- **Front gantry:** the 6903 (17×30×7) sits in a Ø30.1 pocket at y 146–153 (0.05 mm press), behind a Ø18
  lip. The crank's Ø17 stub runs through it to y 155, and its Ø21 shoulder meets the inner race.
- **Knee stack** (along the pin, outboard to inboard): pin head and spacer, thigh 111–119, a 0.5 mm washer,
  then the shin with its 6000ZZ (119.5–127.5), the shin's front plate to 131.5, 5 mm clear, then the coupler.
  The gap seen in the v3 simulation was the display meshes, below.

## Checks (v4 geometry)

- **Assembly** (`src/check_assembly_quad.py`): all 180 body items reachable along their tool paths when the
  build fits them.
- **Clash** (`src/check_quad.py`): pods at ab/ad 0 and ±12° against every body part and each other: clear
  (also the body hardware: PASS).
- **Stress** (`src/stress_test.py`, linear FEA, PETG): worst printed safety factors 5.9 (centre bulkhead),
  6.3 (inner), 11.0 (end frame) under the doubled impact case, as v3.

The narrowing exposed one CAD fault, which is fixed. Each bulkhead merges the left and right pods' cradle
plates, and these reach 34 mm past the centreline. With the pods 28 mm closer, one pod's plate now covered
the other pod's ab/ad drum: about 1,000 mm³ of overlap at ab/ad 0. Each plate is now clipped at the
centreline before merging.

## Simulation (`mujoco/`, `urdf/`)

The v3 URDF's display meshes were simplified by vertex clustering. That erased the small drums' rope
grooves, distorted the ab/ad drums, opened gaps, and left streaks round screw holes on flat faces. v4's
`urdf/build_urdf.py`:
- **Simplifies only hardware** (screws, inserts, nuts, bearings) to 30 %, by quadric edge collapse. Printed
  parts and every drum stay at full CAD detail.
- **Writes OBJ meshes with explicit normals**, creased at 30°: flat faces shade flat and cylinders smooth.

The URDF itself is as v3's, rebuilt for v4:
- motor rotors and small drums on `<mimic>` joints at −9× their output;
- the crank driven by output A;
- the 4-bar closed in MuJoCo.

Controller (`mujoco/gait.py`):
- **Same as v3, re-tuned for v4.**
- **Gaits:** trot 3.0° peak tilt, walk 3.1°, pace 5.0°, bound 6.5°, pronk 5.5°, all tracking their commands.
- **IK:** the limit-aware solver, now with ab/ad clamped to ±12°.
- **Jump:** re-tuned on this model (`mujoco/jump_params.json`). On 30 settings it was not tuned on (mass
  ±10 %, friction 0.6–1.25) it lands and stands **30 of 30**: 0.55–0.81 m (mean 0.70), peak foot force
  580 N, no shin or body contact.

Setup (conda env): [mujoco/SETUP.md](mujoco/SETUP.md). Running it: [mujoco/RUN.md](mujoco/RUN.md).

```
cd fullbuild_v4/mujoco
python teleop.py          # W/S A/D Q/E R/F, 1-5 gaits, Space jump, G shadows
python make_videos.py     # ../video/v4_*.mp4  (--notext: without captions, v4_*_notext.mp4)
```

Films in `video/` (simulation, model-based control, no learning):
- `v4_trot.mp4`, `v4_turn.mp4`, `v4_spin.mp4`, `v4_sidestep.mp4`
- `v4_height.mp4` (R/F)
- `v4_gaits.mp4` (walk, pace, bound, pronk)
- `v4_jump.mp4` (half speed)
- `v4_orbit.mp4`

## In the browser (`web/`)

The same model and controller run on the portfolio site as a keyboard-driven simulator (MuJoCo 3.16 as
WebAssembly, three.js for drawing). `web/export_web.py` writes what the page loads:
- `quad.mjb`: the compiled physics model, visual meshes removed. They never collide and have no mass, so the
  script checks the dynamics are identical step for step before writing it. Binary, because the XML writer
  rounds to 6 digits and the walk drifts apart after a few seconds.
- `visual.bin`, `quad.json`: the CAD meshes per body and colour, decimated to ~360k triangles, plus the pod
  frames and jump parameters the controller reads.

```
cd fullbuild_v4/mujoco
python ../web/export_web.py [OUT]   # default OUT: portfolio-website/public/sims/capstan-quad
```

The site's TypeScript controller is a line-for-line port of `gait.py`. Re-export and re-port it if the model or
the controller changes: on an 8 s walk, turn, jump and gait change, the two agree to 2e-10 m.

## Learned control (`rl/`, `rl2/`)

- `rl/`: the first RL walking controller (PPO on GPU worlds, idealised sensors), and the first SGS experiments.
- `rl2/`: policies that see only what the built robot will measure (IMU, encoders, and a new optical-flow +
  range sensor pair, see `BOM.md` "Sensing"), with noise, dropouts and delay. Five skills: walk, rear up,
  handstand, balance on a board on a ball, a blind terrain course. Deployable policies in `rl2/policies/`. What we
  did and learned: `rl2/sgs-lessons.md`. A new build reuses rl2 unchanged if its URDF keeps the same names:
  `rl2/README.md`, "A new build version". Training runs on a GPU box (cake), not here.

## Regenerating

```
cd fullbuild_v4/src
python generate_quad.py <mesh_dir> <leg_mesh_dir>   # printed/, laser_cut/, legs/, film meshes
python check_quad.py && python check_assembly_quad.py <mesh_dir> && python stress_test.py
# copy <mesh_dir> to ../mujoco/meshes (quad.json paths relative), then:
cd ../urdf && python build_urdf.py
```

Not built or tested on hardware.
