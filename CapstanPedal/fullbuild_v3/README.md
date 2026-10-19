# Capstan quadruped v3 (full build on v9.1 legs)

Four [v9.1](../v9_1_buildable) leg pods (9:1 capstans, **31.5 N·m per output, ±90°**, 200 mm thigh and shin)
in a body that **just screws together**: printed bulkheads with heat-set inserts, two flat laser-cut plates,
and an **electronics cassette** that drops in from the top. No bumpers, no nose or tail pieces.
Earlier versions (v0 to v9, fullbuild_v2) were removed from this repository; the READMEs summarise what each changed.

- `video/quadruped.mp4`: the build, step by step. The cassette is built on the bench, then the body round
  the four ab/ad drives (dock, centre bulkheads with motors C, ab/ad drums, **the ab/ad ropes**, inner
  bulkheads). Each pod's rear end bracket goes onto its hub first, then the pod slides in along the body
  into the bracket's ears. Then the end frames, the top plate, the e-stop, the cassette, the battery and
  the lid. After that it stands up, trots and jumps (scripted).
  `video/quadruped_rl_walk.mp4`, `video/quadruped_rl_jump_long.mp4`, `video/quadruped_rl_jump.mp4`: the
  trained policies driving the full CAD model from the physics simulation (below: the long jump does not
  land yet). `video/quadruped_jump_scripted.mp4`: the scripted (non-RL) jump, which lands. Every film also
  comes as `*_notext.mp4`.
- `printed/stl`, `printed/step`: every printed body part. `laser_cut/dxf`, `laser_cut/step`: the two plates.
  `legs/stl`: the 200 mm 4-bar links. The pods themselves: print from `../v9_1_buildable/print`; pods on +Y
  as is, pods on −Y **mirrored** in the slicer. Not printed for the quadruped: the cradle plates
  (`Hip_Outer_Plate`, `Hip_Inner_Plate`, `Hip_Front_Plate`, `Hip_Spine`) and the rail-stand parts.
- `rl/`: the MuJoCo model, the training environments, the trained policies and the scripted jump.
- `mujoco/`: open it in MuJoCo yourself: `teleop_dog.py` (drive it with W A S D, Q E, R F, Space) and
  `watch_assembly.py` (the whole build in a window, slowed down, free camera). See `mujoco/README.md`.

## Layout

| | |
|---|---|
| Body | 689 × 260 mm top plate, 689 × 220 bottom (bulkheads 280 wide); 221 mm plate to plate, 225 mm with the lid |
| Mass | ~17.4 kg from the CAD (`rl/masses.json`: torso 6.8, each pod 2.3, thigh 0.22, shin 0.16) |
| Pods' ab/ad axes | ±100 mm from the middle; ab/ad ±20° (the pods and the bottom plate clear to there) |
| Hips | ±221.5 mm fore and aft; standing (hip to foot 346 mm) the hips are 369 mm up |
| Heading | walks toward −X, knees trailing. Files: `F` = +X end, `R` = −X end, `L` = +Y |

## The body: screws and inserts only

- **Bulkheads, printed** (PETG): end frames (the trunnions' 6806s), inner bulkheads (the hubs' inner 6808s,
  motor C's tip bearing), centre bulkheads (the hubs' outer 6808s, motors C; moteus C on the bay side). The
  inner and centre ones print in left and right halves.
- **Plates, laser-cut** 3 mm (6061 recommended; carbon works): M3 × 8 countersunk screws into heat-set
  inserts in every bulkhead's top and bottom edge bosses. No T-nuts, no captured nuts anywhere in the body.
- **Drum notches** under each pod's drums: their rims sweep a full circle at ±90°.
- Every screw and insert the body build fits is checked reachable at that moment in the build order
  (`src/check_assembly_quad.py`, below).

## Electronics: the cassette

A printed open rack in the centre bay (60 × 116 mm, between the moteus C boards), from the bottom up:

| Level | What | How it is held |
|---|---|---|
| Floor | mjbots power_dist: XT30s out to the four legs low down | 4× M2.5 into inserts in 4.5 mm posts, set and driven from above through access holes in the shelf (Ø7) and the tray (Ø6), before the Pi goes in |
| Middle shelf | Raspberry Pi 5 (low heatsink) + mjbots pi3hat (a CAN-FD bus per leg) | 4× M2.5 standoffs into inserts; hat on top |
| Open middle | ~100 mm: the legs' harness service loops; the air path | – |
| Top tray | the 6S pack (≤ 46 × 108 × 40 mm, e.g. 1800 mAh high-C), right under the lid | hook-and-loop strap through the tray's slots; end stops |

- **Battery swap, no tools:** 4 thumb screws, lid off, unplug, unstrap, lift.
- **The whole rack** lifts out by the tray's end stops (finger slots) once the leg and e-stop plugs are off.
  It sits in a printed dock screwed to the bottom plate; the lid's two ribs hold it down.
- **E-stop:** a 22 mm latching mushroom switch, fixed in the top plate beside the lid (not on the lid, so the
  lid comes off untethered), wired to the power board's switch input through a plug at the cassette.
- **Air:** in through the dock's and the floor's windows, up through the open rack and the shelf's window,
  out through the lid's six vent slots.
- **Leg harness:** each pod's moteus boards ride on the pod; their CAN and power go through the hollow ab/ad
  hub (Ø19) into the body, then to the cassette with a service loop. Not modelled as solids: route and tie
  them on the first build.

## Reinforcement learning (`rl/`)

A MuJoCo model of this robot (`quad_model.py`): the CAD's masses, the hips where the CAD puts them, the 4-bar
modelled exactly as its two outputs (thigh = output B, the shin's absolute angle = output A), ±90° outputs,
the knee within ±60°, ab/ad ±20°. Each output gets 31.5 N·m up to half its no-load speed (the moteus current
limit), falling to zero at 23.3 rad/s (6S, KV 90, 9:1), and the rotors' inertia × 81 as armature; output A
is held to 25 N·m in the direction that pushes the coupler (v9.1's one software limit). PD at 500 Hz
under a 50 Hz policy. Domain randomisation: masses ±10 %, friction 0.6–1.25×, shoves.

Trained with PPO (stable-baselines3, 12 parallel environments):

**Walk** (`policies/walk.zip`; 40 M steps, then 15 M fine-tuning): tracks a commanded forward speed, sideways
speed and turn rate. In evaluation (three seeds, randomised masses and friction, 8 s per command):

| Command | Achieved |
|---|---|
| 0.5 m/s forward | 0.50–0.52 m/s |
| 0.9 m/s forward | 0.86–0.89 m/s |
| 0.4 m/s + 0.6 rad/s turn | 0.40 m/s, 0.57–0.58 rad/s |
| 0.25 m/s sideways | 0.23–0.24 m/s |
| 0.2 m/s backward | 0.12–0.14 m/s (weak, but no falls) |

The legs' loads while walking, from v9.1's 4-bar statics on every simulated foot contact: coupler 880 N
(636ZZ s0 1.56), knee 1240 N (6000ZZ s0 1.58), coupler push ≤ 450 N (the 25 N·m limit; buckling SF 2.3), hub
insert pull-out 263 N with the pod on its trunnion (SF 3.8; 664 N, SF 1.5, if the trunnion carried nothing).

**Jump: scripted, no learning (`rl/scripted_jump.py`).** A hand-written controller does what the RL jump
could not: it lands every time. Feet are placed through the leg's inverse kinematics and tracked by joint
PD, with the same motor model and limits as training (torque-speed curve, output A's 25 N·m push limit):

1. stand, then crouch slowly to a 0.22 m leg (the rest leg is 0.35 m) with each leg carrying its share of the weight;
2. push: each foot in contact is driven with force control along a line 60° above horizontal, as hard as
   the torque limits allow (60° keeps the push inside the friction cone at the lowest friction it is tested with, 0.6);
   a foot that has left the ground stops pushing (pushing it on kicked the knee into its stop);
3. flight: the legs swing smoothly to a landing pose fixed in the world frame, so body pitch does not change where the feet land;
4. land: soft gains for 0.2 s, then back to standing.

Its 15 numbers were tuned by random search in simulation (`python scripted_jump.py search`, 8 randomised
seeds per candidate), not by hand; the result is in `rl/scripted_jump.json`.

| 20 jumps (mass ±10 %, friction 0.6–1.25) | |
|---|---|
| In the air | **0.36–0.49 m** (mean 0.41); 0.53 m of travel overall |
| Lands and stands, body never down | **20 / 20** |
| Joint-limit load | 0 (no use of the simulated stops) |
| Peak foot force | push-off 182 N, landing 396 N (2.3 × body weight on one foot) |
| v9.1 parts (4-bar statics) | pivots 739 N (636ZZ s0 1.85); coupler push 419 N (buckling SF 2.4); knee 912 N (s0 2.15) |
| Hub inserts | 288 N with the pod on its trunnion (SF 3.5); **1001 N if the trunnion carried nothing (SF 1.0)** |

It jumps about half as far as the RL long jump (0.74 m), which never lands. Film:
`video/quadruped_jump_scripted.mp4`; `mujoco/teleop_dog.py` runs it on Space. The hub's no-trunnion case is
the same one the follow-up review flagged: proof-test the hub before jumping the real robot.

**RL jump: not solved.** Two policies, neither the goal (a long jump that lands):

| Policy | In the air | Flight | Lands and stands (body never down) | Film |
|---|---|---|---|---|
| `policies/jump_long.zip` | **0.66–0.80 m** (mean 0.74; ~1.1 m of travel) | 0.43 s | **0 / 20**: it bottoms out and tips | `video/quadruped_rl_jump_long.mp4` |
| `policies/jump.zip` | 0.01–0.02 m | 0.13 s | **20 / 20** | `video/quadruped_rl_jump.mp4` |

So the legs can throw the robot 0.7–0.8 m (in simulation, within the motor model); catching that landing
is not learned yet. Three tries to bridge it failed: fine-tuning the long jumper with more reward for
standing after landing, half its episodes starting mid-air (landing practice), and a distance curriculum
from the policy that lands (it never reached the first 5 cm target). Next to try: a landing-only policy
trained from mid-air starts, then switched in at touchdown.

Loads in v9.1's parts over 20 jumps each (4-bar statics on every foot contact): pivots s0 ≥ 1.39, knee
s0 ≥ 1.46, hub inserts SF ≥ 3.6 with the trunnion. The long jump's landing pushes the coupler to ~745 N
by statics, which is past the 455 N its 25 N·m limit allows (buckling SF 1.4); that estimate includes the
landing's dynamic loads, not just motor torque, and is a reason to treat any landing as untested.

**What the reviews and the load analysis changed in training.** The first walk policy stood on the
simulated knee stop (±60°) in ~40 % of its steps, with up to 40 N·m on it; the first jump policies landed on
it with 76 N·m. On the robot those are software limits (past them: the 4-bar's dead centre, the ropes'
anchors), so nothing could take that torque. Both policies are now trained with a penalty on any joint-limit
load: the fine-tuned walk touches a limit in under 1 % of its steps. The jump reward was rewritten four
times; each earlier version taught something other than a jump (a 0.19 m hop, a 0.10 m hop, 3 s of
bounding, trotting): see `JumpEnv`'s docstring. `eval_jump.py` reports every jump's loads in v9.1's parts.

Simulation only. Nothing here has run on hardware; sim-to-real needs the measured motor curve, the rope
compliance and the foot's real contact.

## Checks

All pass:

| Check | Result |
|---|---|
| `src/check_quad.py` | the four pods at ab/ad 0 and ±20° against each other, the body and the ab/ad drives; the body parts against each other; **the body's 132 screws, inserts and nuts against the pods (every pose), the drives and the electronics**: clear |
| `src/check_assembly_quad.py` | 180 items the body build fits (cassette, bulkhead halves, edge inserts, dock, hub screws, the pods' ear screws, plates, lid): each reachable along its tool path when the film fits it (the pods' own hardware: v9.1's `check_assembly.py`, 203 items) |
| `src/stress_test.py` | bulkhead FEA (centre, inner, end frame) at 585 N per foot (31.5 N·m outputs), ×2 impact and one leg ×2 (twist): worst SF 6.0 in PETG (inner bulkhead, ×2 impact; 0.18 mm), SF ≥ 37 in 6061. Results in `stress_test/results.md` |

The film's build order is the one checked. The real faults the checker found were fixed in the design or the
order: hub screws behind drum A, nuts inside the hub, the bulkhead halves' joint (unlisted, and its old bolts
ran into the edge inserts' holes), the dock's inserts, inserts sunk below their posts, screws that rode in
with a part they are fitted after. Three false positives were fixed in the checker itself: an insert's own
hole rim, fasteners not fitted yet (still in the parts bin), a nut's own bolt.

## Design review

Three Codex reviews (strength, function and buildability, compactness) were checked against the code;
the leg's findings and actions are in [v9.1's README](../v9_1_buildable/README.md#design-review). For the body:

| Finding | Verdict | Action |
|---|---|---|
| The quadruped build never fitted the four ab/ad ropes | **right** | a step after the drums go in, while the bay is still open from above |
| Battery under two shelves: the whole rack had to come out to swap it | **right** | battery tray on top, power board on the floor |
| No power switch or e-stop | **right** | fixed 22 mm e-stop in the top plate |
| Pi cooling: no air path | right | open rack, shelf and floor windows, lid vents |
| Cassette's board posts 1.2 mm round the inserts | **right** | r 4.5 posts: 2.7 mm round the Ø3.6 insert hole |
| Jump policy dropped 4 observations (incl. "landed") | **right** | fixed; retrained |
| Motor model kept full torque to 85 % of no-load speed | **right** | full torque to 50 %, then linear to zero; walk re-checked, jump retrained |
| Training used lighter default masses | **right** | `masses.json` by default |
| Lower the bottom plate 10 mm | tried: at ab/ad −20° drum A and its peg reach Z −133, inboard of the notches | kept at −140 |
| Shorten the body 20–30 mm by moving the end supports in | right in principle: the end frames set the length | v10 (a different end support), not a v3 change |
| Narrow the top plate | ~40 cm³ saved | not done |
| Keep printed bulkheads + two cut plates (not the all-laser bulkheads) | agreed | this is the v3 build |
| *Follow-up review:* landings not turned into part loads; the jump reward could be gamed | **right** | `eval_jump.py` turns every foot contact into v9.1 part loads, which exposed the policies leaning on the simulated knee stop; retrained against it; jump reward rewritten (below) |
| *Not from a review:* the bulkhead halves' joint and the dock's inserts were missing from the build | found by `check_assembly_quad.py` | added to the film and the BOM (countersunk M3 + nut through the laps; dock inserts on the bench) |
| *Not from a review:* the assembly check skipped any part the film had faded for its camera, so it passed items it could not see past | found while making `mujoco/watch_assembly.py` (the lap steps' camera sat inside the opposite bulkhead) | the check now counts a faded part (`Film.alpha(..., ghosts=False)`); it then found two real faults: the power board's 4 inserts and 4 screws were under the cassette's own shelf and tray (now: access holes over them, the board moved 3 mm so the holes clear the Pi's posts), and 2 of each centre bulkhead's 3 lap nuts sat inside a boss 6.5 mm proud of the face (now: an Ø11 socket pocket down to the lap at each nut). The lap steps are filmed one bulkhead at a time from above |

## Regenerating

```
cd ../v9_1_buildable/src
python generate.py && python export_variants.py <leg_mesh_dir> leg_abad_gantry
cd ../../fullbuild_v3/src
python generate_quad.py <mesh_dir> <leg_mesh_dir>   # printed/, laser_cut/, legs/, film meshes
python check_quad.py                                 # clashes: pods at ab/ad 0 and ±20 against everything
python check_assembly_quad.py <mesh_dir>             # every body screw and insert reachable when fitted
python stress_test.py                                # bulkhead FEA
python make_quad_film.py <mesh_dir>                  # video/quadruped.mp4
cd ../rl
python train.py walk 40 && python train.py jump 25   # policies/ (the shipped ones were also fine-tuned: RL_INIT=...)
python play.py walk <mesh_dir> && python play.py jump <mesh_dir> [--name=jump_long]
python eval_jump.py 20                               # jump distance and landing loads
python scripted_jump.py 20                           # the scripted jump: distance, landings, loads (search: tune it)
python play.py jump <mesh_dir> --name=scripted       # video/quadruped_jump_scripted.mp4
cd ../mujoco
python teleop_dog.py [--cad]                         # drive it: WASD, Q/E turn, R/F height, Space jump
python watch_assembly.py                             # the build film in a MuJoCo window, any speed (mujoco/README.md)
```
