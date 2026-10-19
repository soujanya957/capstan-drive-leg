# Capstan v9.1: the buildable v9

The leg module of the [fullbuild_v3 quadruped](../fullbuild_v3). v9 (v9_strong_compact, no longer in the repo)
made the module stronger; v9.1 makes it **physically assemblable in the order shown, with heat-set inserts
wherever a screw goes into plastic**, and folds in an external design review. v9 is unchanged.

`video/build_leg_abad_gantry.mp4` is the step-by-step build; `src/check_assembly.py` proves, for the film's
order, that every screw, insert and nut can actually be reached when it is fitted (see below).

## What changed from v9

| | v9 | v9.1 | Why |
|---|---|---|---|
| Ab/ad hub → rear end bracket | 6× M4 × 35 through-bolts, nuts dropped into slots *inside* the hollow hub (film steps 68–69) | **8× M4 heat-set inserts in the hub's end face** (pressed on the bench) + **8× M4 × 16** socket heads from the bracket's counterbores | the nuts could not be reached once the hub was in its bearings. 6× M3 inserts were tried first: pull-out SF 1.1 at the hub's moment, so 8× M4 (SF 2.5) |
| Hub screw order | – | **the rear end bracket goes onto the hub on the bench, before the module**; the module then slides into the bracket's ears | with the module in place, drum A sits 28 mm behind the bracket's web and blocks the hex key (found by `check_assembly.py`) |
| Coupler | 16 mm bow, 13 mm neck: ~63 MPa of bending at the crouched 891 N | **solid, 8 thick: 29 mm at the crank pin, a 26 mm waist offset 10.5 mm sideways past the hip, 48 mm in the middle, 34 × 4 mm rib on its back** (from 90 mm out, clear of the crank's hub; 1 mm clear of the shin's front plate) | the bow's bending: waist now ~15 MPa (SF 3.1). The offset clears the crank's gantry stub at the knee limit (a straight bar runs 461 mm³ into it). It pulls when standing (655–891 N) and is pushed only when output A drives the shin: out-of-plane buckling ~1020 N, so **output A is limited to 25 N·m in that direction** (455 N, SF 2.2; standing uses ~5 N·m the other way) |
| Crank and lever pivots | 626ZZ (C0 0.95 kN, s0 1.07 at 891 N) | **636ZZ (6 × 22 × 7, C0 1.37 kN, s0 1.54)** | review: the pivots were at their static rating |
| Flange B | 6 captured M3 nuts; odd curved slits | **6 M3 heat-set inserts; 6 simple radial pill slots** | assembly; looks |
| Flange A | 6 extra blind pockets | removed: only the crank's 6 inserts and the nose's 4 holes | they served nothing |
| Shin sandwich | 3 captured M3 nuts in hex pockets | **3 M3 heat-set inserts, pressed from the back plate's outer face** (the screws pull them deeper) | inserts; an insert pressed from the inside would be pulled out by its screw |
| Gantry bearing | correct in the CAD, **missing from the build film** (looked up under its v8 name) | in the film | – |
| Front wall | 6808 boss 4 mm, lower load path r 40; never run through FEA | **boss 6 mm, r 43** | first FEA run: SF 1.9 → 2.2 |
| Bearing journals | drawn 0.05–0.10 mm under the bore (a slip fit); `check_fits.py` mis-signed them as interference | **drawn nominal (`JOURNAL_CLR = 0`)**: printed outside diameters come out ~0.05–0.15 over, a light press | review: a slip-fit race creeps on a printed journal |
| Knee callout | film said "M6 × 30" | **Ø10 × 20 shoulder screw (ISO 7379, M8 thread), Ø20 washer, Ø13 spacer, M8 nylon nut** | it was always a 6000 on a Ø10 shoulder screw in the CAD |
| Drum colours | all red | **drum A teal, drum B orange, ab/ad drum violet** | tell the three capstans apart |

## v9 (summary)

9:1 capstans (v8: 8:1): 31.5 N·m per output at the moteus 40 A limit; a 6903 on output A's stub in the
gantry (v8: 6902); a 6000ZZ at the knee on a Ø10 shoulder screw (v8: 636ZZ on M6); crank pin end r 15;
end brackets with lightening windows. 200 mm thigh and shin, ±90° outputs, knee ±60°, ab/ad ±20°.

## Assembling it: the rules this version is checked against

- **Nothing is screwed into a nut you cannot reach.** Every nut left is on an open face: the three pin
  stacks (nylon-insert nuts on M6 / M8, held with a spanner) and the six tensioner pegs' nuts (dropped
  into their slots before the drum goes in).
- **Heat-set inserts everywhere else** (M2.5 ×12, M3 ×25, M4 ×24 per leg module). Press each one before
  its part is closed in; the film shows when. Inserts in a screw's pull direction are pressed from the far
  side so the screw pulls them deeper.
- **The film's order is the build order.** `src/check_assembly.py` replays it: for each screw a 60 mm hex
  key path (Ø3–4) from its head, for each insert a 30 mm soldering-iron path, for each nut a 20 mm finger
  path, against every part already in place at that moment. It reports nothing blocked.

## Servicing

Nothing is glued, so any part comes off by running the build film backwards to it. The common jobs:

- **A rope** (A or B): leg links off (lever, crank and knee pins; the thigh's 6 screws), the gantry (2× M4 × 65),
  the front wall (4× M4 into the posts); slacken both pegs, unwind; thread and tension a new loop as in
  the film's *Ropes* step (3 wraps on each peg's shank before its cross-hole); refit in reverse. The motor
  plate, motors and drums stay put.
- **An ab/ad rope** (quadruped): lid, top plate and that end's pods and inner bulkhead off (the film
  backwards), then as in the film's *Ab/ad ropes* step.
- **A pod off the robot**: top plate and that end frame off, then the bracket ears' screws; the pod slides
  out along the body the way it went in. Its hub and rear bracket stay on the body.

Rope and motor changes still take more teardown than they should; a cartridge front end is a v10 item.

## The one software limit

**Output A: 25 N·m in the direction that pushes the coupler** (positive in `rl/envs.py`'s convention; the
direction that extends the shin against something with the foot in the air). Standing and pushing off pull
the coupler, so the full 31.5 N·m stays available there. The RL policies are trained with it; on the robot,
clip the torque command (or set moteus `maximum_torque` to 25 whenever output A's command has that sign).

## Bearings (per leg module, ab/ad + gantry build)

| Qty | Bearing | Where |
|---|---|---|
| 4 | 6806 (30 × 42 × 7) | 2 in the motor plate on drum A's shaft, 2 inside drum B on A's tube |
| 1 | 6808 (40 × 52 × 7) | front wall, drum B's journal |
| 2 | 6902 (15 × 28 × 7) | front wall, the two small-drum tips |
| 1 | 6903 (17 × 30 × 7) | gantry, output A's stub |
| 1 | 6000ZZ (10 × 26 × 8) | knee, on the Ø10 shoulder screw |
| 2 | 636ZZ (6 × 22 × 7) | coupler on the crank pin; shin's lever pin |
| 2 | 6808 | ab/ad hub (in the body's centre and inner bulkheads on the quadruped) |
| 1 | 6902 | motor C's small-drum tip |
| 1 | 6806 | front trunnion (floating: a deliberate slip fit, it takes the module's length change) |

Every part that turns runs on one of these; nothing turns on plastic.

## Fits and the test coupon

Bearing pockets are drawn +0.10 mm on diameter (press), journals nominal, M3/M4 clearance holes +0.4/+0.5,
M6 pin holes +0.4, insert holes per the insert maker (M2.5 Ø3.6, M3 Ø4.0, M4 Ø5.6). `src/check_fits.py`
lists every fit and the thinnest wall round each insert (all ≥ 1.6 mm; ≥ 2.0 round the bearings). **Print
the coupon first** (one 6806 pocket, one Ø30 journal, one M4 insert hole, one tensioner nut slot) and set
`PRESS` / `JOURNAL_CLR` in `src/generate.py` from it: a pocket the bearing presses into by hand with a vice,
a journal the inner race needs a light tap to go onto.

## Checks (v9.1)

All pass (`src/`, run on `leg_abad_gantry`, 200 mm legs):

| Check | Result |
|---|---|
| `check_assembly.py` (new) | 203 screws, inserts and nuts, each reachable along its tool path at the moment the film fits it |
| `check_travel.py` | both outputs over ±90° at 15° steps: every pose inside the knee range clear (the straight first cut of the coupler hit the crank's gantry stub at the knee limit; the offset waist clears it) |
| `check_variants.py` | 33 printed parts, 220 hardware items, 15 leg and ab/ad poses: no clashes |
| `check_leg.py`, `check_clearance.py`, `check_3dof.py` | 4-bar poses, the module at nine output poses, ab/ad ±20° with leg poses: clear |
| `check_fasteners.py` | 114 fasteners, 0 clashes, 0 blocked |
| `check_fits.py` | every fit in its FDM range; every insert wall ≥ 1.6 mm, every bearing wall ≥ 2 mm |
| `check_strength.py` | every item at its target (SF 2 on PETG, s0 ≥ 1.5 on the pivot bearings, SF 4 on the rope), except drum A's nose *without* the gantry, which the gantry build does not use |
| `validate_assembly.py`, `check_stl.py`, `check_overhangs.py` | insertion paths and bolt circles; every STL watertight, one body, fits the H2C; ≤ 40 cm² of support each |

FEA (`fea_check.py`: PETG, linear, the worst case of each load sweep; target SF 2):

| Part | Worst case | SF |
|---|---|---|
| Rear end bracket (8 M4 hub screws) | 682 N + 31.5 N·m | 2.6 |
| Crank | crouched pin, 891 N | 2.1 |
| Thigh | crouched knee, 994 N | 3.2 |
| Shin (core alone / bonded sandwich; 636 at the lever) | standing / crouched | 2.9 / 2.9 |
| Gantry plate | 651 N swept | 2.9 |
| Front wall (v9.1 boss) | 1885 + 651 N swept | 2.2 |
| Motor plate, with the gantry | swept | 6.6 |

The coupler is a straight-load bar and is checked by hand (`check_strength.py`): waist SF 3.1, out-of-plane
buckling SF 2.2 at output A's 25 N·m push limit (1.8 at the full 31.5).

Hand calculations and linear FEA only: no fatigue, no impact, nothing built or load-tested.

## Design review

Three independent Codex reviews of v9.1 and fullbuild_v3 (strength; function and buildability; compactness), read-only, against the code and the check logs.
Each finding was checked against the CAD before acting on it. What changed, and what did not and why:

| Finding | Verdict | Action |
|---|---|---|
| Coupler can buckle (no buckling check; windowed 8 mm bar) | **right**, though only when output A pushes the shin: standing, the coupler is in tension (655–891 N) | solid ribbed coupler, buckling ~1020 N; output A limited to 25 N·m in the pushing direction (455 N, SF 2.2); `check_strength.py` now checks it (stepped-column FE) |
| 626 pivots at their static rating (891 N vs C0 950 N) | **right** | 636ZZ (C0 1.37 kN, s0 1.54) at the crank and lever pins |
| Journals called interference in `check_fits.py` but drawn 0.05–0.10 under | **right** (the table was mis-signed) | journals nominal (`JOURNAL_CLR`), the table reads the CAD; tune with the coupon |
| Knee pin callout "M6 × 30" in the film | **right** (the CAD was always Ø10 shoulder + M8) | caption fixed |
| Front wall FEA predates the boss change | right at the time | rerun: SF 2.2 |
| Landing loads exceed what the motors hold in a crouch (341 N per foot at 4 g on two feet) | **partly**: the capstans are back-drivable, so a leg that is overloaded folds instead of loading the structure past the motor-limited cases the checks use; what remains is the impact transient (the rotor's inertia × 81 at the output) and a leg reaching its end of travel | the jump policy's landings are measured in the physics sim (peak foot force and torque, `fullbuild_v3/rl/eval_jump.py`); the knee's software limit keeps the links off their hard stops |
| Hub's 8 M4 inserts: 2.2 mm of wall outside them; pull-out at 4 g | partly: the hub moment is bounded by the motor-limited foot force (the check uses no front support at all: SF 2.5) | kept; flagged for the first load test (below) |
| Crank / nose / end brackets should be metal for impact and fatigue | a reasonable direction for v10; the static SFs are ≥ 2.1 at the motor limit | not changed |
| Rope SF 4 ignores pretension and the termination | **right** | assembly note: 3 wraps round the peg's shank before its cross-hole (friction takes most of the load off the hole), the cross-hole deburred and chamfered, pretension ~50 N (a luggage scale on a loop) |
| Shin sandwich's three screws may slip | the core alone meets SF 2.9; the plates add stiffness | not changed |
| Rope or motor replacement needs a lot of teardown | right | documented below (service order); a v10 item |

### Follow-up review (Codex, after the changes)

It confirmed the fixes (most as "partly": they hold in the CAD and checks, not yet on hardware) and raised:

| Finding | Action |
|---|---|
| Landing loads were never turned into loads in the parts | `fullbuild_v3/rl/eval_jump.py` now runs v9.1's 4-bar statics on every simulated foot contact. That exposed the real problem: the trained policies were **standing and landing on the simulator's knee stop** (up to 40–76 N·m), which on the robot is a software limit with nothing behind it. Both policies are now trained against any joint-limit load; walking then loads the coupler to 880 N (636 s0 1.56), the knee to 1240 N (6000 s0 1.58) and the hub inserts to 263 N (SF 3.8) |
| The 25 N·m push limit exists only in the simulator | still true: it must go into the robot's command layer (README: *The one software limit*); test both directions on a torque fixture before the leg is loaded |
| The jump reward could pay for take-offs that never land, and missed staggered landings | rewritten (fullbuild_v3 README: the jump's training history) |
| The cassette's plugs and cable loops are not modelled | true; to settle on the first wired build |
| Hub inserts without the trunnion's share | the propped case (pod on hub + trunnion) is now computed alongside the no-trunnion bound; a proof test of the hub joint is on the list below |

### Test before trusting it (none of this has been built)

1. Print the fit coupon; set `PRESS` / `JOURNAL_CLR`.
2. One module on the rail stand: hold 31.5 N·m on each output against a scale arm (static), then 1000 cycles
   of a crouch-to-stand; check the hub screws' torque, the ropes' tension and the pivots' play afterwards.
3. Drop the leg 10 cm onto its foot on the rail stand; then 20 cm.
4. Pull-test one hub end (8 inserts, a printed hub section) to 2 kN; check the inserts after.
5. With the controller: output A's 25 N·m push limit, both directions, on a torque arm.

---

*Below: inherited from v8 / v6.2. Where it disagrees with the v9.1 sections above, they win.*

## Inherited from v6.2

A Stanley-style 8:1 rope-capstan joint module
([hackaday.io/project/176726](https://hackaday.io/project/176726-stanley-the-capstan-based-quadruped-kit)),
created as a separate copy of `../v6_capstan_cradle`. It uses EaglePower 8308 (90 KV) motors on moteus
controllers. The same 2-DOF module (two motors, two coaxial capstan outputs) builds eight ways:

| Build | Output | Ab/ad (3rd motor) | Front gantry | Mount |
|---|---|---|---|---|
| `flange` | 2 test arms | – | – | desk |
| `flange_gantry` | 2 test arms | – | ✓ | desk |
| `flange_abad` | 2 test arms | ✓ | – | desk |
| `flange_abad_gantry` | 2 test arms | ✓ | ✓ | desk |
| `leg` | 4-bar leg | – | – | MGN12 rail stand |
| `leg_gantry` | 4-bar leg | – | ✓ | MGN12 rail stand |
| `leg_abad` | 4-bar leg | ✓ | – | MGN12 rail stand |
| `leg_abad_gantry` | 4-bar leg | ✓ | ✓ | MGN12 rail stand |

- `assembly/<build>.step`: each build with every bearing, screw, nut and insert.
- `print/`: every printable STL in print orientation (`leg_4bar/`, `abad/`, `mounts/`).
- `step/`: the same parts as STEP.
- `BOM.md`: what each build needs, joint-by-joint pin stacks.
- `video/build_leg_abad_gantry.mp4` (1920 × 1080, 30 fps; the overlay now scales with `CAPSTAN_VIDEO_SIZE`): the full 3-DOF gantry build, step by step: every part, then every heat-set insert,
  screw, nut and washer that fixes it, one at a time along its own axis (screws turning in), with the
  camera close on the first; captured nuts go into their pockets before their screws, and any part
  between the camera and the hardware turns see-through while it goes in. Sub-assemblies (shin sandwich, ab/ad unit, spine and rail blocks) are built
  on the bench first. Then a short run.
  `video/jump_leg_abad_gantry.mp4`: that leg squatting, stepping and jumping on the rail.
  Every film also comes as `*_notext.mp4`, without captions or labels.

## What changed from the copied v6 snapshot

The copied v6 STLs were inspected as meshes before editing: all 38 printable files were closed,
single-body meshes that fit the H2C. The CAD and exports in this folder contain these reinforcement
changes:

| Location | v6 | v6.2 | Reason |
|---|---|---|---|
| 4-bar knee | 626ZZ, 6 × 19 × 6 mm, 950 N static rating | [NSK 636ZZ](https://www.nsk.com/id-id/engineering/636zz-esm-md.html), 6 × 22 × 7 mm, 1370 N static rating; local 9 mm boss and M6 × 30 pin | The calculated 861 N knee load left only 1.1 static bearing safety factor; now 1.6 |
| Shin core | 17 / 14 / 10 mm radii at knee / bend / claw; Ø30 lever end | 19 / 16 / 11 mm radii; Ø36 lever end | More section at the loaded bend and material around the lever bearing |
| Shin plates | 14.5 / 11.5 mm radii at the two lower web stations | 16.5 / 12.5 mm | Keep the sandwich plates aligned with the widened core |
| Front wall | 34 mm output bearing ring and 14 mm skeleton struts | 40 mm output bearing ring, 22 mm lower struts; motor struts stay 14 mm | Put material on the 6808 load path to the lower post bosses |

The other pivot bearings remain 626ZZ. The module envelope, eight build options, capstan ratio,
mounting interfaces and output range are unchanged. Use the v6.2 BOM for the longer knee pin and
larger knee bearing.

**Operating limit:** Builds without the front gantry still require output A limited to about
14 N·m (roughly 19 A) for a calculated safety factor of 2 at drum A's nose. Full 28 N·m operation
requires the gantry.

## v6 design history (from v5)

| | v5 | v6 |
|---|---|---|
| Ab/ad support | Hub at the rear end only: the whole module hung off one end | **Cradle**: the hub at the rear (2× 6808) and a Ø30 trunnion at the front (6806), so the module turns on bearings at both ends |
| Cradle frame | Rail plate behind the ab/ad unit | A front end bracket on the module carries the trunnion; a cradle front plate holds its bearing; a 311 mm spine behind the module ties the outer, inner and front plates together |
| Rail position (ab/ad leg) | Behind the ab/ad unit (x −150) | Behind the middle of the cradle (x −40), near the centre of mass |
| Front gantry | 16 mm in front of arm A / crank, on a long stub | In the coupler's plane, 1 mm in front of the crank or arm A; M4 × 55 bolts |
| Shin | 8 mm boomerang | Sandwich: red core (widened near the knee) between two 4 mm black plates, 16 mm thick below the knee, 3× M3 × 16 countersunk |
| Front wall | Solid | Skeleton like the motor plate: a body round the motor tips and drum B's 6808, struts to the posts, a lobe under each gantry post |
| Frame fixing | 4 long M4 through-bolts and nuts | M4 heat-set inserts in both ends of every post and rail; 8 short M4 screws (4 through the motor plate, 4 through the wall) |
| Shin core | Plain neck into the claw; 3 mm wall round the lever's 626 | A web blends the neck into the claw; Ø30 lever end (5 mm wall). Both found by FEA |
| Motor plate | Plain drum A housing on the plate | Conical buttress round the housing's root (FEA) |
| Motor mounts (all three) | Flat plates 4 mm bigger than the Ø92 motors; a 14 mm slot to thread the phase wires through (none for motor C) | Skeleton plates: a body round the motors and the output housing (or the ab/ad hub) with tapered struts to the posts and the spine tab; the motors overhang them, with an open 28 mm bay on each motor's wire side, so a motor goes on square with its board already soldered on. Motor plate 17% lighter; module 182 mm wide (was 193.5) |
| Stator screws | Heads flush; the two standoffs by the encoder sat on their rims | Countersinks start 1 mm below the face: the standoffs sit on plastic, 0.7 mm clear of the heads |
| Small drums | Ø40 disc base | Four lobes round the rotor screws on a Ø24 centre (less filament) |
| Lightening | Solid flanges, gantry and test arms | 6 windows in flange B (−13%), 6 blind pockets in flange A, a window in the gantry plate (−17%), windows between the test arms' load holes; every skeleton plate's inside corners filleted (R6) |
| Rope tensioners | Printed slider in a pocket, pulled by an M3 × 14 from the drum's end face (4 parts per rope end) | Winding peg: an M4 × 18 in from the rim at each end block, head sunk; the rope crosses a well round its shank, goes through a Ø2 cross-hole and out of the end face, and turning the screw winds it tight. M4 nylon-insert nut in a slot below the well holds it. No sliders; nothing proud of the drums |
| Desk mount, ab/ad | One desk angle on the ab/ad plates | Two desk angles, one at each end, bolted through the spine into the cradle plates |
| Colours | Yellow and black | Red and black (drums, flanges, crank, coupler, shin core, trunnion red; plates, thigh, shin plates black) |

Otherwise the 2-DOF module (motors, capstans, bearings) is unchanged.

## Mounting

- **Desk builds** stand on the desk, motors down and outputs up. Without ab/ad, the desk base bolts to
  the two top rails (4× M4 × 10 into heat-set inserts, heads sunk on the desk side). Four 5 mm
  countersunk wood screws, outside the module, hold it to the desk. With ab/ad, two desk angles sit
  behind the spine, one at each end of the cradle. The six M4 × 25 screws pass through each angle and the
  spine into the cradle plates' tab inserts. Each foot takes four wood screws.
- **Leg builds** hang on two MGN12H blocks on a vertical MGN12 rail on a 2040 extrusion stand, so the
  hip moves up and down freely, as on Stanley's single-leg rig. Without ab/ad, a rail carriage on the
  top rails carries the blocks (90 mm apart). With ab/ad, the spine carries them (100 mm apart), behind
  the middle of the cradle.

## The ab/ad cradle

The module turns about the ab/ad axis (along the body, 60 mm below the motors) between two ends:

- **Rear (driven and located):** motor C drives the hollow hub through a third 8:1 capstan. The hub runs
  in two 6808s, in the inner and outer plates, and bolts to the rear end bracket with 6× M4 × 35. These
  two bearings also hold the module axially.
- **Front (floating):** the front end bracket, a mirror of the rear one, clamps under the right-hand
  posts' bolts. A printed trunnion (4× M4 × 10 countersunk into heat-set inserts in the bracket's web)
  runs in a 6806 in the cradle front plate. Its Ø29.9 journal is a 0.1 mm slide fit, so the module can
  grow and shrink without loading the bearings axially.
- **Spine:** a 10 mm plate 85 mm behind the motor plate, with ribs along its edges and windows. It bolts
  to the three plates' rear tabs (6× M4 × 16 into inserts) and carries the rail blocks. Its top edge
  falls toward the front, where it only needs to reach the front plate's short tab. The module sweeps
  within 7 mm of it at the sector's full ±34°. `check_3dof.py` checks that sweep.

Assembly: build the module with both end brackets (the trunnion is already on the front one). Bolt the
hub of the finished ab/ad unit to the rear bracket. Slide the cradle front plate, with its 6806
already pressed in, over the trunnion. Bolt the rail blocks to the spine, then bolt the spine to the
three tabs.

## Checks

Run these from `src/` after regenerating the exports:

| Script | What it checks |
|---|---|
| `check_variants.py` | Each of the eight builds. Over the output range (±34.2°), the leg poses (squat 150–290 mm, ±15° swing) and ab/ad ±20°, nothing that moves touches anything it moves against. The mount, the cradle and their screws fit at rest. Every mount screw backs out along its axis without hitting anything (driver access) |
| `check_stl.py` | Every STL in `print/` is watertight, one body, and fits the Bambu Lab H2C |
| `check_overhangs.py` | Support area of every STL in its print orientation (no part over 40 cm²) |
| `check_fits.py` | Every designed fit (bearing press fits, journals, the floating trunnion, clearance holes, inserts, nut traps, double-D, tensioner peg) against FDM ranges, and the thinnest walls round inserts and nuts |
| `check_strength.py` | Hand calculations (see below) |
| `validate_assembly.py` | Assembly order: each part and bearing slides in along its real insertion path without hitting what is already there (both end brackets and the cradle front plate included) |
| `check_fasteners.py` | The module's fasteners: no clashes, a clear path for each screw |
| `check_clearance.py` | Module motion: nine output poses |
| `check_leg.py`, `check_3dof.py` | Leg poses with every pin, washer, nut, spacer, bearing and the shin plates; the module against the spine and cradle plates at ±34°; ab/ad capstan clear to ±34° |
| `check_printable.py` | Module parts fit the printer |
| `fea_check.py` | Linear-elastic FEA of the most loaded printed parts (below) |

On the v6.2 exports, all eight `check_variants.py` builds passed their sampled poses and mount
access checks. `check_stl.py` passed all 38 printable meshes, `check_overhangs.py` passed their
print orientations, `check_fits.py` passed the listed dimensions and wall thicknesses, and
`validate_assembly.py` passed the insertion paths. `check_fasteners.py` found zero clashes or blocked
paths across 114 checked fasteners; `check_3dof.py` passed the cradle and ab/ad sweep.

### Strength (`check_strength.py`)

Hand calculations: beam formulas, lever rules and catalogue ratings (the FEA below checks the lowest). There is no fatigue or
impact analysis, and nothing has been built or load-tested. Design load is the larger of two cases.
The first is the strongest foot push the motors can make: 28 N·m on each output (~40 A at 8:1),
373 N at the foot when standing. The second is a hop at 4× the estimated weight (~2.4 kg, 93 N).
Printed parts are rated as PETG: 45 MPa along the layers and 25 MPa across them. The weaker figure
is used wherever the stress crosses layers.

| Item | Safety factor |
|---|---|
| Knee pivot 636ZZ static rating (861 N) | 1.6 (v6: 1.1) |
| Crank / lever pivots 626ZZ (568 N) | 1.7 |
| Shin bending: core only near the knee / sandwich below | 3.2–3.7 / 7.9–8.5 (v6: 2.6–2.9 / 6.1–6.6) |
| A's nose bending, with gantry / **without gantry** | 6.4 (v5: 2.9) / **1.0** |
| Crank stub, PETG / PC | 7.8 / 10.9 |
| Gantry 6902, plate, posts, bolts | 4.7, 2.9, 4.2, 21 |
| Wall 6808 (drum B) | 2.8 |
| Rope at 28 N·m (Ø2 rope assumed ≥ 2 kN) | 5.0 |
| Ab/ad hub, inner 6808, hub bolts, nut slots, rear bracket web, counting no front support at all | 3.1, 3.8, 5.5, 2.6, 3.9 |
| Cradle front (189 N share): 6806, trunnion, its screws, front bracket web, front plate, its tab inserts | 15, 58, 65, 42, 18, 3.4 |
| Spine, vertical / sideways bending | 20 / 17 |
| MGN12H pair, 3-DOF / 2-DOF leg (one block alone would see 85 / 74 N·m) | 4.6 / 4.7 |
| Rail carriage upright, top-rail inserts | 6.0, 4.7 |
| Desk base / desk angle wood screws, desk angle upright | 7.1 / 7.1, 9.2 |
| Motor C holding the full foot force sideways | 1.0 (just enough at 28 N·m) |

The exact figures are printed by the script. **Builds without the gantry:** limit output A to about
14 N·m (≈ 19 A) to keep a safety factor of 2 on drum A's nose. The gantry carries about two thirds of
the leg's load into the front wall and is what makes full current safe.

### FEA (`fea_check.py`)

The parts the hand calculations rate lowest, meshed from the CAD (gmsh, 2nd-order tetrahedra) and
solved with scikit-fem under the same design loads. Bearings and pins press on their bores as a cosine
pressure; clamped faces are fixed; plate loads are swept round 8 directions. PETG is treated as
isotropic (45 MPa von Mises along the layers, 25 MPa across them). Stresses within 1.5 mm of a fixed face
and the hottest 0.1 % of the volume are left out. Target SF 2. The table below records the v6 baseline;
the v6.2 shin core and front wall results follow it.

| v6 baseline part | Worst case | von Mises (MPa) | SF | Before the v6 fixes |
|---|---|---|---|---|
| Shin core alone / bonded sandwich | standing, 373 N at the foot | 22.4 / 21.7 | 2.0 / 2.1 | 1.2 / 1.3: 3 mm wall round the lever's 626; the neck/claw corner at 2× the shaft's stress |
| Thigh | knee 861 N | 14.0 | 3.2 | |
| Crank (stub left free) | coupler pin 568 N | 17.3 | 2.6 | |
| Gantry plate | 482 N, swept | 11.3 | 4.0 | |
| Front wall (new windows) | 6808 1429 N + gantry 482 N + rope pull, swept | 22.0 | 2.0 | |
| Motor plate (skeleton), with gantry / without (at its 14 N·m limit) | drum A's 6806s, rope pull and torque, swept | 5.2 / 15.3 | 8.6 / 2.9 | 0.9 without the gantry: the housing's bare root |

The v6.2 core-only shin FEA, using the enlarged 636 knee pocket and reinforced profile, gives
15.4 MPa and SF 2.9 in the standing 373 N case, and 13.8 MPa and SF 3.3 in the crouched 250 N case.
This is a useful lower bound while the sandwich's bolted interfaces remain untested.
The v6.2 front wall FEA sweeps eight directions with 1429 N on the 6808, 482 N on the gantry mounts,
and rope pull on the two tip bearings. Its worst case is 18.7 MPa and SF 2.4 (315°), with 0.49 mm
maximum displacement. Both revised printed parts exceed the SF 2 static target in the model.

The shaft of the shin matches the hand calculation (17.5 MPa); FEA found the two local peaks the beam
formula can't see. Still linear and static: no contact, fatigue, impact or creep, and the sandwich's
screws are taken as bonded. Run the bonded and core-alone cases as the bounds.

## Print notes

PETG or better, 4 walls, 40 % infill; print the crank in PC as Stanley does. Bearing pockets are
+0.1 mm on diameter and journals −0.05 to −0.1 mm (the trunnion is +0.1 mm clear, to slide). Print one
bearing pocket, one insert hole and one tensioner nut slot first and tune if your printer runs tight or
loose. Print the shin plates in black and the core in red, so the core shows through the plates'
windows. The spine is 311 mm long: it fits the H2C's bed lengthwise only.
Press the 636ZZ into the shin's knee pocket from the thigh side; its inner race is clamped by the
M6 × 30 pin stack. Confirm the bearing turns freely after tightening the locknut.

## Regenerating

```
cd src
python generate.py                                   # STEP, STL, the eight assemblies (build123d)
python fea_check.py                                  # FEA (gmsh, scikit-fem, pyamg)
python check_variants.py && python check_stl.py && python check_overhangs.py && python check_fits.py \
  && python check_strength.py && python validate_assembly.py && python check_fasteners.py \
  && python check_clearance.py && python check_leg.py && python check_3dof.py
python export_variants.py <dir>
python make_videos.py <dir> --variants build_flange ... jump_leg_abad_gantry
```

`CAPSTAN_LEG_L=<mm>` (environment) lengthens the 4-bar's thigh and shin together; the shin's plates, bolts and
windows scale along it. Default 200 in v8 (v6.2: 150); the quadruped and v7 used 200.
