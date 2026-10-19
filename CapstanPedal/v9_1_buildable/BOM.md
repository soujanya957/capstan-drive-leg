# Capstan v9.1: bill of materials (one leg module, `leg_abad_gantry`)

Counts are tallied from the exported CAD (`src/bom.py <mesh_dir>` sizes every screw, insert and nut from its
own mesh). This is the build the quadruped uses: 4-bar leg, ab/ad drive, front gantry. The rail stand's own
parts (MGN12 blocks and their screws, cradle rail screws) are listed last; the quadruped does not use them.
Suppliers and part numbers are not checked: confirm lengths against your hardware.

## Motors and electronics

| Qty | Item | Where |
|---|---|---|
| 3 | EaglePower 8308 (90 KV) + Ø8 × 2 diametric magnet | motors A, B (motor plate), C (ab/ad) |
| 3 | moteus r4.11 | behind each motor |

## Bearings

| Qty | Bearing | Where |
|---|---|---|
| 4 | 6806 (30 × 42 × 7) | 2 in the motor plate on drum A's shaft; 2 inside drum B on A's tube |
| 1 | 6806 | front trunnion (a deliberate slip fit) |
| 3 | 6808 (40 × 52 × 7) | front wall (drum B's journal); 2 on the ab/ad hub |
| 3 | 6902 (15 × 28 × 7) | front wall (the two small-drum tips); motor C's tip |
| 1 | 6903 (17 × 30 × 7) | gantry, output A's stub |
| 1 | 6000ZZ (10 × 26 × 8) | knee |
| 2 | 636ZZ (6 × 22 × 7) | coupler on the crank pin; the shin's lever pin |

## Heat-set inserts (61)

| Qty | Size | Where |
|---|---|---|
| 12 | M2.5 (Ø3.6 hole) | moteus boards, 4 each (motor plate ×8, ab/ad outer plate ×4) |
| 6 | M3 (Ø4.0 hole) | flange A (the crank's screws) |
| 4 | M3 | drum A's nose (flange A) |
| 6 | M3 | flange B (the thigh's screws) |
| 6 | M3 | drum B's journal end (flange B) |
| 3 | M3, short (≤ 4 mm) | shin back plate, pressed from its outer face |
| 8 | M4 (Ø5.6 hole) | ab/ad hub's end face (the rear end bracket) |
| 8 | M4 | posts, both ends |
| 2 | M4 | front wall (gantry bolts) |
| 2 | M4 | ab/ad inner plate (post bolts) |
| 4 | M4 | front end bracket (trunnion screws) |

## Screws, pins, nuts

| Qty | Item | Where |
|---|---|---|
| 12 | M2.5 × 6 socket head | moteus boards |
| 12 | M2.5 M-F standoff, 8 mm | moteus boards |
| 12 | M3 × 10 socket head | small drums → rotors (4 each) |
| 16 | M3 × 10 socket head | crank → flange A (6), flange A → nose (4), thigh → flange B (6) |
| 6 | M3 × 16 socket head | flange B → drum B's journal |
| 3 | M3 × 16 countersunk | shin sandwich |
| 2 + 2 | M3 × 40 + M3 nylon-insert nut | foot |
| 12 | M4 × 12 countersunk (DIN 7991) | motor stators, 4 each |
| 4 | M4 × 10 countersunk | trunnion → front end bracket |
| 8 | M4 × 16 socket head | rear end bracket → ab/ad hub's inserts |
| 8 | M4 × 25 socket head | frame: motor plate and wall into the posts |
| 2 | M4 × 65 socket head | gantry |
| 2 | M4 × 70 socket head | ab/ad posts |
| 6 + 6 | M4 × 18 socket head, Ø2 cross-hole drilled 6.9 mm under the head + M4 nylon-insert nut | rope tensioner pegs (drums A, B, ab/ad) |
| 1 | Ø10 × 20 shoulder screw (ISO 7379, M8 thread) + M8 nylon-insert nut + Ø20 washer + Ø13 spacer | knee |
| 1 | M6 × 25 socket head + M6 nylon-insert nut + washers, Ø9 spacer, shim | crank pin |
| 1 | M6 × 35 socket head + M6 nylon-insert nut + washers, Ø9 spacers | lever pin |

## Rope

~3.6 m of Ø1.5 mm Dyneema SK99 (≥ 1.8 kN): loops A and B (10 turns on each small drum) and the ab/ad loop.
3 wraps on each peg's shank before the cross-hole.

## Rail stand only (not on the quadruped)

8× M3 × 14 (MGN12H blocks), 6× M4 × 16 (cradle rails), the MGN12 rail and 2 blocks, the stand's prints.

## Print (Bambu Lab H2C; `src/check_stl.py` checks every file)

PETG, 0.4 mm nozzle; `print/` holds every part in its print orientation. Print the **fit coupon** first
(README: *Fits and the test coupon*). Drum colours in the films: drum A teal, drum B orange, ab/ad drum
violet; print them in any colour.
