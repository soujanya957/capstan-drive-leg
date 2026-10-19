# Capstan quadruped v3: bill of materials

Four v9.1 leg modules plus the body. Counts come from the CAD (`src/generate_quad.py`: `hardware()`
lists the body's 132 items; each module's are in [v9.1's BOM](../v9_1_buildable/BOM.md)). Suppliers and
part numbers are not checked.

## Legs: 4× the v9.1 `leg_abad_gantry` module

Everything in [v9.1's BOM](../v9_1_buildable/BOM.md) × 4, **less the rail-stand items** and the cradle plates
(`Hip_Outer_Plate`, `Hip_Inner_Plate`, `Hip_Front_Plate`, `Hip_Spine`): the body's bulkheads carry their
bearings instead. That is 12 motors (EaglePower 8308, 90 KV), 12 moteus r4.11, and per leg the bearings
listed there. Pods on −Y print mirrored.

## Body

| Qty | Item | Where |
|---|---|---|
| 2 | Laser-cut plates, 3 mm (6061 recommended; carbon works): `laser_cut/dxf` | top (with the cassette's opening and the e-stop hole), bottom (drum notches, dock screws) |
| 10 | Printed bulkheads (PETG): 2 end frames whole; inner and centre bulkheads in halves | `printed/stl` |
| 3 | Printed: cassette dock, cassette, lid | `printed/stl` |
| 36 + 36 | M3 heat-set insert (Ø4.0 hole, ≤ 5.7 mm) + M3 × 8 countersunk | plates into every bulkhead's edge bosses (18 top, 18 bottom) |
| 12 + 12 | M3 × 12 countersunk + M3 nylon-insert nut | the inner and centre bulkheads' half-laps, 3 each; on the bench |
| 4 + 4 | M3 heat-set insert + M3 × 6 countersunk | dock, from under the bottom plate |
| 4 + 4 | M3 heat-set insert + M3 × 20 thumb screw | lid, into the centre bulkheads' tops |
| 8 | M2.5 heat-set insert (Ø3.6 hole) | cassette: power board posts (4), Pi posts (4) |
| 4 | M2.5 M-F standoff, 12 mm | Pi 5 → pi3hat |
| 8 | M2.5 × 6 socket head | pi3hat (4), power board (4) |

## Electronics

| Qty | Item | Notes |
|---|---|---|
| 1 | Raspberry Pi 5 + a low heatsink (≤ 9 mm) | under the hat |
| 1 | mjbots pi3hat r4.5 | CAN-FD: one bus per leg (3 moteus each) |
| 1 | mjbots power_dist r4.5 | battery in, XT30 out to the legs and the Pi's supply |
| 1 | 6S LiPo, ≤ 46 × 108 × 40 mm (e.g. 1800 mAh, high C) | in the top tray; hook-and-loop strap |
| 1 | 22 mm latching mushroom e-stop (NC contact block) | top plate; to the power board's switch input, via a plug at the cassette |
| – | CAN and XT30 harness per leg, through the hollow ab/ad hub (Ø19), with a service loop | not modelled: route and tie on the first build |
