# Capstan quadruped v5: bill of materials

Everything in [v4's BOM](../fullbuild_v4/BOM.md) (body, four v9.1 leg modules, sensing), with these changes:

- **Battery:** the 6S 1800 mAh pack in the cassette's top tray (40 Wh) is replaced by a 6S 5000 mAh pack in a
  belly pod (111 Wh). The top tray stays, empty: room for a second pack or a payload.
- **Bottom plate:** a 20 × 12 mm slot in the middle for the battery pigtail (`printed/stl`, `laser_cut/dxf`).
- **Face end frame:** v4's, with four pads for the head's inserts; print it from `printed/stl/End_Frame_R.stl`.
- **Plus the head** (below).

## Belly pod: printed (PETG, `printed/stl`)

| Part | Volume | Print |
|---|---|---|
| `Belly_Saddle` | 163 cm³ | flange down, walls and lugs up, no supports |
| `Belly_Cover` | 68 cm³ | flat, the rounded edge up; 4+ perimeters (it is the skid) |

## Belly pod: bought and hardware

| Qty | Item | Notes |
|---|---|---|
| 1 | 6S 5000 mAh LiPo, ≤ 166 × 48 × 48 mm (e.g. Gens Ace 5000 mAh 60C: 155–165 × 45–47 × 44–47, ~750 g) | XT90 to suit the power board |
| 1 | XT90 pigtail, ~150 mm, 10 AWG | power board's battery input, down through the plate's and the saddle's slots into the lane beside the pack |
| 1 | 2 mm foam pad, 160 × 45 | between the pack and the cover |
| 4 | M3 × 12 countersunk | through the saddle's flange and the plate into the middle four bulkheads' bottom inserts (these replace four M3 × 8) |
| 4 | M3 heat-set insert (4.6 OD × 5.7) | the saddle's lugs |
| 4 | M3 thumb screws, 10 mm | cover |

## Head: printed (PETG, `printed/stl`)

| Part | Volume | Print |
|---|---|---|
| `Head_Mount` | 187 cm³ | on its back plate (the face that meets the end frame): walls, cheeks and posts grow straight up, no supports |
| `Head_Face` | 28 cm³ | flat, countersinks up |
| `Camera_Cradle` | 36 cm³ | on the shelf's underside (the tabs are flat with it), no supports |

## Head: bought

| Qty | Item | Notes |
|---|---|---|
| 1 | NVIDIA Jetson Orin Nano Super Developer Kit (8 GB) | the dev kit's plastic base comes off; the carrier board screws to the head's posts through its 86 × 58 mm corner holes. 9–20 V in, ≤ 25 W |
| 1 | Stereolabs ZED Mini | the ZED SDK needs the Jetson's GPU. Or an Intel RealSense D435/D435i (runs on the Pi 5 too) or D455: the cradle is drilled for all three |
| 1 | USB-C (camera) to USB-A 3.x cable, ~0.4 m, **right-angle at the camera end** | the camera's ends are 8 mm inside the cheeks; the lead runs down the outside of a wall to the Jetson's ports |
| 1 | Pololu D36V50F12 (12 V, 4.5 A step-down, up to 50 V in) | 6S (18–25.2 V) in, 12 V to the Jetson; on the left wall with two cable ties |
| 1 | 5.5 × 2.5 mm barrel plug lead | converter to the Jetson |
| 1 | XT30 lead from the power_dist board's battery side (or an XT30 Y-lead) | converter's input, switched by the e-stop with everything else |
| 1 | Ethernet patch cable, slim, 0.5 m | Jetson to Pi 5; the Pi keeps the 400 Hz CAN loop, the Jetson does vision |

## Head: hardware

| Qty | Item | Where |
|---|---|---|
| 12 | M3 heat-set insert (4.6 OD × 5.7) | 4 in the end frame's pads, 4 in the face posts, 4 in the cradle's tabs |
| 4 | M2.5 heat-set insert (3.9 × 4) | the Jetson's posts |
| 4 | M3 × 10 socket head | head to the end frame, from inside the head |
| 4 | M2.5 × 6 socket head | Jetson carrier board |
| 4 | M3 × 8 countersunk | face plate (flush: the cradle swings past it) |
| 4 | M3 × 12 socket head | two pivots, two locks (one per cheek) |
| 2 | M2 × 4 socket head | ZED Mini (heads in counterbores: 1.7 mm of thread in its 2 mm holes) |
| (2) | M3 × 6 socket head | instead, for a D435 (its threads are 3 mm deep) |
| (2 + 2) | M4 × 8 socket head + M4 washer | instead, for a D455 (4 mm deep) |
| 4 | cable ties, 2.5 mm | converter (2), camera lead and Jetson leads (strain relief slots at the head's bottom front) |

Head mass (CAD and catalogue): printed parts ~190 g, Jetson ~175 g, ZED Mini 63 g, converter and hardware ~40 g:
about 0.47 kg, 0.39 m ahead of the body's centre.
