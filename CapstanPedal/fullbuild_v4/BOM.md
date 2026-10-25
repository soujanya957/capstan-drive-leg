# Capstan quadruped v4: bill of materials

Same as [v3's BOM](../fullbuild_v3/BOM.md), item for item: 132 body hardware items and four v9.1 leg
modules. The generator lists the same body parts, re-dimensioned:

| Part | Change from v3 |
|---|---|
| Top plate (laser-cut, 3 mm) | 28 mm narrower (half-width 116); a full X-truss window set in each pod bay (`laser_cut/dxf`) |
| Bottom plate (laser-cut, 3 mm) | 28 mm narrower (96); full X windows clear of the drum notches |
| Bulkheads (printed) | 28 mm narrower (half-width 126); inner and centre ones cut clear of the ab/ad drums' ±14° sweep |
| End frames (printed) | narrower with the pods; still flat and one piece |

Print and cut from `printed/stl` and `laser_cut/dxf`. Pods: `../v9_1_buildable/print` (unchanged).

## Sensing (new in v4, for the learned controller)

The learned walking policy (`rl2/`) needs the body's ground speed, which nothing in v3's electronics measures:
the pi3hat's IMU gives rotation rate and tilt, the moteus encoders give the joints. A downward-facing optical-flow
sensor and a range sensor give it, the way small drones hold position: flow (pixels per frame) times height is
ground speed.

| Qty | Item | Notes |
|---|---|---|
| 1 | PMW3901 optical-flow breakout (e.g. Pimoroni PIM453) | SPI to the Pi 5; under the bottom plate, lens down, near the centre. Needs ≥ 80 mm to the ground (standing: ~0.3 m) |
| 1 | VL53L1X time-of-flight breakout (e.g. Pimoroni PIM373) | I2C to the Pi 5; beside the flow sensor, also facing down; leads up through a small hole in the plate |
| 4 + 4 + 4 | M2.5 × 6 mm nylon standoff, M2.5 × 6 screw, M2.5 nut | two per breakout, through Ø2.7 holes drilled in the bottom plate's centre bay (check the hole spacing on the boards you buy) |
| – | 6-wire SPI and 4-wire I2C leads to the Pi's header, through the cassette | not modelled |

`rl2/hw.py` holds the numbers the simulator assumes for these and for the existing sensors (rates, noise, delays);
replace them with measurements once the robot is built.
