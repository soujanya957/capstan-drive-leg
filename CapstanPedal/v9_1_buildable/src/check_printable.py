"""Every printed part, in its exported print orientation, must fit the Bambu Lab H2C.

Uses the smallest zone (dual-nozzle, 300 x 320 x 320 mm) less a 5 mm margin; parts may be
turned 90 deg on the bed. Z is the print height of the exported STL.
"""
import generate as g

BED_X, BED_Y, BED_Z, MARGIN = 300.0, 320.0, 320.0, 5.0

bad = []
for name, (solid, placed, _) in g.build().items():
    bb = solid.bounding_box()
    x, y, z = bb.size.X, bb.size.Y, bb.size.Z
    fits = z <= BED_Z - MARGIN and ((x <= BED_X - MARGIN and y <= BED_Y - MARGIN) or
                                    (y <= BED_X - MARGIN and x <= BED_Y - MARGIN))
    print(f"{'ok  ' if fits else 'FAIL'} {name:24s} {x:6.1f} x {y:6.1f} x {z:6.1f} mm  qty {len(placed)}")
    if not fits:
        bad.append(name)
assert not bad, f"too large for the H2C: {bad}"
print("pass: all printed parts fit the H2C (300 x 320 x 320 mm dual-nozzle zone, 5 mm margin)")
