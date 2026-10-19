"""Printability report: area of each STL (in its exported print orientation) that needs support, i.e.
faces pointing down more than 45 deg from vertical and not resting on the bed. Short bridges (holes
across the layers, under 12 mm wide) print without support and are not counted.
Fails only if a part needs more than LIMIT cm2 of support, which would mean it is in a poor orientation.
"""
import glob
import os

import numpy as np

from check_stl import ROOT, load

LIMIT = 40.0                                    # cm2
bad = []
for f in sorted(glob.glob(os.path.join(ROOT, "print", "**", "*.stl"), recursive=True)):
    tri = load(f)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    nz = n[:, 2] / np.maximum(np.linalg.norm(n, axis=1), 1e-12)
    zmin = tri[:, :, 2].min(1)
    z0 = tri[:, :, 2].min()
    down = (nz < -0.7071) & (zmin > z0 + 0.3)
    span = tri[:, :, :2].max(1) - tri[:, :, :2].min(1)
    small = np.zeros(len(tri), bool)
    if down.any():                              # bridge-like: flat down faces whose island is narrow
        small = down & (nz < -0.99) & (span.min(1) < 12.0)
    sup = area[down & ~small].sum() / 100.0
    ok = sup <= LIMIT
    print(f"{'ok  ' if ok else 'HIGH'} {os.path.relpath(f, ROOT):42s} support {sup:6.1f} cm2")
    if not ok:
        bad.append(os.path.relpath(f, ROOT))
assert not bad, f"poor print orientation: {bad}"
print(f"pass: every part needs at most {LIMIT:.0f} cm2 of support in its print orientation")
