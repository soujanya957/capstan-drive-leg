"""Every STL in print/ must be a closed (watertight) single-body mesh that fits the Bambu Lab H2C.

Watertight: every edge is shared by exactly two triangles. One body: the triangles form a single
connected piece. Fit: 300 x 320 x 320 mm dual-nozzle zone less 5 mm, turning 90 deg on the bed allowed.
"""
import glob
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    d = open(path, "rb").read()
    if d[:5] == b"solid" and b"facet" in d[:300]:
        v = [[float(x) for x in l.split()[1:4]] for l in d.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
        return np.array(v).reshape(-1, 3, 3)
    n = struct.unpack("<I", d[80:84])[0]
    return np.frombuffer(d[84:84 + n * 50], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"].astype(float)


bad = []
for f in sorted(glob.glob(os.path.join(ROOT, "print", "**", "*.stl"), recursive=True)):
    tri = load(f)
    key = np.round(tri.reshape(-1, 3), 4)
    uniq, idx = np.unique(key, axis=0, return_inverse=True)
    faces = idx.reshape(-1, 3)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = int(np.sum(counts != 2))
    parent = np.arange(len(uniq))                                   # connected components (union-find)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in faces:
        for x, y in ((a, b), (b, c)):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
    bodies = len({find(i) for i in range(len(uniq))})
    sz = tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0)
    fits = sz[2] <= 315 and ((sz[0] <= 295 and sz[1] <= 315) or (sz[1] <= 295 and sz[0] <= 315))
    ok = open_edges == 0 and bodies == 1 and fits
    rel = os.path.relpath(f, ROOT)
    print(f"{'ok  ' if ok else 'FAIL'} {rel:34s} {len(tri):7d} tris  open edges {open_edges:3d}  bodies {bodies}  "
          f"{sz[0]:6.1f} x {sz[1]:6.1f} x {sz[2]:6.1f}")
    if not ok:
        bad.append(rel)
assert not bad, f"bad STLs: {bad}"
print("pass: every STL is watertight, one body, and fits the H2C")
