"""Hardware tally from the exported CAD: every screw, insert and nut the build film fits, sized from its own
mesh (a screw's thread is the radius of the far 40 % of its shank; an insert's size from its knurl; a nut's
from its width across corners), grouped by size and the part group it belongs to. Bearings by name.

Run: python bom.py <mesh_dir> [variant]          (prints a Markdown table)
Library: tally(meta, verts) -> rows for any meta / vertex set (the quadruped's too).
"""
import ast
import collections
import json
import sys
from pathlib import Path

import numpy as np

INSERT_KNURL = {3.9: "M2.5", 4.6: "M3", 6.0: "M4", 7.0: "M5"}
NUT_AC = {5.8: "M2.5", 6.4: "M3", 8.1: "M4", 9.2: "M5", 11.5: "M6", 15.0: "M8"}


def _fit(v):
    f = v["fit"]
    return ast.literal_eval(f) if isinstance(f, str) else f


def size_of(kind, pts, d, length):
    c = pts.mean(0)
    rel = pts - c
    s = rel @ d
    rad = np.linalg.norm(rel - np.outer(s, d), axis=1)
    if kind == "screw":
        far = s < s.min() + 0.4 * (s.max() - s.min())                  # away from the head (d points to the head)
        r = rad[far].max() if far.any() else rad.max()
        return f"M{round(2 * r * 2) / 2:g}"
    if kind == "insert":
        k = 2 * rad.max()
        return min(INSERT_KNURL.items(), key=lambda kv: abs(kv[0] - k))[1]
    if kind == "nut":
        k = 2 * rad.max()
        return min(NUT_AC.items(), key=lambda kv: abs(kv[0] - k))[1]
    return "?"


def tally(meta, verts, group_of=None):
    """rows: (kind, size, length, group, count)."""
    group_of = group_of or (lambda k, v: v.get("part", k))
    rows = collections.Counter()
    for k, v in meta.items():
        if "fit" not in v or k not in verts:
            continue
        f = _fit(v)
        d = np.asarray(f["dir"], float)
        d /= np.linalg.norm(d)
        size = size_of(f["kind"], np.asarray(verts[k]), d, f.get("len", 0))
        L = f"{f.get('len', 0):g}" if f["kind"] == "screw" else ""
        rows[(f["kind"], size, L, group_of(k, v))] += 1
    return sorted((k + (n,) for k, n in rows.items()), key=lambda r: (r[0], r[1], r[2] and float(r[2]), r[3]))


def bearings(meta):
    c = collections.Counter()
    for k, v in meta.items():
        p = v.get("part", k)
        if "Bearing" in p or p.startswith("REF_Bearing"):
            c[p] += 1
    return c


def main():
    import make_videos as mv
    mesh_dir = Path(sys.argv[1]).resolve()
    vname = sys.argv[2] if len(sys.argv) > 2 else "leg_abad_gantry"
    meta = json.loads((mesh_dir / f"{vname}.json").read_text())
    models, _ = mv.load(mesh_dir, meta)
    verts = {k: t.reshape(-1, 3) for k, t in mv.load_tris(models["light"], meta).items()}
    print(f"| Kind | Size | Length (mm) | Part group | Qty |\n|---|---|---|---|---|")
    for kind, size, L, grp, n in tally(meta, verts):
        print(f"| {kind} | {size} | {L} | {grp} | {n} |")
    tot = collections.Counter()
    for kind, size, L, grp, n in tally(meta, verts):
        tot[(kind, size, L)] += n
    print("\nTotals:\n\n| Kind | Size | Length (mm) | Qty |\n|---|---|---|---|")
    for (kind, size, L), n in sorted(tot.items(), key=lambda r: (r[0][0], r[0][1], r[0][2] and float(r[0][2]))):
        print(f"| {kind} | {size} | {L} | {n} |")
    print("\nBearings (by part name):\n")
    for p, n in sorted(bearings(meta).items()):
        print(f"- {n}× {p}")


if __name__ == "__main__":
    main()
