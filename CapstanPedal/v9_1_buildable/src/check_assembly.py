"""Physical-assembly check (v9.1): can every fastener actually be fitted, in the order the build film fits it?

For each screw, insert and nut, at the moment the film fits it, a tool path runs out from the item along
the direction it goes in from: a hex key from a screw's head (60 mm), a soldering-iron tip from an
insert's top (30 mm), a finger / pliers path from a nut (20 mm). Every part already in place at that
moment (visible in the film, at its pose then) is tested against that path: exact line-triangle
intersection along its axis, plus any vertex inside the tool's radius. A blocked path is the kind of
fault v9's hub nuts were (steps 68-69: nuts dropped into slots inside a hub that was already closed in).

Run: python check_assembly.py <leg_mesh_dir> [variant]       (default variant leg_abad_gantry)
Library: check(F, meta, tris, fitinfo, labels) for any film built on make_videos.Film.
"""
import json
import sys
from pathlib import Path

import numpy as np

TOOL = {"screw": (60.0, 1.5, 0.3), "insert": (30.0, 2.5, 1.0), "nut": (20.0, 2.0, 0.3)}   # path length, radius, start (mm)
# (an insert's path starts 1 mm up: it sits up to that far down its hole, whose rim the iron's tip enters)


def _segment_hits(o, d, L, tri, s0=0.3):
    """Triangles (n, 3, 3) the segment o + s d, 0 < s < L crosses (Moller-Trumbore)."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-9
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    sv = o - v0
    u = f * (sv * h).sum(1)
    q = np.cross(sv, e1)
    v = f * (q @ d)
    t = f * (q * e2).sum(1)
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > s0) & (t < L)


def _near_axis(o, d, L, r, pts, s0=0.3):
    rel = pts - o
    s = rel @ d
    rad = np.linalg.norm(rel - np.outer(s, d), axis=1)
    return (s > s0) & (s < L) & (rad < r)


def check(F, meta, tris, fitinfo, labels, pose=None, quiet=False):
    """fitinfo: label -> {"kind", "dir" (unit, the side it comes in from), "len"}. tris: label -> (n, 3, 3)
    model-frame triangles. pose(k, t) -> (R, p) (default F.pose_at). Returns [(label, step, blockers)]."""
    pose = pose or F.pose_at
    # checked when it is FITTED: the end of its own first move (its approach); items carried in with a
    # sub-assembly (no approach of their own) are not checked again when the sub-assembly moves
    arrive = {}
    for k in labels:
        offs = F.offsets.get(k, [])
        if offs and offs[0][2] is not None:
            arrive[k] = offs[0][2]
    labels = [k for k in labels if k in arrive]
    part = lambda m: meta[m].get("part", meta[m].get("base", m)) if isinstance(meta[m], dict) else m
    steps = sorted((c[0], c[2], c[3]) for c in F.captions if c[2])
    step_at = lambda t: next((f"{n} {h}" for t0, n, h in reversed(steps) if t0 <= t + 0.2), "?")
    bad = []
    for k in sorted(labels, key=lambda k: arrive[k]):
        f = fitinfo[k]
        if f["kind"] not in TOOL:
            continue
        L, r, s0 = TOOL[f["kind"]]
        if f["kind"] == "screw":
            r = 1.5 if f["len"] < 30 else 2.0
        t = arrive[k] + 0.01
        d = np.asarray(f["dir"], float)
        d = d / np.linalg.norm(d)
        R, p = pose(k, t)
        own = tris[k].reshape(-1, 3) @ R.T + p
        o = own[np.argmax(own @ d)]                                        # the end it is driven from
        o = own.mean(0) + d * ((o - own.mean(0)) @ d)                      # on its axis
        hits = []
        for m, tm in tris.items():
            if m == k or m == "Q_Build_Stand" or F.alpha(m, t, ghosts=False) < 0.5 or part(m) == part(k):   # its own stack
                continue                                                   # (faded for the camera still counts: it is there)
            if m in arrive and arrive[m] > t:                              # a fastener not fitted yet: in the parts bin
                continue
            Rm, pm = pose(m, t)
            w = tm @ Rm.T + pm
            if f["kind"] == "nut" and fitinfo.get(m, {}).get("kind") == "screw":   # the bolt this nut threads onto
                c = w.reshape(-1, 3).mean(0) - o
                dm = np.asarray(fitinfo[m]["dir"], float)
                if np.linalg.norm(c - d * (c @ d)) < 1.0 and abs(dm @ d) / np.linalg.norm(dm) > 0.99:
                    continue
            lo, hi = w.reshape(-1, 3).min(0), w.reshape(-1, 3).max(0)
            seg_lo, seg_hi = np.minimum(o, o + d * L) - r, np.maximum(o, o + d * L) + r
            if np.any(hi < seg_lo) or np.any(lo > seg_hi):
                continue
            if _segment_hits(o, d, L, w, s0).any() or _near_axis(o, d, L, r, w.reshape(-1, 3), s0).any():
                hits.append(meta[m].get("part", m) if isinstance(meta[m], dict) else m)
        if hits:
            hits = sorted(set(hits))
            bad.append((k, step_at(t), hits))
            if not quiet:
                print(f"BLOCKED  {k:44s} [{step_at(t)}]  by {', '.join(hits[:4])}{' ...' if len(hits) > 4 else ''}", flush=True)
    return bad


def main():
    import make_videos as mv
    mesh_dir = Path(sys.argv[1]).resolve()
    vname = sys.argv[2] if len(sys.argv) > 2 else "leg_abad_gantry"
    meta = json.loads((mesh_dir / f"{vname}.json").read_text())
    models, verts = mv.load(mesh_dir, meta)
    tris = mv.load_tris(models["light"], meta)
    F = mv.author_variant(meta, verts, vname, tris)
    labels = [k for k in meta if "fit" in meta[k] and any(o[2] is not None for o in F.offsets.get(k, []))]
    fitinfo = {k: meta[k]["fit"] for k in labels}
    print(f"{vname}: {len(labels)} fitted items checked along their tool paths", flush=True)
    bad = check(F, meta, tris, fitinfo, labels)
    print(f"{len(bad)} blocked" if bad else "pass: every screw, insert and nut can be reached when the film fits it")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
