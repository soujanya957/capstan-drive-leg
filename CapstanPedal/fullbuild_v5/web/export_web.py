"""Export the quadruped for the browser simulator (MuJoCo WASM + three.js on the portfolio site).

Writes to OUT (default: the website's public/sims/capstan-quad):
- quad.mjb: the compiled physics model of ../mujoco/model.py, with the visual meshes removed. They never
  collide and have no mass, so the dynamics are identical (checked below, step for step); .mjb rather than
  XML because the XML writer rounds to 6 digits and the walk drifts apart after a few seconds.
- visual.bin + quad.json: the visual meshes, merged per (body, colour), decimated by quadric edge collapse,
  positions quantised to uint16 in each chunk's box, in body frames. The page computes creased normals.
  quad.json also carries what the controller reads: frames.json's pod frames and jump_params.json.

Run (from fullbuild_v4/mujoco, conda env capstan-sim):  python ../web/export_web.py [OUT] [triangles]
"""
import json
import sys
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mujoco"))
import gait as G  # noqa: E402
import model as M  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[4] / "portfolio-website" / "public" / "sims" / "capstan-quad"
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 260_000     # triangles in all
MIN_TRI = 60


def physics_spec():
    spec = M.build_spec()
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for me in list(spec.meshes):
        spec.delete(me)
    spec.geom("floor").material = ""                               # textures only matter to MuJoCo's renderer
    for x in list(spec.materials) + list(spec.textures):
        spec.delete(x)
    spec.compiler.meshdir = ""
    return spec


def run(m, n=3000):
    """A walk, a turn, a stop and a jump: the torso's path."""
    d = mujoco.MjData(m)
    c = G.Controller(m, d)
    c.reset_pose()
    out = []
    for i in range(n):
        c.cmd = np.array([0.6 if i < 1500 else 0.0, 0.0, 0.5 if i > 1000 else 0.0])
        if i == 2200:
            c.jump()
        c.step(m.opt.timestep)
        mujoco.mj_step(m, d)
        out.append(d.qpos[:7].copy())
    return np.array(out)


def visual_chunks(m):
    """(body, colour) -> triangles in the body frame. The colour is the URDF material, the mesh name's suffix."""
    groups = {}
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        k = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(np.float64)
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]].astype(np.int64)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, k)
        key = (int(m.geom_bodyid[g]), name.rsplit("__", 1)[-1], tuple(np.round(m.geom_rgba[g], 4)))
        groups.setdefault(key, []).append((v, f))
    chunks = []
    for (b, colour, rgba), parts in groups.items():
        vs, fs, n = [], [], 0
        for v, f in parts:
            vs.append(v)
            fs.append(f + n)
            n += len(v)
        chunks.append(dict(body=b, colour=colour, rgba=rgba, v=np.concatenate(vs), f=np.concatenate(fs)))
    return chunks


def main():
    m_full = M.build()
    spec = physics_spec()
    m_phys = spec.compile()
    a, b = run(m_full), run(m_phys)
    err = float(np.abs(a - b).max())
    print(f"physics without meshes vs full model: max |qpos| difference {err:.1e} over {len(a)} steps")
    assert err == 0.0, "removing the visual meshes changed the dynamics"

    OUT.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m_phys, str(OUT / "quad.mjb"), None)
    m_back = mujoco.MjModel.from_binary_path(str(OUT / "quad.mjb"))
    assert float(np.abs(run(m_back, 500) - b[:500]).max()) == 0.0

    chunks = visual_chunks(m_full)
    total = sum(len(c["f"]) for c in chunks)
    keep = min(1.0, BUDGET / total)
    blob, meta = bytearray(), []
    n_tri = 0
    for c in chunks:
        v, f = c["v"], c["f"]
        if keep < 1.0 and len(f) * keep > MIN_TRI:
            v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1.0 - keep)
            v, f = v2.astype(np.float64), f2.astype(np.int64)
        lo, hi = v.min(0), v.max(0)
        span = np.maximum(hi - lo, 1e-9)
        q = np.round((v - lo) / span * 65535).astype(np.uint16)
        idx = f.astype(np.uint16 if len(v) <= 65535 else np.uint32).ravel()
        while len(blob) % 4:
            blob.append(0)
        off_v = len(blob)
        blob += q.tobytes()
        while len(blob) % 4:
            blob.append(0)
        off_i = len(blob)
        blob += idx.tobytes()
        meta.append(dict(
            body=mujoco.mj_id2name(m_full, mujoco.mjtObj.mjOBJ_BODY, c["body"]),
            colour=c["colour"],
            rgba=[float(x) for x in c["rgba"]],
            lo=[float(x) for x in lo], span=[float(x) for x in span],
            nvert=len(v), vert=off_v, nidx=len(idx), index=off_i, index32=bool(len(v) > 65535),
        ))
        n_tri += len(f)
    (OUT / "visual.bin").write_bytes(bytes(blob))
    frames = json.load(open(HERE.parent / "urdf" / "frames.json"))
    jp = json.load(open(HERE.parent / "mujoco" / "jump_params.json"))
    (OUT / "quad.json").write_text(json.dumps(dict(
        chunks=meta,
        frames={p: {k: frames[p][k] for k in ("ab", "hip", "leg_y")} for p in M.PODS},
        jump=jp,
        mass=float(m_phys.body_subtreemass[m_phys.body("torso").id]),
        triangles=dict(drawn=n_tri, cad=total),
    ), indent=1))
    print(f"{OUT}: quad.mjb {(OUT / 'quad.mjb').stat().st_size / 1e6:.2f} MB, visual.bin {len(blob) / 1e6:.2f} MB, "
          f"{len(meta)} chunks, {n_tri} of {total} triangles")


if __name__ == "__main__":
    main()
