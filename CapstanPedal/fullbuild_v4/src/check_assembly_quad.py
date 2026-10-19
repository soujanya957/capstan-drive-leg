"""Physical-assembly check of the quadruped's build (fullbuild_v4): every screw, insert and nut the film fits,
at the moment it fits it, along its tool path (hex key from a screw's head, soldering-iron tip from an
insert) against every part already in place, as v9.1's check_assembly.py does for the leg module.

Run: python check_assembly_quad.py <quad_mesh_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

import make_quad_film as qf

sys.path.insert(0, str(qf.V62))
import check_assembly as ca                                            # noqa: E402


def main():
    qdir = Path(sys.argv[1]).resolve()
    q = json.loads((qdir / "quad.json").read_text())
    meta, lay = q["meta"], q["layout"]
    model = qf.build_model(meta)
    verts, tris = qf.mesh_data(model, meta)
    F, t_built, POD = qf.author(meta, verts, tris, lay)
    labels, fitinfo = [], {}
    for k, v in meta.items():                                       # what the body film fits itself; the pods' own
        if k not in F.fitted or not any(o[2] is not None for o in F.offsets.get(k, [])):     # hardware is fitted, and
            continue                                                # checked, in v9.1's module build
        f = dict(v["fit"])
        if "pod" in v:
            f["dir"] = list(np.diag(v["mirror"]) @ np.array(f["dir"]))
        labels.append(k)
        fitinfo[k] = f
    print(f"quadruped: {len(labels)} fitted items checked along their tool paths", flush=True)
    bad = ca.check(F, meta, tris, fitinfo, labels)
    print(f"{len(bad)} blocked" if bad else "pass: every screw, insert and nut can be reached when the film fits it")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
