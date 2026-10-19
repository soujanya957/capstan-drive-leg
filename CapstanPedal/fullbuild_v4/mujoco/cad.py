"""The full CAD quadruped as a MuJoCo scene (mocap bodies, mm), from mujoco/meshes (a copy of
generate_quad.py's film meshes: regenerate with `python ../src/generate_quad.py <mesh_dir> <leg_mesh_dir>` and
point --meshes at it). Posed either from the physics model's state (pose_from_physics, as rl/play.py) or from
the build film's timeline (watch_assembly.py).
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import make_quad_film as qf                                          # noqa: E402

mv = qf.mv


class qm:                                                             # the leg names and rest angles
    PODS = ("LF", "LR", "RF", "RR")
    Q0 = (-60.0, -120.0)
MESHES = Path(__file__).resolve().parent / "meshes"


class Cad:
    def __init__(self, mesh_dir=MESHES):
        mesh_dir = Path(mesh_dir).resolve()
        q = json.loads((mesh_dir / "quad.json").read_text())
        self.meta, self.layout = q["meta"], q["layout"]
        for v in self.meta.values():
            if not v["file"].startswith("/"):
                v["file"] = str(mesh_dir / v["file"])
        self.model = qf.build_model(self.meta)
        self.data = mujoco.MjData(self.model)
        self.verts, self.tris = qf.mesh_data(self.model, self.meta)
        m = self.model
        self.mocap = {k: m.body_mocapid[m.body(k).id] for k in self.meta}
        self.geom = {k: m.body_geomadr[m.body(k).id] for k in self.meta}
        m.geom_group[self.geom["Q_Build_Stand"]] = 5                 # the bench block: hidden unless the film shows it
        self.pods = {k: (qf.classify(self.meta[k]), np.diag(self.meta[k]["mirror"]), np.array(self.meta[k]["shift"]))
                     for k in self.meta if "pod" in self.meta[k]}
        self.floor = m.geom("floor").id
        self.floor0 = m.geom_pos[self.floor].copy()
        self.states = {}
        self.R, self.pb = np.eye(3), np.zeros(3)

    def set_pose(self, R, pb, states, extra=None):
        """Torso rotation R and position pb (mm), per-pod leg states; extra(k) -> (R0, p0) applied first."""
        quat = np.zeros(4)
        self.R, self.pb, self.states = R, pb, states
        for k in self.meta:
            if k in self.pods:
                (kind, rides), D, sh = self.pods[k]
                Rk, pk = qf.pod_pose(kind, rides, D, sh, states[self.meta[k]["pod"]])
            else:
                Rk, pk = np.eye(3), np.zeros(3)
            if extra:
                R0, p0 = extra(k)
                Rk, pk = R0 @ Rk, R0 @ pk + p0
            self.data.mocap_pos[self.mocap[k]] = R @ pk + pb
            mujoco.mju_mat2Quat(quat, (R @ Rk).flatten())
            self.data.mocap_quat[self.mocap[k]] = quat
        tile = 2.0 * 2 * self.model.geom_size[self.floor][0] / 40.0   # the floor follows in whole checker tiles
        self.model.geom_pos[self.floor][:2] = self.floor0[:2] + np.round(pb[:2] / tile) * tile
        mujoco.mj_forward(self.model, self.data)

    def pose_from_physics(self, pm, pd):
        """Pose from quad_model's state (m, rad) -> CAD (mm, deg): as rl/play.py."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, pd.qpos[3:7])
        states = {}
        for p in qm.PODS:
            hip, knee, ab = (pd.qpos[pm.joint(f"{p}_{n}").qposadr[0]] for n in ("hip", "knee", "abad"))
            states[p] = qf.leg_state(qm.Q0[0] - math.degrees(hip), qm.Q0[1] - math.degrees(hip + knee),
                                     ab if p[0] == "L" else -ab)
        self.set_pose(R.reshape(3, 3), pd.qpos[0:3] * 1000.0, states)

    def ropes(self, scn, offset=None, show=lambda k: True, ab_show=lambda: True):
        """The capstan ropes, drawn into the scene (as the films do)."""
        R, pb = self.R, self.pb
        for p in qm.PODS:
            st = self.states[p]
            kc, kd = f"{p}__Motor_Plate", f"{p}__Abad_Drum"
            if show(kc):
                D, sh = self.pods[kc][1], self.pods[kc][2]
                off = offset(kc) if offset else 0.0
                for stage, motor, y_mid in (("A", mv.MA, mv.A_MID), ("B", mv.MB, mv.B_MID)):
                    pts = mv.rope_path(motor, y_mid, st["th"][stage])
                    pts = (st["Rab"] @ (pts - mv.AB_C).T).T + mv.AB_C
                    mv.draw_rope(scn, (pts @ D.T + sh + off) @ R.T + pb, 1.0, 1.3, False)
            if kd in self.meta and show(kd) and ab_show():
                D, sh = self.pods[kd][1], self.pods[kd][2]
                off = offset(kd) if offset else 0.0
                pts = mv.ab_map(mv.rope_path(mv.MB, mv.AB_MID, -st["ab"], ab=True))
                mv.draw_rope(scn, (pts @ D.T + sh + off) @ R.T + pb, 1.0, 1.3, False)
