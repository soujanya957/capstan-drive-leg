"""Battery range estimate: the model trots under gait.py and the motors' electrical power is summed. Prints the
mean draw, the runtime on v4's and v5's packs, and the centre of mass. Runs in fullbuild_v4/mujoco too (copy it).

Electrical power per motor: positive mechanical power over a 0.85 drive efficiency (no regeneration counted)
plus copper loss 1.5 I^2 R, with I = motor torque / Kt (Kt = 60 / (2 pi KV), KV 90, 9:1) and R = 0.12 ohm a
phase (EaglePower 8308-class, an assumption: not measured). Electronics: Pi 5 + pi3hat 7 W, 12 moteus 12 W,
and in v5 the Jetson (15 W mode) 15 W. Packs used to 80 %.

Run: python energy.py [speed m/s]
"""
import math
import sys

import mujoco
import numpy as np

import gait as G
import model as M

KT = 60.0 / (2 * math.pi * 90.0)
GEAR, R_PH, ETA = 9.0, 0.12, 0.85
PACKS = {"v4 top tray, 6S 1800 mAh": 40.0, "v5 belly, 6S 5000 mAh": 111.0}


def main(speed=0.6, secs=12.0):
    m = M.build()
    d = mujoco.MjData(m)
    c = G.Controller(m, d)
    c.reset_pose()
    head = "zed" in [m.camera(i).name for i in range(m.ncam)]
    elec = 7.0 + 12.0 + (15.0 if head else 0.0)
    dt = m.opt.timestep
    n0, n = int(1.0 / dt), int(secs / dt)
    e_mech = e_cu = 0.0
    x0 = None
    for i in range(n):
        c.cmd[:] = (speed if i > n0 else 0.0, 0.0, 0.0)
        c.step(dt)
        mujoco.mj_step(m, d)
        if i == 2 * n0:
            x0, t0 = d.qpos[0], d.time
        if i > 2 * n0:
            tau, w = d.actuator_force, d.actuator_velocity
            e_mech += float(np.clip(tau * w, 0.0, None).sum()) / ETA * dt
            e_cu += float((1.5 * (tau / GEAR / KT) ** 2 * R_PH).sum()) * dt
    T = d.time - t0
    v = abs(d.qpos[0] - x0) / T
    p_motor = (e_mech + e_cu) / T
    p = p_motor + elec
    com = d.subtree_com[m.body("torso").id] - d.xpos[m.body("torso").id]
    print(f"mass {float(m.body_subtreemass[m.body('torso').id]):.2f} kg, CoM {1000 * com[2]:+.0f} mm from the torso origin (z), "
          f"standing torso z {d.xpos[m.body('torso').id][2]:.3f} m")
    print(f"trot {v:.2f} m/s: motors {p_motor:.0f} W (mechanical {e_mech / T:.0f}, copper {e_cu / T:.0f}), electronics {elec:.0f} W, total {p:.0f} W")
    for k, wh in PACKS.items():
        h = 0.8 * wh / p
        print(f"  {k} ({wh:.0f} Wh): {60 * h:.0f} min, {v * 3.6 * h:.1f} km")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.6)
