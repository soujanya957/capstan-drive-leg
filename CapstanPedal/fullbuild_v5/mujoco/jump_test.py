"""Jump robustness on settings the jump was not tuned on: 30 draws of mass x0.9-1.1 and floor friction 0.6-1.25.
A jump passes if the robot lands and stands (upright, torso above 0.3 m, 2.5 s after take-off). Prints the pass
count and the torso's rise. Runs in fullbuild_v4/mujoco too (copy it).

Run: python jump_test.py [n]
"""
import sys

import mujoco
import numpy as np

import gait as G
import model as M


def one(scale, mu):
    m = M.build()
    m.body_mass[:] *= scale
    m.body_inertia[:] *= scale
    m.geom_friction[m.geom("floor").id, 0] = mu
    d = mujoco.MjData(m)
    c = G.Controller(m, d)
    c.reset_pose()
    dt = m.opt.timestep
    t = 0.0
    while t < 0.8:
        c.step(dt); mujoco.mj_step(m, d); t += dt
    z0 = d.xpos[m.body("torso").id][2]
    c.jump()
    zmax = z0
    while t < 4.0:
        c.step(dt); mujoco.mj_step(m, d); t += dt
        zmax = max(zmax, d.xpos[m.body("torso").id][2])
    R = d.xmat[m.body("torso").id].reshape(3, 3)
    ok = R[2, 2] > 0.9 and d.xpos[m.body("torso").id][2] > 0.3
    return ok, zmax - z0


def main(n=30):
    rng = np.random.default_rng(0)
    res = [one(rng.uniform(0.9, 1.1), rng.uniform(0.6, 1.25)) for _ in range(n)]
    rise = np.array([r for _, r in res])
    print(f"lands and stands {sum(ok for ok, _ in res)} of {n}; torso rise {rise.min():.2f}-{rise.max():.2f} m (mean {rise.mean():.2f})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
