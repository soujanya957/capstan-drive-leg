"""Re-tune the jump for v5 (the head and the belly pack move the centre of mass): a grid over the crouch's fore-aft
offset, the front legs' extra extension and the front/rear push split, each scored on 12 draws (mass x0.9-1.1, floor
friction 0.6-1.25; seed 1, not jump_test.py's). The best goes to jump_params.json; then run jump_test.py.

Run: python jump_tune.py
"""
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

import gait as G
import model as M

HERE = Path(__file__).resolve().parent
BASE = json.load(open(HERE / "jump_params.json"))
DRAWS = np.random.default_rng(1).uniform((0.9, 0.6), (1.1, 1.25), (12, 2))


def jump(jp, scale, mu):
    m = M.build()
    m.body_mass[:] *= scale
    m.body_inertia[:] *= scale
    m.geom_friction[m.geom("floor").id, 0] = mu
    d = mujoco.MjData(m)
    c = G.Controller(m, d)
    c.jp = jp
    c.reset_pose()
    dt, t = m.opt.timestep, 0.0
    while t < 0.8:
        c.step(dt); mujoco.mj_step(m, d); t += dt
    z0 = d.xpos[m.body("torso").id][2]
    c.jump()
    zmax, tilt = z0, 0.0
    while t < 4.0:
        c.step(dt); mujoco.mj_step(m, d); t += dt
        R = d.xmat[m.body("torso").id].reshape(3, 3)
        zmax, tilt = max(zmax, d.xpos[m.body("torso").id][2]), max(tilt, float(np.degrees(np.arccos(min(1.0, R[2, 2])))))
    ok = R[2, 2] > 0.9 and d.xpos[m.body("torso").id][2] > 0.3
    return ok, zmax - z0, tilt


def score(p):
    jp = dict(BASE, X_C=p[0], FRONT_EXTRA=p[1], SPLIT=p[2])
    r = [jump(jp, s, mu) for s, mu in DRAWS]
    n = sum(ok for ok, _, _ in r)
    return n, -float(np.mean([t for _, _, t in r])), float(np.mean([h for _, h, _ in r])), p


def main():
    x, f, s = BASE["X_C"], BASE["FRONT_EXTRA"], BASE["SPLIT"]                 # round v4's tuned values
    grid = list(itertools.product([x + k * 0.015 for k in (-2, -1, 0, 1, 2)], [f - 0.02, f, f + 0.02], [s - 0.1, s, s + 0.1]))
    with Pool(8) as pool:
        res = sorted(pool.map(score, grid), reverse=True)
    for n, nt, h, p in res[:5]:
        print(f"{n}/12  peak tilt {-nt:.0f} deg  rise {h:.2f} m  X_C {p[0]} FRONT_EXTRA {p[1]} SPLIT {p[2]}")
    n, nt, h, p = res[0]
    json.dump(dict(BASE, X_C=p[0], FRONT_EXTRA=p[1], SPLIT=p[2]), open(HERE / "jump_params.json", "w"), indent=1)
    print("written jump_params.json")


if __name__ == "__main__":
    main()
