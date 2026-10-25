"""Does a build's robot fit rl2? Run it before training on a new build version (README.md, "A new build version").
CPU only, about a minute.

    JAX_PLATFORMS=cpu python check_robot.py                                  # this build (fullbuild_v4)
    CAPSTAN_BUILD=../../fullbuild_v5 JAX_PLATFORMS=cpu python check_robot.py # another build

Checks, for every task model (plain, terrain course, ball and board):
  - the build's mujoco/model.py and gait.py give what core.py reads (PODS, TAU, W0, A_PUSH, Leg, limits, FOOT_R ...)
  - the URDF names rl2 looks up exist: body "torso" with a collision box, geoms and sites foot_<pod>, "_rotor" joints
  - both models compile ("physics" for the GPU, "full" for the CPU) and have the same joints and actuators
  - the robot stands: PD on the default pose (no policy) for 2 s keeps the body level at about the standing height
    (flat ground and the terrain course's start; the ball and board need a policy)
  - one observation frame is finite and the size the policy expects (51 x 3)
  - a policy (policies/<name>.npz, --policy=walk2_best) runs on the new model without falling for 4 s (optional:
    a policy trained on another build will often fail here; this tells you whether to retrain)
"""
import sys

import jax
import mujoco
import numpy as np

import core as C
import env as E

opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {what}")


print(f"build {C.BUILD} (tag {C.TAG})")
print("interface")
for mod, names in ((C.M, ("PODS", "TAU", "W0", "A_PUSH", "build_spec")),
                   (C.G, ("Leg", "FOOT_R", "H_STAND", "ABAD_LIM", "OUT_LIM", "KNEE_SOFT"))):
    for n in names:
        check(hasattr(mod, n), f"{mod.__name__}.{n}")
check(len(C.M.PODS) == 4, f"four pods {C.M.PODS} (core.HIND / FRONT index them: hind {C.HIND}, front {C.FRONT})")

for extras in ((), ("terrain",), ("board",)):
    print(f"model {extras or 'plain'}")
    ms = {k: C.build_spec(k, extras).compile() for k in ("physics", "full")}
    m = ms["full"]
    check(ms["physics"].nu == m.nu == 12, f"12 actuators in both models ({ms['physics'].nu}, {m.nu})")
    L = C.Layout(m, extras)
    Lp = C.Layout(ms["physics"], extras)
    check(len(L.qa) == 12 and np.all(np.isfinite(L.default)), "12 driven joints, finite default pose")
    check(np.allclose(L.default, Lp.default), "same default pose in both models")
    for p in C.M.PODS:
        check(m.site(f"foot_{p}").id >= 0, f"foot site foot_{p}")
    check(L.sensor_dz < 0, f"torso collision box found: sensors {L.sensor_dz * 1000:.0f} mm below the torso origin")
    print(f"       mass {m.body_subtreemass[L.torso]:.2f} kg, standing height {L.z_stand:.3f} m, "
          f"range sensor at stance {L.tof_nom:.3f} m")
    check(C.hw.TOF_RANGE[0] < L.tof_nom < C.hw.TOF_RANGE[1] and L.tof_nom > C.hw.FLOW_MIN_H,
          "standing range inside the range and flow sensors' working distances")

    # stand on PD alone: the default pose must hold the robot up level (CPU, full model, the task's own reset). Not on
    # the ball and board: that needs a balancing policy, so only the observation is checked there
    task = E.make_task("balance" if "board" in extras else ("terrain" if "terrain" in extras else "walk"))
    ce = E.CpuEnv(task)
    cfg = {k: np.asarray(v[0]) for k, v in task.bank(8, 0).items()}
    cfg.update(push_t=np.int32(10 ** 6), fric=np.float32(1.0), mass_add=np.float32(0.0), strength=np.float32(1.0),
               kp_s=np.float32(1.0), kd_s=np.float32(1.0), w0_s=np.float32(1.0))
    with jax.default_device(jax.devices("cpu")[0]):
        obs = ce.reset(cfg, seed=0)
        check(obs.shape == (C.N_OBS,) and np.all(np.isfinite(obs)), f"observation {obs.shape}, finite")
        for _ in range(100):
            obs, *_ = ce.step(np.zeros(12))
    if "board" in extras:
        continue
    R = ce.d.xmat[L.torso].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    h = ce.d.qpos[2] - float(task.ground(ce.cfg, 0.0, 0.0))
    check(tilt < 10 and abs(h - L.z_stand) < 0.05, f"stands on PD alone 2 s: tilt {tilt:.1f} deg, height {h:.3f} m "
          f"(standing {L.z_stand:.3f})")

if "policy" in opts:
    print(f"policy {opts['policy']}")
    import play
    pol, task = play.load(opts["policy"])
    ce = E.CpuEnv(task)
    cfg = {k: np.asarray(v[0]) for k, v in task.bank(8, 0).items()}
    cfg.update(push_t=np.int32(10 ** 6))
    with jax.default_device(jax.devices("cpu")[0]):
        obs, fell = ce.reset(cfg, seed=0), False
        for _ in range(200):
            obs, r, terms, done, success, fallen, err = ce.step(pol(obs))
            fell |= fallen
            if done:
                break
    check(not fell, f"{opts['policy']} ({task.name}) 4 s without a fall (if not: retrain on this build)")

print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
