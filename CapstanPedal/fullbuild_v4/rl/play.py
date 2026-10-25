"""A trained walking policy (policies/<name>.npz) in the FULL v4 model on the CPU (quad.py: model.py with every
mesh, the rotors on their joints, real contacts): the model it was not trained on when it came from the GPU.

  python play.py eval [name] [--n=200] [--bank=base|hard]   success rate on random episodes (two 4 s commands,
                                                  a shove); --bank draws friction, mass, motor strength, gains,
                                                  commands and shove from quad.BANKS (--hard = --bank=base)
  python play.py film [name] [--stress]           ../video/v4_rl_<name>[_stress].mp4: a scripted drive, camera
                                                  following (MUJOCO_GL=egl over SSH on Linux); --stress on ice,
                                                  heavy, weak motors, fast commands and three shoves
"""
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

import policy as P
import quad as Q

HERE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 30
SCRIPT = [(1.5, (0.0, 0.0, 0.0)), (4.0, (0.6, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0)), (4.0, (0.4, 0.0, 0.8)),
          (3.0, (0.0, 0.3, 0.0)), (3.0, (0.0, -0.3, 0.0)), (3.0, (0.0, 0.0, -1.0)), (3.0, (-0.4, 0.0, 0.0)),
          (2.0, (0.0, 0.0, 0.0))]
# --stress: the hard bank's corner. Floor friction x0.3, +3 kg on the torso, motors at 70 %, faster commands, and
# a sideways shove (m/s) at the times given
STRESS = dict(fric=0.3, mass=3.0, strength=0.7, shoves={4.0: (0.0, 1.2), 9.5: (0.0, -1.4), 15.0: (1.0, 0.8)})
SCRIPT_STRESS = [(1.5, (0.0, 0.0, 0.0)), (4.0, (1.0, 0.0, 0.0)), (3.0, (1.4, 0.0, 0.0)), (3.5, (0.6, 0.0, 1.6)),
                 (3.0, (0.0, 0.5, 0.0)), (3.0, (0.0, 0.0, -1.8)), (3.0, (-0.6, 0.0, 0.0)), (2.0, (0.0, 0.0, 0.0))]


def fallen(q):
    return q.base()[3][2] > -0.5 or q.d.qpos[2] < 0.22 or not np.all(np.isfinite(q.d.qpos))


def episode(q, pol, rng, bank):
    """One random episode; bank None = nominal robot and floor, else a configuration drawn from quad.BANKS[bank]."""
    m = q.m
    if not hasattr(q, "fric0"):
        q.fric0, q.mass0 = m.geom_friction.copy(), m.body_mass.copy()
    m.geom_friction[:] = q.fric0
    m.body_mass[:] = q.mass0
    q.kp, q.kd = Q.KP.copy(), Q.KD.copy()
    q.set_strength(1.0)
    b = Q.BANKS[bank or "base"]
    u = lambda ab: rng.uniform(ab[0], ab[1])                           # noqa: E731
    if bank:
        m.geom_friction[:, 0] *= u(b["fric"])
        m.body_mass[q.torso] += u(b["mass"])
        q.set_strength(u(b["strength"]))
        q.kp, q.kd = q.kp * u(b["kp"]), q.kd * u(b["kd"])
    q.reset(yaw=rng.uniform(-math.pi, math.pi))
    q.phase = rng.random()
    lo, hi = np.array(b["cmd"]).T
    cmds = [rng.uniform(lo, hi) for _ in range(2)]
    a = rng.uniform(-math.pi, math.pi)
    push_t, push = rng.integers(60, 360), u(b["push"] if bank else [0.0, 0.8]) * np.array([math.cos(a), math.sin(a)])
    e_lin = e_yaw = n = 0.0
    for t in range(400):
        q.cmd = cmds[t // 200]
        if t == push_t:
            q.d.qvel[0:2] += push
        a, _, _ = q.step(pol(q.obs()), Q.T_GAIT)
        q.last_a = a
        if fallen(q):
            return False, True, np.nan
        if (t + 1) % 200 >= Q.SETTLE:
            _, v, w, _ = q.base()
            e_lin += (-v[0] - q.cmd[0]) ** 2 + (v[1] - q.cmd[1]) ** 2
            e_yaw += (w[2] - q.cmd[2]) ** 2
            n += 1
    rl, ry = math.sqrt(e_lin / n), math.sqrt(e_yaw / n)
    return rl < Q.OK_LIN and ry < Q.OK_YAW, False, rl


def evaluate(name, n_ep, bank):
    pol = P.Policy(HERE / "policies" / f"{name}.npz")
    q = Q.Quad()
    rng = np.random.default_rng(123)
    res = [episode(q, pol, rng, bank) for _ in range(n_ep)]
    ok = np.mean([r[0] for r in res])
    fell = np.mean([r[1] for r in res])
    rms = np.nanmean([r[2] for r in res])
    print(f"{name} on the full CPU model{f' ({bank} bank)' if bank else ''}: success {ok:.1%}, fell {fell:.1%}, "
          f"rms velocity error {rms:.3f} m/s over {n_ep} episodes")


def film(name, stress=False):
    pol = P.Policy(HERE / "policies" / f"{name}.npz")
    q = Q.Quad()
    m, d = q.m, q.d
    script, shoves = SCRIPT, {}
    if stress:
        m.geom_friction[:, 0] *= STRESS["fric"]
        m.body_mass[q.torso] += STRESS["mass"]
        q.set_strength(STRESS["strength"])
        script, shoves = SCRIPT_STRESS, dict(STRESS["shoves"])
    q.reset(yaw=0.0)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    out = HERE.parent / "video" / f"v4_rl_{name}{'_stress' if stress else ''}.mp4"
    out.parent.mkdir(exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    look, t, next_frame = d.qpos[0:3].copy(), 0.0, 0.0
    cmd = np.zeros(3)
    err = []
    t_fell = None
    for seconds, target in script:
        t_end = t + seconds
        while t < t_end - 1e-9 and (t_fell is None or t < t_fell + 2.0):     # film 2 s past a fall
            for ts in [ts for ts in shoves if t >= ts]:
                d.qvel[0:2] += shoves.pop(ts)
            cmd += np.clip(np.array(target) - cmd, -0.04, 0.04)          # 2 m/s^2 ramp at 50 Hz
            q.cmd = cmd.copy()
            a, _, _ = q.step(pol(q.obs()), Q.T_GAIT)
            q.last_a = a
            t += Q.DT * Q.DECIM
            _, v, w, _ = q.base()
            if t_fell is None:
                err.append(((-v[0], v[1], w[2]), tuple(cmd)))
                if fallen(q):
                    t_fell = t
            while t >= next_frame:
                look += (d.qpos[0:3] - look) * 0.2
                cam.lookat[:] = look - np.array([0.0, 0.0, 0.12])
                cam.distance, cam.azimuth, cam.elevation = 1.4, 120.0 + 6.0 * t, -15.0
                r.update_scene(d, cam, opt)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                ff.stdin.write(r.render().tobytes())
                next_frame += 1.0 / FPS
        if t_fell is not None:
            break
    ff.stdin.close()
    ff.wait()
    e = np.array([np.subtract(a, b) for a, b in err])
    print(f"{out.name}: {t:.1f} s, rms error fwd {np.sqrt(np.mean(e[:, 0] ** 2)):.3f} side {np.sqrt(np.mean(e[:, 1] ** 2)):.3f} m/s, "
          f"yaw {np.sqrt(np.mean(e[:, 2] ** 2)):.3f} rad/s, " + (f"fell at {t_fell:.1f} s" if t_fell else "upright"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
    name = args[1] if len(args) > 1 else "walk_gpu"
    if args[0] == "eval":
        evaluate(name, int(opts.get("n", 200)), "base" if "hard" in opts else opts.get("bank"))
    else:
        film(name, "stress" in opts)
