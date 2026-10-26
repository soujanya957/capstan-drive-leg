"""A trained rl2 policy in the FULL v4 model on the CPU (every mesh, rotors on their joints), with the hardware's
sensors, delays and noise (hw.py): the model it was not trained on.

  python play.py eval <name> [--n=200] [--seed=2000]   success on held-out configurations (a bank the training
                                                       never drew), overall and per terrain type / level
  python play.py film <name> [--level=5] [--secs=30]   ../video/v4_rl2_<name>.mp4: terrain: one course per terrain
                                                       type at that level; walk: a scripted drive; skills: three
                                                       tries in a row. MUJOCO_GL=egl over SSH on Linux
"""
import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np

import env as E
import policy as P
import tasks as T

HERE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 30
WALK_SCRIPT = [(1.5, (0.0, 0.0, 0.0)), (4.0, (0.6, 0.0, 0.0)), (3.0, (1.0, 0.0, 0.0)), (4.0, (0.4, 0.0, 0.8)),
               (3.0, (0.0, 0.3, 0.0)), (3.0, (0.0, -0.3, 0.0)), (3.0, (0.0, 0.0, -1.0)), (3.0, (-0.4, 0.0, 0.0)),
               (2.0, (0.0, 0.0, 0.0))]


def load(name):
    path = HERE / "policies" / f"{name}.npz"
    return P.Policy(path), E.task_for(path)


def row(bank, i):
    return {k: np.asarray(v[i]) for k, v in bank.items()}


def evaluate(name, n, seed):
    pol, task = load(name)
    ce = E.CpuEnv(task)
    bank = task.bank(n, seed)
    ok = np.zeros(n, bool)
    fell = np.zeros(n, bool)
    for i in range(n):
        obs = ce.reset(row(bank, i), seed=seed + i)
        for _ in range(task.EP):
            obs, r, terms, done, success, fallen, err = ce.step(pol(obs))
            if done:
                ok[i], fell[i] = success, fallen
                break
    kind, level = np.asarray(bank["kind"]), np.asarray(bank["level"])
    print(f"{name} ({task.name}) on the full CPU model: success {ok.mean():.1%}, fell {fell.mean():.1%} over {n} held-out episodes")
    table = {}
    if task.name == "terrain":
        print("level        " + " ".join(f"{lv:>5d}" for lv in range(10)))
        for k, kn in enumerate(T.TERRAINS):
            cells = [ok[(kind == k) & (level == lv)] for lv in range(10)]
            print(f"{kn:<12} " + " ".join(f"{c.mean():5.2f}" if len(c) else "    -" for c in cells))
            table[kn] = [float(c.mean()) if len(c) else None for c in cells]
    else:
        cells = [ok[level == lv] for lv in range(10)]
        print("by level     " + " ".join(f"{c.mean():5.2f}" if len(c) else "    -" for c in cells))
        table["all"] = [float(c.mean()) if len(c) else None for c in cells]
    out = HERE / "logs" / f"{name}_cpu_eval.json"
    out.parent.mkdir(exist_ok=True)
    json.dump(dict(success=float(ok.mean()), fell=float(fell.mean()), n=n, table=table), open(out, "w"))


class Film:
    def __init__(self, ce, out):
        self.ce = ce
        self.r = mujoco.Renderer(ce.m, H, W)
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                    "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        self.t, self.next = 0.0, 0.0
        self.look = None

    def frame(self, dist=1.6, elev=-15.0):
        d = self.ce.d
        self.t += 1.0 / 50
        while self.t >= self.next:
            self.look = d.qpos[0:3].copy() if self.look is None else self.look + (d.qpos[0:3] - self.look) * 0.2
            self.cam.lookat[:] = self.look - np.array([0.0, 0.0, 0.12])
            self.cam.distance, self.cam.azimuth, self.cam.elevation = dist, 120.0 + 4.0 * self.t, elev
            self.r.update_scene(d, self.cam, self.opt)
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            self.ff.stdin.write(self.r.render().tobytes())
            self.next += 1.0 / FPS

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def film(name, level, secs):
    pol, task = load(name)
    ce = E.CpuEnv(task)
    out = E.C.VIDEO / f"{E.C.TAG}_rl2_{name}.mp4"
    out.parent.mkdir(exist_ok=True)
    f = Film(ce, out)
    log = []
    if task.name == "walk":
        bank = task.bank(1, 7)
        cfg = row(bank, 0)
        cfg.update(push_t=np.int32(10 ** 6), yaw0=np.float32(0.0), fric=np.float32(1.0), mass_add=np.float32(0.0),
                   strength=np.float32(1.0), kp_s=np.float32(1.0), kd_s=np.float32(1.0), w0_s=np.float32(1.0))
        obs = ce.reset(cfg, seed=3)
        cmd = np.zeros(3)
        fell = False
        for seconds, target in WALK_SCRIPT:
            for _ in range(int(seconds * 50)):
                cmd += np.clip(np.array(target) - cmd, -0.04, 0.04)
                ce.aux = dict(ce.aux, cmd=jp.asarray(cmd, jp.float32), t=jp.int32(0))
                obs, r, terms, done, ok, fallen, err = ce.step(pol(obs))
                f.frame()
                fell |= fallen
        log.append(f"scripted drive, {'fell' if fell else 'upright'}")
    else:
        bank = task.bank(400, 4242)
        level_arr, kind_arr = np.asarray(bank["level"]), np.asarray(bank["kind"])
        picks = ([int(np.flatnonzero((kind_arr == k) & (level_arr == level))[0]) for k in range(len(T.TERRAINS))]
                 if task.name == "terrain" else [int(i) for i in np.flatnonzero(level_arr == level)[:3]])
        for i in picks:
            cfg = row(bank, i)
            obs = ce.reset(cfg, seed=i)
            f.look = None
            res = "time-out"
            for _ in range(task.EP):
                obs, r, terms, done, ok, fallen, err = ce.step(pol(obs))
                f.frame(dist=2.0 if task.name == "balance" else 1.6)
                if done:
                    res = "success" if ok else ("fell" if fallen else "time-out")
                    for _ in range(25):                                    # hold the last pose half a second
                        f.frame()
                    break
            what = T.TERRAINS[int(cfg["kind"])] if task.name == "terrain" else task.name
            log.append(f"{what} level {int(cfg['level'])}: {res}")
            if f.t > secs:
                break
    f.close()
    print(f"{out.name}: " + "; ".join(log))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
    if args[0] == "eval":
        evaluate(args[1], int(opts.get("n", 200)), int(opts.get("seed", 2000)))
    else:
        film(args[1], int(opts.get("level", 5)), float(opts.get("secs", 60)))
