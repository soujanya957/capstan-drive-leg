"""Teacher, student and blind robot on the same terrain courses, side by side (Experiment 5, sgs-lessons.md). The
teacher's height scan (what it sees and the others do not) is drawn as dots on the ground. Full CPU model.

    MUJOCO_GL=egl JAX_PLATFORMS=cpu python compare_film.py [--teacher=terrain_teacher_s0] [--snap=59]
        -> ../video/v4_rl2_ts_<terrain>_<level>.mp4 (1920 x 720: teacher | student | blind), one per scene,
           ../video/v4_rl2_ts_all.mp4 (all of them), ../video/v4_rl2_ts_teacher_<terrain>_<level>.mp4 (1280 x 720,
           the teacher alone, orbiting camera), logs/ts_film.json (what happened in each panel)

The course of each scene is the first of that type and level in the film bank (seed 4242, as play.py film).
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import core as C
import env as E
import play
import policy as P
import tasks as T
import train as Tr

HERE = Path(__file__).resolve().parent
FPS = 30
SCENES = [("gaps", 3), ("gaps", 6), ("stairs_up", 7), ("mixed", 6), ("obstacles", 7), ("stairs_down", 6)]
HERO = [("gaps", 5), ("stairs_up", 8)]
SIDES = [("teacher", None), ("student", "terrain_student_s0_best"), ("blind", "terrain_blind_s0_best")]


class Teacher:
    """A teacher snapshot on the CPU: reads the observation and CpuEnv.priv."""

    def __init__(self, path, ce):
        ck = pickle.load(open(path, "rb"))
        self.ce = ce
        p, nm = jax.tree_util.tree_map(jp.asarray, ck["params"]), jax.tree_util.tree_map(jp.asarray, ck["norm"])
        self.f = jax.jit(lambda ob, pv: jp.clip(Tr.actor_mean(p, nm, ob, pv)[0], -1.0, 1.0))

    def __call__(self, obs):
        return np.asarray(self.f(jp.asarray(obs), jp.asarray(self.ce.priv)))


def scan_fn(task):
    @jax.jit
    def f(boxes, p, R):
        yaw = jp.arctan2(R[1, 0], R[0, 0])
        fwd, side = -jp.array([jp.cos(yaw), jp.sin(yaw)]), jp.array([-jp.sin(yaw), jp.cos(yaw)])
        xy = (p[:2] + jp.asarray(task.SCAN_FWD)[:, None, None] * fwd + jp.asarray(task.SCAN_SIDE)[None, :, None] * side).reshape(-1, 2)
        z = jax.vmap(lambda q: C.ground_z(boxes, q[0], q[1]))(xy)
        return jp.concatenate([xy, z[:, None]], 1)
    return f


class Cam:
    def __init__(self, ce, out, w, h, orbit=False):
        self.ce, self.w, self.h, self.orbit = ce, w, h, orbit
        self.r = mujoco.Renderer(ce.m, h, w)
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                    "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        self.t, self.next, self.look, self.dots = 0.0, 0.0, None, None

    def frame(self):
        d = self.ce.d
        self.t += 1.0 / 50
        while self.t >= self.next:
            self.look = d.qpos[0:3].copy() if self.look is None else self.look + (d.qpos[0:3] - self.look) * 0.15
            self.cam.lookat[:] = self.look - np.array([0.25, 0.0, 0.1])
            self.cam.distance = 2.0 if self.orbit else 1.9
            self.cam.azimuth = 100.0 + 9.0 * self.t if self.orbit else 112.0
            self.cam.elevation = -20.0
            self.r.update_scene(d, self.cam, self.opt)
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            if self.dots is not None:
                sc = self.r.scene
                for q in self.dots:
                    if sc.ngeom >= sc.maxgeom:
                        break
                    mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]),
                                        q + np.array([0, 0, 0.012]), np.eye(3).ravel(), np.array([0.1, 0.85, 0.75, 1.0], np.float32))
                    sc.ngeom += 1
            self.ff.stdin.write(self.r.render().tobytes())
            self.next += 1.0 / FPS

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def play_one(ce, pol, cfg, cam, scan=None):
    obs = ce.reset(cfg, seed=7)
    res, dist = "time-out", 0.0
    for _ in range(ce.task.EP):
        if scan is not None:
            cam.dots = np.asarray(scan(jp.asarray(cfg["boxes"]), jp.asarray(ce.d.qpos[:3]), jp.asarray(ce.d.xmat[ce.L.torso].reshape(3, 3))))
        obs, r, terms, done, ok, fallen, err = ce.step(pol(obs))
        cam.frame()
        dist = max(dist, -float(ce.d.qpos[0]))
        if done:
            res = "success" if ok else ("fell" if fallen else "time-out")
            break
    for _ in range(40):
        cam.frame()
    return dict(result=res, distance=round(dist, 2), goal=round(float(cfg["goal"]), 2))


def main():
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
    tname, snap = opts.get("teacher", "terrain_teacher_s0"), int(opts.get("snap", 59))
    task_t = E.make_task("terrain")
    ce_t = E.CpuEnv(task_t)
    teacher = Teacher(HERE / "policies" / "snaps" / tname / f"{snap:05d}M.pkl", ce_t)
    envs = dict(teacher=(ce_t, teacher))
    for side, name in SIDES[1:]:
        path = HERE / "policies" / f"{name}.npz"
        envs[side] = (E.CpuEnv(E.task_for(path)), P.Policy(path))
    scan = scan_fn(task_t)
    bank = task_t.bank(400, 4242)
    kind, level = np.asarray(bank["kind"]), np.asarray(bank["level"])
    vid = C.VIDEO
    tmp = HERE / "logs" / "ts_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    log, clips = {}, []
    for kn, lv in SCENES:
        i = int(np.flatnonzero((kind == T.TERRAINS.index(kn)) & (level == lv))[0])
        cfg = play.row(bank, i)
        tag = f"{kn}_{lv}"
        log[tag] = {}
        for side, _ in SIDES:
            ce, pol = envs[side]
            cam = Cam(ce, tmp / f"{tag}_{side}.mp4", 640, 720)
            log[tag][side] = play_one(ce, pol, cfg, cam, scan if side == "teacher" else None)
            cam.close()
        out = vid / f"{C.TAG}_rl2_ts_{tag}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + sum((["-i", str(tmp / f"{tag}_{s}.mp4")] for s, _ in SIDES), [])
                       + ["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", str(out)], check=True)
        clips.append(out)
        print(tag, log[tag], flush=True)
    lst = tmp / "all.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(vid / f"{C.TAG}_rl2_ts_all.mp4")], check=True)
    for kn, lv in HERO:
        i = int(np.flatnonzero((kind == T.TERRAINS.index(kn)) & (level == lv))[0])
        cfg = play.row(bank, i)
        cam = Cam(ce_t, vid / f"{C.TAG}_rl2_ts_teacher_{kn}_{lv}.mp4", 1280, 720, orbit=True)
        log[f"hero_{kn}_{lv}"] = play_one(ce_t, teacher, cfg, cam, scan)
        cam.close()
        print(f"hero {kn} {lv}", log[f"hero_{kn}_{lv}"], flush=True)
    json.dump(log, open(HERE / "logs" / "ts_film.json", "w"), indent=1)


if __name__ == "__main__":
    main()
