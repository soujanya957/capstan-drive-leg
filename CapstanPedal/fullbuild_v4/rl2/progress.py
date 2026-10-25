"""A training-progress film: the same scenario played by the policy at several points of one training run, each
clip labelled with that moment's statistics (steps, held-out success, falls, reward). Full CPU model, real sensors.

    MUJOCO_GL=egl python progress.py <name> [--n=6] [--secs=6]     -> ../video/v4_rl2_<name>_progress.mp4

Uses the snapshots train.py saves at each held-out evaluation (policies/snaps/<name>/) and logs/<name>.csv / _eval.jsonl.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import env as E
import play
import policy as P
import tasks as T

HERE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 30
SCENE = dict(terrain=("stairs_up", 3), walk=None, rear=3, handstand=3, balance=4)   # what each task's clips show


def stats(name):
    ev = [json.loads(l) for l in open(HERE / "logs" / f"{name}_eval.jsonl")]
    rows = list(csv.DictReader(open(HERE / "logs" / f"{name}.csv")))
    steps = np.array([float(r["steps"]) for r in rows])
    out = {}
    for e in ev:
        i = int(np.argmin(np.abs(steps - e["steps"])))
        lo = max(0, i - 5)
        out[int(e["steps"] // 1_000_000)] = dict(
            steps=e["steps"], success=e["success"],
            fall=float(np.mean([float(r["fall"]) for r in rows[lo:i + 1]])),
            reward=float(np.mean([float(r["reward"]) for r in rows[lo:i + 1]])), total=float(steps[-1]))
    return out


def scenario(task):
    bank = task.bank(400, 4242)
    kind, level = np.asarray(bank["kind"]), np.asarray(bank["level"])
    pick = SCENE[task.name]
    if task.name == "terrain":
        i = int(np.flatnonzero((kind == T.TERRAINS.index(pick[0])) & (level == pick[1]))[0])
    elif task.name == "walk":
        i = 0
    else:
        i = int(np.flatnonzero(level == pick)[0])
    cfg = play.row(bank, i)
    if task.name == "walk":
        cfg.update(push_t=np.int32(10 ** 6))
    if task.name in ("rear", "handstand"):
        cfg.update(t_on=np.int32(40))
    return cfg


def label(img, lines, frac, font_big, font):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, W, 112], fill=(10, 12, 16, 170))
    dr.text((28, 18), lines[0], fill=(240, 242, 245, 255), font=font_big)
    dr.text((28, 66), lines[1], fill=(200, 205, 212, 255), font=font)
    dr.rectangle([28, 100, W - 28, 104], fill=(80, 86, 96, 255))
    dr.rectangle([28, 100, 28 + int((W - 56) * frac), 104], fill=(255, 77, 94, 255))
    return np.asarray(im)


def main():
    name = sys.argv[1]
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[2:] if a.startswith("--"))
    n, secs = int(opts.get("n", 6)), float(opts.get("secs", 6))
    snaps = sorted((HERE / "policies" / "snaps" / name).glob("*M.npz"))
    st = stats(name)
    picks = [snaps[int(round(k))] for k in np.linspace(0, len(snaps) - 1, n)]
    task = E.make_task(str(np.load(picks[0])["task"]))
    ce = E.CpuEnv(task)
    cfg = scenario(task)
    r = mujoco.Renderer(ce.m, H, W)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    font_big, font = ImageFont.load_default(size=34), ImageFont.load_default(size=22)
    out = E.C.VIDEO / f"{E.C.TAG}_rl2_{name}_progress.mp4"
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    summary = []
    for k, path in enumerate(picks):
        m_steps = int(path.stem[:-1])
        s = st.get(m_steps) or st[min(st, key=lambda x: abs(x - m_steps))]
        pol = P.Policy(path)
        obs = ce.reset(cfg, seed=11)
        look = ce.d.qpos[0:3].copy()
        what = {"terrain": f"{SCENE['terrain'][0].replace('_', ' ')}, level {SCENE['terrain'][1]}", "walk": "walk forward, then turn",
                "rear": "stand up on the hind legs", "handstand": "stand on the front legs", "balance": "board on a ball"}[task.name]
        res, t, nxt = "", 0.0, 0.0
        for i in range(int(secs * 50)):
            if task.name == "walk":
                cmd = (0.7, 0.0, 0.0) if i < secs * 25 else (0.3, 0.0, 0.8)
                ce.aux = dict(ce.aux, cmd=jp.asarray(cmd, jp.float32), t=jp.int32(0))
            if not res:
                obs, rew, terms, done, ok, fallen, err = ce.step(pol(obs))
                if done:
                    res = "success" if ok else ("fell" if fallen else "")
            t += 1 / 50
            while t >= nxt:
                look += (ce.d.qpos[0:3] - look) * 0.2
                cam.lookat[:] = look - np.array([0.0, 0.0, 0.12])
                cam.distance, cam.azimuth, cam.elevation = 2.0 if task.name == "balance" else 1.7, 115.0, -14.0
                r.update_scene(ce.d, cam, opt)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                img = label(r.render(), [f"{name}: after {s['steps'] / 1e6:.0f} M steps   ({k + 1}/{n})   {res}",
                                         f"{what}   |   held-out success {s['success']:.0%}   falls in training {s['fall']:.0%}   "
                                         f"reward/step {s['reward']:+.3f}"], s["steps"] / s["total"], font_big, font)
                ff.stdin.write(img.tobytes())
                nxt += 1 / FPS
        summary.append(f"{m_steps}M {s['success']:.0%} {res or 'ok'}")
    ff.stdin.close()
    ff.wait()
    print(f"{out.name}: " + ", ".join(summary))


if __name__ == "__main__":
    main()
