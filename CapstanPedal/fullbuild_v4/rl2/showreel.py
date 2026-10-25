"""One film of the learned skills: robots walking, rearing, doing handstands, balancing on a board and crossing
terrain courses side by side in one world, a camera that orbits the whole yard, then flies to each skill in turn and
circles it. Full CPU model, real sensors.

    MUJOCO_GL=egl JAX_PLATFORMS=cpu python showreel.py [--fmt=landscape|story|post] [--secs=70]
        landscape  1280 x 720, five robots, one per skill          -> ../video/v4_rl2_showreel.mp4
        story      1080 x 1920 (Instagram story / reel), 16 robots, no text  -> ../video/v4_rl2_showreel_story.mp4
        post       1080 x 1350 (Instagram feed, 4:5), 16 robots, no text     -> ../video/v4_rl2_showreel_post.mp4

Each robot runs in its own world (its own CpuEnv, policy and episodes); the worlds are recorded, then drawn together,
each shifted to its spot in the yard. A pose robot that finishes an episode starts the next one where it stands.
Robots of the same skill play different scenarios. A walker's spot is the centre of its circles.
"""
import subprocess
import sys
from pathlib import Path

import jax.numpy as jp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import env as E
import play
import tasks as T

HERE = Path(__file__).resolve().parent
FPS = 30
POLICY = dict(walk="walk2_best", rear="rear_best", handstand="handstand2_best", balance="balance2_best",
              terrain="terrain_uniform_s0_best")
LABEL = dict(walk=("Walk on joystick commands", "walk2_best · 100 % held-out"),
             rear=("Rear up on the hind legs", "rear_best · 100 % held-out"),
             handstand=("Handstand on the front legs", "handstand2_best · 82 % held-out"),
             balance=("Balance on a board on a ball (easy level)", "balance2_best · 100 % at level 0, 39.5 % overall"),
             terrain=("Blind terrain course: stairs, slopes, obstacles", "terrain_uniform_s0_best · 75.5 % held-out"))
TOUR = ["walk", "rear", "handstand", "balance", "terrain"]

# kind, (x, y) in the yard
YARD = [("walk", (-3.0, 0.8)), ("rear", (-1.4, 2.8)), ("handstand", (1.4, 2.8)), ("balance", (2.8, 0.4)),
        ("terrain", (4.5, -4.2))]
CROWD = [("walk", (-3.4, 2.6)), ("walk", (0.0, 5.0)), ("walk", (3.6, 2.8)), ("walk", (-1.0, -2.0)),
         ("rear", (-1.0, 2.2)), ("rear", (2.4, 0.4)), ("rear", (-4.2, -0.4)), ("rear", (1.6, 4.6)),
         ("handstand", (1.2, 2.2)), ("handstand", (-2.4, 0.4)), ("handstand", (4.6, -0.2)),
         ("balance", (0.2, -0.2)), ("balance", (-3.6, -2.4)), ("balance", (2.6, -2.2)),
         ("terrain", (4.5, -4.4)), ("terrain", (4.5, -5.6))]
FORMATS = dict(landscape=(1280, 720, YARD, ""), story=(1080, 1920, CROWD, "_story"), post=(1080, 1350, CROWD, "_post"))


def walk_cmd(t, v):
    """Closed circles, left then right, so the robot stays near home; each walker starts at a different point."""
    t = (t + 4.7 * v) % 19.0
    if t < 1.0:
        return (0.0, 0.0, 0.0)
    turn = 0.6 if (t < 10.0) == (v % 2 == 0) else -0.6
    return (0.45, 0.0, turn)


def configs(task, kind, v):
    bank = task.bank(400, 4242)
    level = np.asarray(bank["level"])
    if kind == "terrain":
        k = np.asarray(bank["kind"])
        names = ("stairs_up", "obstacles", "slope", "stairs_down", "rough", "mixed")
        names = names[v % 2:] + names[:v % 2] if v else names
        picks = [int(np.flatnonzero((k == T.TERRAINS.index(n)) & (level == 4))[v % 3]) for n in names]
    elif kind == "balance":
        picks = [int(i) for i in np.flatnonzero(level == 0)[12 * v:12 * v + 12]]
    elif kind == "walk":
        picks = [v]
    else:
        picks = [int(i) for i in np.flatnonzero(level == 5)[12 * v:12 * v + 12]]
    out = []
    for i in picks:
        cfg = play.row(bank, i)
        if kind == "walk":
            cfg.update(push_t=np.int32(10 ** 6), fric=np.float32(1.0), mass_add=np.float32(0.0), strength=np.float32(1.0))
        if kind in ("rear", "handstand"):
            cfg.update(t_on=np.int32(40 + 30 * v))                           # not all at once
        out.append(cfg)
    return out


def record(kind, v, n):
    """n control steps (50 Hz) of one robot: qpos and mocap poses per step, the episode results."""
    name = POLICY[kind]
    pol, task = play.load(name)
    ce = E.CpuEnv(task)
    cfgs, c = configs(task, kind, v), 0
    obs = ce.reset(cfgs[0], seed=1 + 17 * v)
    qs, mp, mq, log, cmd = [], [], [], [], np.zeros(3)
    for i in range(n):
        if kind == "walk":
            cmd += np.clip(np.array(walk_cmd(i / 50, v)) - cmd, -0.04, 0.04)
            ce.aux = dict(ce.aux, cmd=jp.asarray(cmd, jp.float32), t=jp.int32(0))
        obs, r, terms, done, ok, fallen, err = ce.step(pol(obs))
        qs.append(ce.d.qpos.copy())
        mp.append(ce.d.mocap_pos.copy())
        mq.append(ce.d.mocap_quat.copy())
        if done:
            log.append("success" if ok else ("fell" if fallen else "time-out"))
            for _ in range(25 if kind != "terrain" else 10):           # hold the end pose a moment
                qs.append(qs[-1]); mp.append(mp[-1]); mq.append(mq[-1])
            c = (c + 1) % len(cfgs)
            obs = ce.reset(cfgs[c], seed=100 + c + 17 * v)
    print(f"{kind:<10} #{v} {name:<24} episodes: {', '.join(log)}", flush=True)
    return ce, np.array(qs[:n]), np.array(mp[:n]), np.array(mq[:n])


def free_qadr(m):
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def main():
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
    fmt = opts.get("fmt", "landscape")
    W, H, layout, suffix = FORMATS[fmt]
    tall = H > W
    secs = float(opts.get("secs", 70 if fmt == "landscape" else 45))
    n = int(secs * 50)
    rec, offs, first_of = [], [], {}
    count = {}
    for kind, (x, y) in layout:
        v = count[kind] = count.get(kind, -1) + 1
        ce, qs, mps, mqs = record(kind, v, n)
        off = np.array([x, y, 0.0])
        if kind == "walk":                                              # centre the circles on the spot
            off[:2] -= qs[:, 0:2].mean(0)
        first_of.setdefault(kind, len(rec))
        rec.append((kind, ce, qs, mps, mqs))
        offs.append(off)

    # the first world is drawn with update_scene (sky, lights, floor); the others add only their own geoms
    for j, (kind, ce, *_) in enumerate(rec):
        ce.m.stat.extent = 8.0                                          # shadows over the whole yard
        ce.m.vis.global_.offwidth, ce.m.vis.global_.offheight = max(W, 1920), max(H, 1920)
        if j:
            ce.m.geom_rgba[ce.m.geom("floor").id, 3] = 0.0
    r = mujoco.Renderer(rec[0][1].m, H, W)
    cam, opt, pert = mujoco.MjvCamera(), mujoco.MjvOption(), mujoco.MjvPerturb()
    font_big, font = ImageFont.load_default(size=36), ImageFont.load_default(size=22)
    out = E.C.VIDEO / f"{E.C.TAG}_rl2_showreel{suffix}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    # shot list: wide orbit, one shot per skill, wide orbit again
    wide_t = 8.0 if fmt == "landscape" else 6.0
    shot = (secs - 2 * wide_t) / len(TOUR)
    centre = np.array([0.4, -0.4, 0.25])
    wide = 9.5 if fmt == "landscape" else (11.0 if tall else 9.5)
    near = 1.0 if fmt == "landscape" else (1.45 if tall else 1.2)       # a narrow frame needs the camera further back

    def target(t, i):
        """(lookat, distance, elevation, label kind) of the ideal camera at time t."""
        if t < wide_t or t >= secs - wide_t:
            return centre, wide, -30.0, None, -1
        kind = TOUR[min(int((t - wide_t) // shot), len(TOUR) - 1)]
        j = first_of[kind]
        p = rec[j][2][i][0:3] + offs[j] - np.array([0.0, 0.0, 0.12])
        return p, near * (2.6 if kind in ("balance", "terrain") else 2.1), -16.0, kind, j

    look, dist, elev = centre.copy(), wide, -30.0
    kind_of = [k for k, *_ in rec]
    for f in range(int(secs * FPS)):
        t = f / FPS
        i = min(int(t * 50), n - 1)
        lt, ld, le, kind, subject = target(t, i)
        a = 0.06                                                        # camera eases towards the shot's ideal
        look += (lt - look) * a
        dist += (ld - dist) * a
        elev += (le - elev) * a
        cam.lookat[:], cam.distance, cam.elevation = look, dist, elev
        cam.azimuth = 135.0 + 14.0 * t                                  # always circling
        az, el = np.radians(cam.azimuth), np.radians(elev)
        eye = look - dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        for j, (_, ce, qs, mps, mqs) in enumerate(rec):
            m, d = ce.m, ce.d
            d.qpos[:] = qs[i]
            for adr in free_qadr(m):
                d.qpos[adr:adr + 2] += offs[j][:2]
            if m.nmocap:
                d.mocap_pos[:] = mps[i] + offs[j]
                d.mocap_quat[:] = mqs[i]
            if j != subject and kind_of[j] != "terrain" and np.linalg.norm(qs[i][0:2] + offs[j][:2] - eye[:2]) < 1.6:
                if j:
                    continue                                            # a bystander next to the lens: leave it out
                d.qpos[0:3] = [0.0, 0.0, -50.0]                         # (world 0 also draws the floor and sky)
            mujoco.mj_kinematics(m, d)
            if j == 0:
                r.update_scene(d, cam, opt)
            else:
                mujoco.mjv_addGeoms(m, d, opt, pert, mujoco.mjtCatBit.mjCAT_ALL, r.scene)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        img = r.render()
        if fmt == "landscape":
            img = Image.fromarray(img)
            dr = ImageDraw.Draw(img, "RGBA")
            title, sub = LABEL[kind] if kind else ("Capstan v4 · five skills, five separate policies",
                                                   "each trained in simulation with the robot's real sensors, noise and delays")
            tk = (t - wide_t) % shot if kind else (t if t < wide_t else t - (secs - wide_t))
            alpha = int(255 * min(1.0, tk / 0.8))                       # fade the caption in at each cut
            dr.rectangle([0, H - 112, W, H], fill=(10, 12, 16, int(170 * alpha / 255)))
            dr.text((28, H - 98), title, fill=(240, 242, 245, alpha), font=font_big)
            dr.text((28, H - 48), sub, fill=(200, 205, 212, alpha), font=font)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    print(out)


if __name__ == "__main__":
    main()
