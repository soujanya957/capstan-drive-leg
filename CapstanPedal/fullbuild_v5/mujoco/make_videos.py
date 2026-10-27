"""Short films of the v5 quadruped in simulation: the URDF model (model.py) driven by the model-based
controller (gait.py), filmed from several angles. No learning anywhere.

Run (from fullbuild_v5/mujoco):  python make_videos.py [--notext] [clip ...]     -> ../video/v5_<clip>[_notext].mp4
Clips: trot, turn, spin, sidestep, height, gaits, jump, orbit, camera (default: all). "camera" insets what the head's
ZED Mini would see (the model's "zed" camera: colour, and depth 0.2 to 3 m) while it walks past some blocks.
"""
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gait as G
import model as M

W, H, FPS = 1280, 720, 30
OUT = Path(__file__).resolve().parent.parent / "video"
FONT = FONT_S = None
for _f in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):   # macOS, Linux (cake)
    try:
        FONT, FONT_S = ImageFont.truetype(_f, 30), ImageFont.truetype(_f, 20)
        break
    except OSError:
        pass
if FONT is None:
    FONT = FONT_S = ImageFont.load_default()

# each clip: (title, subtitle, slow-motion factor, [(seconds, (fwd, side, yaw), gait or None, action)], camera(t, pos, yaw))
#   action: None, "jump", or ("height", h)
CLIPS = {
    "trot": ("Trot", "0.6 m/s, diagonal pairs", 1.0,
             [(1.0, (0, 0, 0), "trot", None), (5.0, (0.6, 0, 0), None, None), (1.0, (0, 0, 0), None, None)],
             lambda t, p, yaw: (p, 1.08, 90.0, -8.0)),
    "turn": ("Turning", "0.4 m/s forward, 0.8 rad/s", 1.0,
             [(0.8, (0, 0, 0), "trot", None), (6.0, (0.4, 0, 0.8), None, None), (0.8, (0, 0, 0), None, None)],
             lambda t, p, yaw: (p, 1.22, 135.0, -22.0)),
    "spin": ("Spin in place", "1.0 rad/s, no forward speed", 1.0,
             [(0.8, (0, 0, 0), "trot", None), (6.5, (0, 0, 1.0), None, None), (0.8, (0, 0, 0), None, None)],
             lambda t, p, yaw: (p, 1.15, 60.0, -35.0)),
    "sidestep": ("Side-step", "0.3 m/s left, then right", 1.0,
                 [(0.8, (0, 0, 0), "trot", None), (2.5, (0, 0.3, 0), None, None), (2.5, (0, -0.3, 0), None, None),
                  (0.8, (0, 0, 0), None, None)],
                 lambda t, p, yaw: (p, 1.08, 180.0, -12.0)),
    "height": ("Body height", "R / F: hip 26 to 38.5 cm above the feet", 1.0,
               [(0.8, (0, 0, 0), "trot", ("height", 0.385)), (1.6, (0, 0, 0), None, ("height", 0.26)),
                (1.6, (0, 0, 0), None, ("height", 0.385)), (1.0, (0, 0, 0), None, ("height", 0.336))],
               lambda t, p, yaw: (p, 0.94, 150.0, -10.0)),
    "gaits": ("Gaits", "walk, pace, bound, pronk at 0.4 m/s", 1.0,
              [(0.6, (0, 0, 0), "walk", None), (3.2, (0.4, 0, 0), None, None), (0.6, (0, 0, 0), "pace", None),
               (3.2, (0.4, 0, 0), None, None), (0.6, (0, 0, 0), "bound", None), (3.2, (0.4, 0, 0), None, None),
               (0.6, (0, 0, 0), "pronk", None), (3.2, (0.4, 0, 0), None, None), (0.8, (0, 0, 0), None, None)],
              lambda t, p, yaw: (p, 1.15, 110.0, -6.0)),
    "jump": ("Jump", "crouch, push, swing, land  (half speed)", 0.5,
             [(0.6, (0, 0, 0), "trot", None), (3.2, (0, 0, 0), None, "jump")],
             lambda t, p, yaw: (p, 1.37, 90.0, -5.0)),
    "orbit": ("Capstan quadruped v5", "9:1 capstan legs, a head with a Jetson and a ZED Mini, trotting at 0.5 m/s", 1.0,
              [(0.6, (0, 0, 0), "trot", None), (7.4, (0.5, 0, 0), None, None)],
              lambda t, p, yaw: (p, 1.01, 40.0 * t, -18.0 + 8.0 * math.sin(t))),
    "camera": ("What the head sees", "ZED Mini, 30° down: colour and depth, trotting and turning", 1.0,
               [(0.8, (0, 0, 0), "trot", None), (4.5, (0.4, 0, 0), None, None), (4.0, (0.3, 0, 0.5), None, None),
                (0.8, (0, 0, 0), None, None)],
               lambda t, p, yaw: (p, 1.25, yaw - 35.0, -16.0)),
}
PIP = {"camera"}                                                 # clips with the head camera's view inset
PIP_W, PIP_H = 384, 216


def scenery(spec):
    """Blocks and posts beside the route for the camera to see (drawn only: they never collide)."""
    rng = np.random.default_rng(5)
    cols = [(0.85, 0.33, 0.25), (0.95, 0.70, 0.20), (0.25, 0.60, 0.85), (0.35, 0.75, 0.45), (0.80, 0.80, 0.82)]
    for i in range(26):
        x = -0.7 - 0.22 * i + rng.uniform(-0.05, 0.05)
        y = (1 if i % 2 else -1) * rng.uniform(0.38, 0.75) + (0.12 * max(0, i - 10) if i > 10 else 0.0)
        hx, hy, hz = rng.uniform(0.03, 0.08), rng.uniform(0.03, 0.08), rng.uniform(0.03, 0.22)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], pos=[x, y, hz], rgba=[*cols[i % 5], 1.0],
                                contype=0, conaffinity=0, group=0)
    return spec


def depth_rgb(dep, near=0.2, far=3.0):
    """Depth (m) as colour: near warm, far cool, beyond `far` dark."""
    u = np.clip((dep - near) / (far - near), 0.0, 1.0)
    r = np.clip(1.6 - 2.2 * u, 0, 1)
    g = np.clip(1.2 - 2.4 * np.abs(u - 0.45), 0, 1)
    b = np.clip(0.3 + 1.4 * u, 0, 1) * (u < 0.999)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def inset(img, rgb, dep, notext):
    """The head camera's colour and depth views, top left."""
    im = img.copy()
    d = ImageDraw.Draw(im)
    for k, a in enumerate((rgb, dep)):
        x0, y0 = 24, 24 + k * (PIP_H + 14)
        d.rectangle((x0 - 3, y0 - 3, x0 + PIP_W + 2, y0 + PIP_H + 2), fill=(230, 232, 236))
        im.paste(Image.fromarray(a), (x0, y0))
        if not notext:
            d.text((x0 + 8, y0 + 6), ("ZED Mini (left)" if k == 0 else "depth 0.2 to 3 m"), font=FONT_S, fill=(255, 255, 255))
    return im


def caption(img, title, sub, t, dur):
    a = max(0.0, min(1.0, t / 0.4, (dur - t) / 0.4))
    if a <= 0:
        return img
    im = img.convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w = max(d.textlength(title, font=FONT), d.textlength(sub, font=FONT_S)) + 48
    d.rounded_rectangle((32, H - 128, 32 + w, H - 32), 14, fill=(20, 24, 32, int(200 * a)))
    d.text((56, H - 118), title, font=FONT, fill=(255, 255, 255, int(255 * a)))
    d.text((56, H - 74), sub, font=FONT_S, fill=(205, 212, 222, int(255 * a)))
    d.text((W - 360, 24), "simulation  ·  model-based control", font=FONT_S, fill=(220, 225, 232, int(180 * a)))
    return Image.alpha_composite(im, layer).convert("RGB")


def film(name, notext=False):
    title, sub, slow, script, camf = CLIPS[name]
    m = scenery(M.build_spec()).compile() if name in PIP else M.build()
    d = mujoco.MjData(m)
    c = G.Controller(m, d)
    c.reset_pose()
    r = mujoco.Renderer(m, H, W)
    rp = mujoco.Renderer(m, PIP_H, PIP_W) if name in PIP else None
    opt = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    OUT.mkdir(exist_ok=True)
    path = OUT / f"v5_{name}{'_notext' if notext else ''}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    dt = m.opt.timestep
    frame_dt = slow / FPS                                        # sim seconds per video frame
    dur = sum(s[0] for s in script)
    t_sim, next_frame, n = 0.0, 0.0, 0
    look = d.qpos[0:3].copy()
    cmd = np.zeros(3)
    seg_end = 0.0
    for seconds, target, gait, action in script:
        seg_end += seconds
        if gait:
            c.request_gait(gait)
        if action == "jump":
            c.jump()
        while t_sim < seg_end - 1e-9:
            cmd += np.clip(np.array(target, float) - cmd, -1.5 * dt, 1.5 * dt)
            c.cmd = cmd
            if isinstance(action, tuple) and action[0] == "height":
                c.h += float(np.clip(action[1] - c.h, -0.08 * dt, 0.08 * dt))
            c.step(dt)
            mujoco.mj_step(m, d)
            t_sim += dt
            if t_sim >= next_frame:
                look += (d.qpos[0:3] - look) * (0.25 if n else 1.0)
                R = d.xmat[1].reshape(3, 3)
                yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                p, dist, az, el = camf(t_sim, look, yaw)
                cam.lookat[:] = p - np.array([0.0, 0.0, 0.12])
                cam.distance, cam.azimuth, cam.elevation = dist, az, el
                r.update_scene(d, cam, opt)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                img = Image.fromarray(r.render())
                if rp is not None:
                    rp.update_scene(d, camera="zed")
                    rgb = rp.render()
                    rp.enable_depth_rendering()
                    dep = depth_rgb(rp.render())
                    rp.disable_depth_rendering()
                    img = inset(img, rgb, dep, notext)
                if not notext:
                    img = caption(img, title, sub, t_sim, dur)
                ff.stdin.write(img.tobytes())
                next_frame += frame_dt
                n += 1
    ff.stdin.close()
    ff.wait()
    r.close()
    if rp is not None:
        rp.close()
    up = d.xmat[1][8]
    print(f"{path.name}: {n / FPS:.1f} s, ended upright {up:.2f}", flush=True)
    return up > 0.9


if __name__ == "__main__":
    args = sys.argv[1:]
    notext = "--notext" in args
    names = [a for a in args if a != "--notext"] or list(CLIPS)
    ok = [film(nm, notext) for nm in names]
    sys.exit(0 if all(ok) else 1)
