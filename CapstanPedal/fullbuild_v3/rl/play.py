"""Run a trained policy in the physics model and film it on the full CAD model.

The policy drives quad_model.py (MuJoCo physics); every 1/30 s the torso pose and the joint angles are
recorded, then each CAD part is posed from them (as make_quad_film does for the scripted films): thigh's
CAD angle = Q0_thigh - hip, shin's = Q0_shin - (hip + knee) (a MuJoCo hinge about +Y turns the X-Z angle
the other way), ab/ad = the hinge angle (its sign flipped on the right pods, whose CAD is mirrored).

Run:  python play.py walk <quad_mesh_dir> [--at=t1,t2]     -> ../video/quadruped_rl_walk.mp4
      python play.py jump <quad_mesh_dir> [--name=jump_long]  -> ../video/quadruped_rl_jump.mp4 / _jump_long.mp4
      python play.py jump <quad_mesh_dir> --name=scripted     -> ../video/quadruped_jump_scripted.mp4 (scripted_jump.py)
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import envs                                                          # noqa: E402
import quad_model as qm                                              # noqa: E402
import make_quad_film as qf                                          # noqa: E402

mv = qf.mv
FPS = mv.FPS

WALK_SCRIPT = [(1.5, (0.0, 0.0, 0.0), "Standing"), (5.0, (0.5, 0.0, 0.0), "Trot forward, 0.5 m/s"),
               (4.0, (0.9, 0.0, 0.0), "Faster, 0.9 m/s"), (4.0, (0.4, 0.0, 0.6), "Turning, 0.6 rad/s"),
               (4.0, (0.0, 0.25, 0.0), "Side-step, 0.25 m/s"), (2.5, (0.0, 0.0, 0.0), "Stop")]


def load(task, name):
    masses = json.load(open(HERE / "masses.json")) if (HERE / "masses.json").exists() else None
    cls = envs.WalkEnv if task == "walk" else envs.JumpEnv
    venv = VecNormalize.load(str(HERE / "policies" / f"{name}_vecnorm.pkl"), DummyVecEnv([lambda: cls(seed=7, masses=masses)]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(str(HERE / "policies" / name), device="cpu")
    return venv, model


JUMP_CAPS = {"scripted": ("Scripted jump (no learning)", "Crouch, full-torque push along 60°, swing, soft landing: 20 of 20 land, ~0.4 m in the air"),
             "jump": ("Jump, landed", "A short jump it always lands: 20 of 20 in evaluation"),
             "jump_long": ("Long jump", "0.66-0.80 m in the air; it does not stick the landing yet (0 of 20)")}


def rollout_scripted():
    """The scripted jump (scripted_jump.py), recorded as rollout() does."""
    import scripted_jump as sj
    masses = json.load(open(HERE / "masses.json"))
    frames, nxt = [], [0.0]
    cap = JUMP_CAPS["scripted"][0]

    def rec(e, j):
        d = e.data
        while nxt[0] <= d.time:
            q = {p: (d.qpos[e.model.joint(f"{p}_hip").qposadr[0]], d.qpos[e.model.joint(f"{p}_knee").qposadr[0]],
                     d.qpos[e.model.joint(f"{p}_abad").qposadr[0]]) for p in qm.PODS}
            frames.append((nxt[0], d.qpos[0:3].copy(), d.qpos[3:7].copy(), q, cap))
            nxt[0] += 1.0 / FPS
    r = sj.run(sj.params(), 507, masses, record=rec, T=3.5)
    return frames, {k: v for k, v in r.items() if k != "env"}


def rollout(task, name):
    """Recorded frames: (t, torso pos m, torso quat, {pod: (hip, knee, abad)}, caption)."""
    if name == "scripted":
        return rollout_scripted()
    venv, model = load(task, name)
    env = venv.envs[0]
    obs = venv.reset()
    frames, t, next_frame = [], 0.0, 0.0
    ctl = envs.DT * envs.DECIM
    script = WALK_SCRIPT if task == "walk" else [(3.0, None, JUMP_CAPS[name][0])]
    total = sum(s[0] for s in script)
    info_last = {}
    while t < total:
        acc, cap = 0.0, script[-1][2]
        for dur, cmd, c in script:
            if t < acc + dur:
                cap = c
                if task == "walk":
                    env.cmd = np.array(cmd, float)
                break
            acc += dur
        action, _ = model.predict(obs, deterministic=True)
        if task == "walk":
            env.t = 1                                                  # keep the script's command (no resampling)
        obs, r, done, info = venv.step(action)
        info_last = info[0]
        t += ctl
        d = env.data
        while next_frame <= t:
            q = {p: (d.qpos[env.model.joint(f"{p}_hip").qposadr[0]], d.qpos[env.model.joint(f"{p}_knee").qposadr[0]],
                     d.qpos[env.model.joint(f"{p}_abad").qposadr[0]]) for p in qm.PODS}
            frames.append((next_frame, d.qpos[0:3].copy(), d.qpos[3:7].copy(), q, cap))
            next_frame += 1.0 / FPS
        if done[0]:
            print("episode ended at", round(t, 2), flush=True)
            break
    if task == "jump":                                                 # a second of standing first, then the hold at the end
        f0 = frames[0]
        frames = [(i / FPS, f0[1], f0[2], f0[3], "Standing") for i in range(FPS)] + \
                 [(f[0] + 1.0, f[1], f[2], f[3], f[4]) for f in frames]
    return frames, info_last


def render(task, qdir, at=None, name=None):
    name = name or task
    frames, info = rollout(task, name)
    print(f"{task}: {len(frames)} frames; {info}", flush=True)
    q = json.loads((qdir / "quad.json").read_text())
    meta = q["meta"]
    for v in meta.values():                                            # mujoco/meshes stores paths relative to itself
        if not v["file"].startswith("/"):
            v["file"] = str(qdir / v["file"])
    model = qf.build_model(meta)
    verts, _ = qf.mesh_data(model, meta)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, mv.H, mv.W, max_geom=20000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 1, 1, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    bid = {k: model.body(k).id for k in meta}
    mocap = {k: model.body_mocapid[i] for k, i in bid.items()}
    geom_of = {k: model.body_geomadr[i] for k, i in bid.items()}
    model.geom_group[geom_of["Q_Build_Stand"]] = 5
    pods = {k: (qf.classify(meta[k]), np.diag(meta[k]["mirror"]), np.array(meta[k]["shift"])) for k in meta if "pod" in meta[k]}
    F = mv.Film(meta, verts)
    F.theme = "dark"                                                   # light ink on the dark sky
    fl = model.geom("floor").id
    fl0 = model.geom_pos[fl].copy()
    tile = 2.0 * 2 * model.geom_size[fl][0] / 40.0                     # two checker tiles (texrepeat 40 over the plane)
    caps, last = [], None
    for f in frames:
        if f[4] != last:
            caps.append([f[0], f[0], "", f[4], JUMP_CAPS[name][1] if f[4] in (JUMP_CAPS.get(name, ("",))[0],)
                         else "Scripted controller in a MuJoCo physics model of this robot" if name == "scripted"
                         else "Policy trained with PPO in a MuJoCo physics model of this robot"])
            last = f[4]
        caps[-1][1] = f[0]
    F.captions = [(a + 0.1, b - 0.1, s, h, d) for a, b, s, h, d in caps]
    dur = frames[-1][0]
    F.cards.append((dur + 0.1, dur + 2.8, "Capstan quadruped v3", ("Scripted jump" if name == "scripted" else "RL policy") +
                    " in physics simulation · v9.1 legs · not yet on hardware"))
    F.duration = dur + 3.0
    stem = "quadruped_jump_scripted" if name == "scripted" else f"quadruped_rl_{name}"
    out = qdir.parent / "preview" / f"{stem}.mp4" if at else HERE.parent / "video" / f"{stem}.mp4"
    out.parent.mkdir(exist_ok=True)
    procs = []
    if not at:
        for path in (out, out.with_name(out.stem + "_notext.mp4")):
            procs.append(subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                           "-s", f"{mv.W}x{mv.H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                                           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE))
    quat = np.zeros(4)
    idx = [int(x * FPS) for x in at] if at else list(range(len(frames))) + [len(frames) - 1] * int(3 * FPS)
    for n, i in enumerate(idx):
        t_film = i / FPS if not at else at[n]
        tt, pos, qt_, qs, cap = frames[min(i, len(frames) - 1)]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, qt_)
        R = R.reshape(3, 3)
        pb = pos * 1000.0
        states = {}
        for p in qm.PODS:
            hip, knee, ab = qs[p]
            qt = qm.Q0[0] - math.degrees(hip)
            qsh = qm.Q0[1] - math.degrees(hip + knee)
            states[p] = qf.leg_state(qt, qsh, ab if p[0] == "L" else -ab)
        for k in meta:
            if k in pods:
                (kind, rides), D, sh = pods[k]
                Rk, pk = qf.pod_pose(kind, rides, D, sh, states[meta[k]["pod"]])
            else:
                Rk, pk = np.eye(3), np.zeros(3)
            data.mocap_pos[mocap[k]] = R @ pk + pb
            mujoco.mju_mat2Quat(quat, (R @ Rk).flatten())
            data.mocap_quat[mocap[k]] = quat
        model.geom_pos[fl][:2] = fl0[:2] + np.round(pb[:2] / tile) * tile   # the floor follows in whole tiles
        mujoco.mj_forward(model, data)
        cam.lookat[:] = (pb[0], pb[1], pb[2] - 140.0)
        cam.distance, cam.azimuth, cam.elevation = (1900.0, -60.0 + 25.0 * math.sin(t_film / 6.0), -14.0)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        scn = renderer.scene
        for p in qm.PODS:
            st = states[p]
            kc, kd = f"{p}__Motor_Plate", f"{p}__Abad_Drum"
            D, sh = pods[kc][1], pods[kc][2]
            for stage, motor, y_mid in (("A", mv.MA, mv.A_MID), ("B", mv.MB, mv.B_MID)):
                pts = mv.rope_path(motor, y_mid, st["th"][stage])
                pts = (st["Rab"] @ (pts - mv.AB_C).T).T + mv.AB_C
                mv.draw_rope(scn, (pts @ D.T + sh) @ R.T + pb, 1.0, 1.3, False)
            if kd in meta:
                D, sh = pods[kd][1], pods[kd][2]
                pts = mv.ab_map(mv.rope_path(mv.MB, mv.AB_MID, -st["ab"], ab=True))
                mv.draw_rope(scn, (pts @ D.T + sh) @ R.T + pb, 1.0, 1.3, False)
        img = Image.fromarray(renderer.render())
        raw = img
        img = mv.overlay(img, F, t_film, (np.array(cam.lookat), cam.distance, cam.azimuth, cam.elevation),
                         f"Capstan quadruped v3  ·  {'scripted jump' if name == 'scripted' else 'RL ' + name.replace('_', ' ')}", mv.THEMES["dark"])
        if at:
            img.save(out.parent / f"{out.stem}_{int(t_film * FPS):05d}.jpg", quality=85)
        else:
            procs[0].stdin.write(img.tobytes())
            procs[1].stdin.write(raw.convert("RGB").tobytes())
    for pr in procs:
        pr.stdin.close()
        pr.wait()
    renderer.close()
    print("written", out, flush=True)


if __name__ == "__main__":
    task, qdir = sys.argv[1], Path(sys.argv[2]).resolve()
    at = [float(x) for a in sys.argv if a.startswith("--at=") for x in a[5:].split(",")]
    name = next((a[7:] for a in sys.argv if a.startswith("--name=")), task)     # e.g. --name=jump_long
    render(task, qdir, at or None, name)
