"""Drive an rl2 policy (policies/<name>.npz) with the keyboard, real time, in the full v4 model with the hardware's
sensors and delays. Needs a display: run it on the Mac with a policy copied from cake (the .npz is ~1 MB).
Needs numpy, mujoco, glfw and jax (CPU) in the env.

  walk:              W / S forward / back, A / D side-step, Q / E turn, Shift run
  terrain:           W / S faster / slower down the course, 1-8 terrain type, - / = level, R new course
  rear, handstand:   Space up / down
  every task:        K shove   X reset   [ / ] time speed   P pause   C camera follow   Esc quit

Run (from fullbuild_v4/rl2):  python teleop.py <name>
"""
import sys

import glfw
import jax.numpy as jp
import numpy as np

import env as E
import play
import tasks as T
from viewer import Clock, Viewer                                     # ../mujoco (core.py puts it on the path)

WALK = np.array([0.8, 0.35, 1.0])
RAMP = np.array([2.0, 1.5, 3.0])


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "walk"
    pol, task = play.load(name)
    ce = E.CpuEnv(task)
    bank = task.bank(400, 777)
    kind_arr, level_arr = np.asarray(bank["kind"]), np.asarray(bank["level"])
    st = dict(kind=0, level=3, up=False, follow=True, v=0.6, i=0)

    def pick():
        hits = np.flatnonzero((kind_arr == st["kind"]) & (level_arr == st["level"])) if task.name == "terrain" else np.arange(len(kind_arr))
        st["i"] = int(hits[np.random.randint(len(hits))])
        cfg = play.row(bank, st["i"])
        cfg.update(push_t=np.int32(10 ** 6))                              # shoves come from K
        if task.name == "terrain":
            cfg.update(v_cmd=np.float32(st["v"]), goal=np.float32(100.0))
        if task.name in ("rear", "handstand"):
            cfg.update(t_on=np.int32(10 ** 6))
        return cfg

    st["obs"] = ce.reset(pick())

    def on_key(key, mods):
        if key == glfw.KEY_X or key == glfw.KEY_R:
            st["obs"] = ce.reset(pick())
        elif key == glfw.KEY_K:
            ce.d.qvel[1] += 0.8
        elif key == glfw.KEY_SPACE:
            st["up"] = not st["up"]
        elif glfw.KEY_1 <= key <= glfw.KEY_8 and task.name == "terrain":
            st["kind"] = key - glfw.KEY_1
            st["obs"] = ce.reset(pick())
        elif key in (glfw.KEY_MINUS, glfw.KEY_EQUAL) and task.name == "terrain":
            st["level"] = int(np.clip(st["level"] + (1 if key == glfw.KEY_EQUAL else -1), 0, 9))
            st["obs"] = ce.reset(pick())
        elif key == glfw.KEY_P:
            clock.paused = not clock.paused
        elif key == glfw.KEY_LEFT_BRACKET:
            clock.slower()
        elif key == glfw.KEY_RIGHT_BRACKET:
            clock.faster()
        elif key == glfw.KEY_C:
            st["follow"] = not st["follow"]

    clock = Clock(1.0)
    v = Viewer(ce.m, ce.d, f"Capstan v4: rl2 policy {name} ({task.name})", max_geom=3000, on_key=on_key)
    v.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    v.shadows = False
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.2, -18.0, 120.0
    look = ce.d.qpos[0:3].copy()
    cmd = np.zeros(3)
    owed, dt = 0.0, 1.0 / 50
    status = ""
    while v.open():
        owed = min(owed + clock.advance(), 0.1)
        while owed >= dt:
            if task.name == "walk":
                run = 1.5 if (v.held(glfw.KEY_LEFT_SHIFT) or v.held(glfw.KEY_RIGHT_SHIFT)) else 1.0
                want = np.array([(v.held(glfw.KEY_W) - 0.5 * v.held(glfw.KEY_S)) * WALK[0] * run,
                                 (v.held(glfw.KEY_A) - v.held(glfw.KEY_D)) * WALK[1],
                                 (v.held(glfw.KEY_Q) - v.held(glfw.KEY_E)) * WALK[2] * run])
                cmd += np.clip(want - cmd, -RAMP * dt, RAMP * dt)
                ce.aux = dict(ce.aux, cmd=jp.asarray(cmd, jp.float32), t=jp.int32(0))
            elif task.name == "terrain":
                st["v"] = float(np.clip(st["v"] + 0.5 * dt * (v.held(glfw.KEY_W) - v.held(glfw.KEY_S)), 0.0, 1.0))
                ce.cfg = dict(ce.cfg, v_cmd=jp.float32(st["v"]))
                ce.aux = dict(ce.aux, t=jp.int32(0))
            elif task.name in ("rear", "handstand"):
                ce.aux = dict(ce.aux, cmd=jp.array([1.0 if st["up"] else 0.0, 0.0, 0.0]), t=jp.int32(0))
            obs, r, terms, done, ok, fallen, err = ce.step(pol(st["obs"]))
            st["obs"] = obs
            owed -= dt
            if fallen:
                status = "fell: reset"
                st["obs"] = ce.reset(pick())
        if st["follow"]:
            look += (ce.d.qpos[0:3] - look) * 0.15
            v.cam.lookat[:] = look - np.array([0.0, 0.0, 0.15])
        c = np.asarray(ce.aux["cmd"])
        extra = (f"terrain {T.TERRAINS[st['kind']]} level {st['level']}   speed {st['v']:.2f} m/s" if task.name == "terrain"
                 else f"command {c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}")
        v.draw(text_tl="K shove   X reset   [ ] time   P pause   C follow   Esc quit\n" + {
            "walk": "W/S A/D Q/E drive, Shift run", "terrain": "W/S speed  1-8 terrain  -/= level  R new course",
            "rear": "Space: up / down", "handstand": "Space: up / down", "balance": ""}[task.name],
            text_tr=f"{name}\n{extra}\ntorso {ce.d.qpos[2] * 100:.0f} cm   {status}")
    v.close()


if __name__ == "__main__":
    main()
