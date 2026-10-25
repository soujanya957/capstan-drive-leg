"""Drive a trained walking policy (policies/<name>.npz) with the keyboard, real time, in the full v4 model
(quad.py). Needs only the capstan-sim env (numpy, mujoco, glfw): the policy runs in numpy (policy.py).

  W / S    forward / back            A / D    side-step left / right
  Q / E    turn left / right         Shift    run (forward and turn x1.6)
  K        shove the robot sideways  X        reset
  [ / ]    slower / faster time      P  pause     C  camera follow     G  shadows     V  collision shapes
  mouse    left drag orbit, right drag pan, scroll zoom                                   Esc  quit

Run (from fullbuild_v4/rl):  python teleop.py [name]          (default walk_gpu)
"""
import math
import sys
import time
from pathlib import Path

import glfw
import numpy as np

import policy as P
import quad as Q
from viewer import Clock, Viewer                                     # ../mujoco (quad.py puts it on the path)

HERE = Path(__file__).resolve().parent
WALK = np.array([0.8, 0.35, 1.0])      # forward m/s, sideways m/s, yaw rad/s at a key press
RUN = 1.5                              # Shift: forward and yaw x this (1.2 m/s, the edge of the training range)
BACK = 0.5
RAMP = np.array([2.0, 1.5, 3.0])       # command slew, per s


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "walk_gpu"
    pol = P.Policy(HERE / "policies" / f"{name}.npz")
    q = Q.Quad()
    m, d = q.m, q.d
    q.reset()
    clock = Clock(1.0)
    st = dict(follow=True, shapes=False)

    def on_key(key, mods):
        if key == glfw.KEY_X:
            q.reset()
        elif key == glfw.KEY_K:
            d.qvel[1] += 0.8
        elif key == glfw.KEY_P:
            clock.paused = not clock.paused
        elif key == glfw.KEY_LEFT_BRACKET:
            clock.slower()
        elif key == glfw.KEY_RIGHT_BRACKET:
            clock.faster()
        elif key == glfw.KEY_C:
            st["follow"] = not st["follow"]
        elif key == glfw.KEY_G:
            v.shadows = not v.shadows
        elif key == glfw.KEY_V:
            st["shapes"] = not st["shapes"]
            v.opt.geomgroup[3] = st["shapes"]

    v = Viewer(m, d, f"Capstan quadruped v4: RL policy ({name})", max_geom=2000, on_key=on_key)
    v.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    v.shadows = False
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.0, -18.0, 120.0
    look = d.qpos[0:3].copy()
    cmd = np.zeros(3)
    owed, dt = 0.0, Q.DT * Q.DECIM
    frames, t_fps, fps = 0, time.perf_counter(), 0.0
    while v.open():
        run = RUN if (v.held(glfw.KEY_LEFT_SHIFT) or v.held(glfw.KEY_RIGHT_SHIFT)) else 1.0
        fwd = v.held(glfw.KEY_W) - BACK * v.held(glfw.KEY_S)
        want = np.array([fwd * WALK[0] * run, (v.held(glfw.KEY_A) - v.held(glfw.KEY_D)) * WALK[1],
                         (v.held(glfw.KEY_Q) - v.held(glfw.KEY_E)) * WALK[2] * run])
        owed = min(owed + clock.advance(), 0.1)
        while owed >= dt:                                               # one policy step = 10 physics steps
            cmd += np.clip(want - cmd, -RAMP * dt, RAMP * dt)
            q.cmd = cmd.copy()
            q.last_a, _, _ = q.step(pol(q.obs()), Q.T_GAIT)
            owed -= dt
            if d.qpos[2] < 0.15 or q.base()[3][2] > -0.3:                # fallen over: start again
                q.reset()
        if st["follow"]:
            look += (d.qpos[0:3] - look) * 0.15
            v.cam.lookat[:] = look - np.array([0.0, 0.0, 0.15])
        frames += 1
        now = time.perf_counter()
        if now - t_fps > 0.5:
            fps, frames, t_fps = frames / (now - t_fps), 0, now
        _, vb, w, grav = q.base()
        tilt = math.degrees(math.acos(min(1.0, -grav[2])))
        hud_l = ("W/S forward-back   A/D side-step   Q/E turn   Shift run\n"
                 "K shove   X reset   [ ] time speed   P pause\nC follow   G shadows   V collision shapes   Esc quit")
        hud_r = (f"{'PAUSED' if clock.paused else f'time x{clock.speed:g}'}   {fps:.0f} fps   policy {name}\n"
                 f"command  {cmd[0]:+.2f} m/s  {cmd[1]:+.2f} m/s  {cmd[2]:+.2f} rad/s\n"
                 f"actual   {-vb[0]:+.2f} m/s  {vb[1]:+.2f} m/s  {w[2]:+.2f} rad/s\n"
                 f"torso {d.qpos[2] * 100:.0f} cm   tilt {tilt:.1f} deg")
        v.draw(text_tl=hud_l, text_tr=hud_r)
    v.close()


if __name__ == "__main__":
    main()
