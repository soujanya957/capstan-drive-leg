"""Drive the quadruped in MuJoCo: the URDF model (model.py, every CAD part on its real link) walking under the
model-based trot controller (gait.py), in real time.

  W / S    forward / back            A / D    side-step left / right
  Q / E    turn left / right         R / F    body up / down (hold)
  Space    jump                      Shift    run (forward and turn x1.6)
  [ / ]    slower / faster time      P  pause     X  reset     C  camera follow
  G        shadows on/off            V  collision shapes
  mouse    left drag orbit, right drag pan, scroll zoom                           Esc  quit

Run (from fullbuild_v3/mujoco):  python teleop.py
"""
import math
import time

import glfw
import mujoco
import numpy as np

import gait as G
import model as M
from viewer import Clock, Viewer

WALK = np.array([0.6, 0.3, 0.8])       # forward m/s, sideways m/s, yaw rad/s at a key press
RUN = 1.6                              # Shift: forward and yaw x this
BACK = 0.5                             # backward is this fraction of forward
RAMP = np.array([1.5, 1.2, 3.0])       # command slew, per s


def main():
    m = M.build()
    d = mujoco.MjData(m)
    c = G.Controller(m, d)
    c.reset_pose()
    clock = Clock(1.0)
    st = dict(follow=True, shapes=False)

    def on_key(key, mods):
        if key == glfw.KEY_SPACE:
            c.jump()
        elif key == glfw.KEY_X:
            c.reset_pose()
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

    v = Viewer(m, d, "Capstan quadruped v3: teleop", max_geom=2000, on_key=on_key)
    v.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    v.shadows = False                                                   # G: shadows (about half the frame rate)
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.0, -18.0, 120.0
    look = d.qpos[0:3].copy()
    cmd = np.zeros(3)
    owed, dt = 0.0, m.opt.timestep
    frames, t_fps, fps = 0, time.perf_counter(), 0.0
    while v.open():
        run = RUN if (v.held(glfw.KEY_LEFT_SHIFT) or v.held(glfw.KEY_RIGHT_SHIFT)) else 1.0
        fwd = v.held(glfw.KEY_W) - BACK * v.held(glfw.KEY_S)
        want = np.array([fwd * WALK[0] * run,
                         (v.held(glfw.KEY_A) - v.held(glfw.KEY_D)) * WALK[1],
                         (v.held(glfw.KEY_Q) - v.held(glfw.KEY_E)) * WALK[2] * run])
        sim_dt = clock.advance()
        if v.held(glfw.KEY_R) or v.held(glfw.KEY_F):
            c.h = float(np.clip(c.h + (0.08 if v.held(glfw.KEY_R) else -0.08) * sim_dt, *G.H_RANGE))
        owed = min(owed + sim_dt, 0.1)                                  # never more than 0.1 s behind
        while owed >= dt:
            cmd += np.clip(want - cmd, -RAMP * dt, RAMP * dt)
            c.cmd = cmd
            c.step(dt)
            mujoco.mj_step(m, d)
            owed -= dt
            if d.qpos[2] < 0.12 or c.body()[0][2, 2] < 0.3:                # fallen over: stand it back up
                c.reset_pose()
        if st["follow"]:
            look += (d.qpos[0:3] - look) * 0.15
            v.cam.lookat[:] = look - np.array([0.0, 0.0, 0.15])
        frames += 1
        now = time.perf_counter()
        if now - t_fps > 0.5:
            fps, frames, t_fps = frames / (now - t_fps), 0, now
        R, Rrp, vb, w, roll, pitch = c.body()
        mode = ("jump: " + c.jump_state["phase"]) if c.jump_state else ("trot" if c.walking else "stand")
        hud_l = ("W/S forward-back   A/D side-step   Q/E turn   R/F height\n"
                 "Space jump   Shift run   [ ] time speed   P pause   X reset\nC follow   G shadows   V collision shapes   Esc quit")
        hud_r = (f"{'PAUSED' if clock.paused else f'time x{clock.speed:g}'}   {fps:.0f} fps\n"
                 f"command  {cmd[0]:+.2f} m/s  {cmd[1]:+.2f} m/s  {cmd[2]:+.2f} rad/s\n"
                 f"actual   {-vb[0]:+.2f} m/s  {vb[1]:+.2f} m/s  {w[2]:+.2f} rad/s\n"
                 f"hip height {c.h * 100:.0f} cm   roll {math.degrees(roll):+.1f}  pitch {math.degrees(pitch):+.1f}   {mode}")
        v.draw(text_tl=hud_l, text_tr=hud_r)
    v.close()


if __name__ == "__main__":
    main()
