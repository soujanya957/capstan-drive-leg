"""Drive the quadruped in MuJoCo from the keyboard: the physics model (rl/quad_model.py) under the trained walk
policy, with the scripted jump on Space. --cad draws the full CAD robot on the physics instead.

  W / S    forward / back           (up to 0.9 m/s forward, 0.2 back)
  A / D    side-step left / right   (0.3 m/s)
  Q / E    turn left / right        (0.8 rad/s)
  R / F    body up / down           (hold; about 6 cm of range)
  Space    jump (scripted: crouch, push, land)
  [ / ]    slower / faster time     P  pause      X  reset      C  camera follow on/off
  mouse    left drag orbit, right drag pan, scroll zoom          Esc  quit

Run (from fullbuild_v3/mujoco):  python teleop_dog.py [--cad] [--meshes=<dir>]
"""
import math
import sys

import glfw
import numpy as np

import dog as dg
import quad_model as qm
from viewer import Clock, Viewer

LOOKS = '''  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.36 0.42" rgb2="0.05 0.06 0.08" width="64" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.2 0.23" rgb2="0.12 0.13 0.15" mark="edge" markrgb="0.3 0.33 0.36" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="80 80" reflectance="0.04"/>
  </asset>
  <statistic extent="2" center="0 0 0.3"/>
  <visual><quality shadowsize="4096"/><map znear="0.01"/></visual>
'''
_mjcf = qm.mjcf


def mjcf_looks(*a, **k):                                               # a checker floor and a sky; the physics is unchanged
    s = _mjcf(*a, **k)
    s = s.replace("<worldbody>", LOOKS + "  <worldbody>", 1)
    s = s.replace('rgba="0.15 0.15 0.15 0.4"', 'rgba="0.30 0.33 0.37 1"').replace('rgba="0.2 0.2 0.2 0.25"', 'rgba="0.16 0.42 0.45 1"')
    return s.replace('rgba="0.3 0.32 0.35 1" contype="1" conaffinity="3"', 'material="grid" contype="1" conaffinity="3"')


qm.mjcf = mjcf_looks


def main():
    cad_on = "--cad" in sys.argv
    meshes = next((a[9:] for a in sys.argv if a.startswith("--meshes=")), None)
    d = dg.Dog()
    clock = Clock(1.0)
    state = dict(follow=True)
    if cad_on:
        import cad as cd
        skin = cd.Cad(meshes or cd.MESHES)
        model, data, scale = skin.model, skin.data, 1000.0
    else:
        model, data, scale = d.model, d.data, 1.0

    def on_key(key, mods):
        if key == glfw.KEY_SPACE:
            d.jump()
        elif key == glfw.KEY_X:
            d.reset()
        elif key == glfw.KEY_P:
            clock.paused = not clock.paused
        elif key == glfw.KEY_LEFT_BRACKET:
            clock.slower()
        elif key == glfw.KEY_RIGHT_BRACKET:
            clock.faster()
        elif key == glfw.KEY_C:
            state["follow"] = not state["follow"]

    v = Viewer(model, data, "Capstan quadruped v3: teleop", on_key=on_key)
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.9 * scale, -18.0, -60.0
    if not cad_on:
        v.opt.geomgroup[:] = 1
    owed = 0.0
    ctl = dg.envs.DT * dg.envs.DECIM
    cmd = np.zeros(3)
    while v.open():
        dt_wall = 1.0 / 60.0
        want = np.array([(v.held(glfw.KEY_W) - v.held(glfw.KEY_S) * 0.22) * dg.CMD_MAX[0],
                         (v.held(glfw.KEY_D) - v.held(glfw.KEY_A)) * dg.CMD_MAX[1],
                         (v.held(glfw.KEY_Q) - v.held(glfw.KEY_E)) * dg.CMD_MAX[2]])
        if v.held(glfw.KEY_R) or v.held(glfw.KEY_F):
            d.height = float(np.clip(d.height + (0.03 if v.held(glfw.KEY_R) else -0.03) * dt_wall, *dg.H_RANGE))
        owed += clock.advance()
        while owed >= ctl:
            cmd += np.clip(want - cmd, -1.5 * ctl, 1.5 * ctl)            # commands ramp at 1.5 per s
            d.cmd = cmd
            d.step()
            owed -= ctl
            if d.fallen():
                d.reset()
        if cad_on:
            skin.pose_from_physics(d.model, d.data)
        if state["follow"]:
            v.cam.lookat[:] = d.data.qpos[0:3] * scale - np.array([0, 0, 0.12 * scale])
        fwd, side, yaw = d.speed()
        hud = (f"W/S forward  A/D side  Q/E turn  R/F height  Space jump\n[ ] speed  P pause  X reset  C follow  Esc quit"
               f"\tcommand  {cmd[0]:+.2f} m/s  {cmd[1]:+.2f} m/s  {cmd[2]:+.2f} rad/s\n"
               f"actual   {fwd:+.2f} m/s  {side:+.2f} m/s  {yaw:+.2f} rad/s\nheight {d.height * 100:+.1f} cm   {d.mode}")
        speed = "PAUSED" if clock.paused else f"time x{clock.speed:g}"
        v.draw(add_geoms=skin.ropes if cad_on else None, text_tl=hud.split("\t")[0],
               text_tr=speed + "\n" + hud.split("\t")[1])
    v.close()


if __name__ == "__main__":
    main()
