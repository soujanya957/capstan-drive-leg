"""The full CAD quadruped in a MuJoCo window, playing the build film's timeline (src/make_quad_film.py: every
part and fastener in assembly order, then stand up, trot, jump) at a speed you choose, with a free camera.

  Space    pause / play               [ / ]   slower / faster (0.05x to 2x)
  N / B    next / previous step       Left / Right   back / forward 5 s (Shift: 1 s)
  , / .    one frame back / forward   Home    restart
  C        film camera / free camera  (dragging the mouse switches to the free camera)
  mouse    left drag orbit, right drag pan, scroll zoom (Shift: finer)       Esc  quit

Run (from fullbuild_v4/mujoco):  python watch_assembly.py [--speed=0.5] [--at=<s>] [--meshes=<dir>]
"""
import sys

import glfw
import numpy as np

import cad as cd
from viewer import Clock, Viewer

qf = cd.qf
ASCII = str.maketrans({"×": "x", "·": "-", "°": " deg", "–": "-", "—": "-", "→": "->", "≤": "<=", "≥": ">=", "Ø": "dia ", "±": "+-"})


def plain(s):
    """MuJoCo's overlay font is ASCII only."""
    return "".join(c for c in s.translate(ASCII) if ord(c) < 128)


def main():
    meshes = next((a[9:] for a in sys.argv if a.startswith("--meshes=")), cd.MESHES)
    speed = float(next((a[8:] for a in sys.argv if a.startswith("--speed=")), 0.5))
    t = float(next((a[5:] for a in sys.argv if a.startswith("--at=")), 0.0))
    print("loading the CAD and authoring the build timeline...", flush=True)
    c = cd.Cad(meshes)
    F, M, t_built = qf.setup(c.meta, c.verts, c.tris, c.layout)
    dur = M.t_end + 3.0
    poser = qf.Poser(F, M, c.meta, c.model, c.data)
    steps = sorted({cap[0] for cap in F.captions})
    clock = Clock(min(Clock.SPEEDS, key=lambda s: abs(s - speed)))
    st = dict(t=t, film_cam=True)

    def seek(x):
        st["t"] = float(np.clip(x, 0.0, dur))

    def on_key(key, mods):
        shift = mods & glfw.MOD_SHIFT
        if key == glfw.KEY_SPACE:
            clock.paused = not clock.paused
        elif key == glfw.KEY_LEFT_BRACKET:
            clock.slower()
        elif key == glfw.KEY_RIGHT_BRACKET:
            clock.faster()
        elif key == glfw.KEY_RIGHT:
            seek(st["t"] + (1.0 if shift else 5.0))
        elif key == glfw.KEY_LEFT:
            seek(st["t"] - (1.0 if shift else 5.0))
        elif key == glfw.KEY_PERIOD:
            seek(st["t"] + 1.0 / qf.mv.FPS)
        elif key == glfw.KEY_COMMA:
            seek(st["t"] - 1.0 / qf.mv.FPS)
        elif key == glfw.KEY_N:
            seek(next((s for s in steps if s > st["t"] + 0.05), dur))
        elif key == glfw.KEY_B:
            seek(max([s for s in steps if s < st["t"] - 0.5] or [0.0]))
        elif key == glfw.KEY_HOME:
            seek(0.0)
        elif key == glfw.KEY_C:
            st["film_cam"] = not st["film_cam"]

    v = Viewer(c.model, c.data, "Capstan quadruped v3: assembly", on_key=on_key)
    v.opt.geomgroup[:] = [1, 1, 1, 1, 1, 0]
    btn = v._button

    def button(win, b, action, mods):                                   # grabbing the view frees the camera
        if action == glfw.PRESS:
            st["film_cam"] = False
        btn(win, b, action, mods)
    glfw.set_mouse_button_callback(v.win, button)

    while v.open():
        st["t"] = min(st["t"] + clock.advance(), dur)
        t = st["t"]
        pb, states = poser(t)
        if st["film_cam"]:
            lookat, dist, az, el = qf.camera(F, M, t)
            v.cam.lookat[:], v.cam.distance, v.cam.azimuth, v.cam.elevation = lookat, dist, az, el
        cap = next(((s, h, d) for t0, t1, s, h, d in reversed(F.captions) if t0 - 0.2 <= t <= t1 + 0.2), None)
        top = (f"Step {cap[0]}: " if cap and cap[0] else "") + (cap[1] if cap else "")
        detail = cap[2] if cap else ""
        state = "PAUSED" if clock.paused else f"x{clock.speed:g}"
        v.draw(add_geoms=lambda scn: poser.ropes(scn, t, pb, states),
               text_tl=plain(f"{top}\n{detail}") if top else "",
               text_tr=f"{t:6.1f} / {dur:.0f} s   {state}\n{'film camera' if st['film_cam'] else 'free camera (C: film)'}",
               text_bl="Space pause  [ ] speed  N/B step  Left/Right seek  , . frame  C camera  Esc quit")
        if t >= dur:
            clock.paused = True
    v.close()


if __name__ == "__main__":
    main()
