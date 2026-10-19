"""A small GLFW window for a MuJoCo model, with the keyboard left free for the scripts (MuJoCo's own viewer
binds W, A, S, D, Q, E, R, F to display toggles).

Mouse: left drag orbits, right drag pans, scroll zooms (shift: finer). The scripts poll held keys with
held() and get one-shot presses through on_key.
"""
import time

import glfw
import mujoco


class Viewer:
    def __init__(self, model, data, title, max_geom=20000, on_key=None):
        if not glfw.init():
            raise RuntimeError("could not start GLFW (no display?)")
        glfw.window_hint(glfw.SAMPLES, 4)
        self.win = glfw.create_window(1400, 860, title, None, None)
        glfw.make_context_current(self.win)
        glfw.swap_interval(1)
        self.model, self.data = model, data
        self.scn = mujoco.MjvScene(model, maxgeom=max_geom)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.on_key = on_key
        self.shadows = True                                         # off: about twice the frame rate
        self._btn, self._last = None, (0.0, 0.0)
        glfw.set_key_callback(self.win, self._key)
        glfw.set_mouse_button_callback(self.win, self._button)
        glfw.set_cursor_pos_callback(self.win, self._move)
        glfw.set_scroll_callback(self.win, self._scroll)

    # --- input -------------------------------------------------------------------------------------
    def held(self, key):
        return glfw.get_key(self.win, key) == glfw.PRESS

    def _key(self, win, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif self.on_key:
                self.on_key(key, mods)

    def _button(self, win, button, action, mods):
        self._btn = button if action == glfw.PRESS else None
        self._last = glfw.get_cursor_pos(win)

    def _move(self, win, x, y):
        if self._btn is None:
            return
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        w, h = glfw.get_window_size(win)
        if self._btn == glfw.MOUSE_BUTTON_LEFT:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._btn == glfw.MOUSE_BUTTON_RIGHT:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        self._cam_move(act, dx / h, dy / h)

    def _scroll(self, win, dx, dy):
        fine = self.held(glfw.KEY_LEFT_SHIFT)
        self._cam_move(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * dy * (0.3 if fine else 1.0))

    def _cam_move(self, act, rx, ry):
        try:                                                        # MuJoCo >= 3.16: no scene argument
            mujoco.mjv_moveCamera(self.model, int(act), rx, ry, self.cam)
        except TypeError:
            mujoco.mjv_moveCamera(self.model, int(act), rx, ry, self.scn, self.cam)

    # --- drawing -----------------------------------------------------------------------------------
    def open(self):
        return not glfw.window_should_close(self.win)

    def draw(self, add_geoms=None, text_tl="", text_tr="", text_bl=""):
        """Render one frame; add_geoms(scn) may append to the scene (ropes, markers)."""
        w, h = glfw.get_framebuffer_size(self.win)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows
        if add_geoms:
            add_geoms(self.scn)
        mujoco.mjr_render(vp, self.scn, self.ctx)
        for pos, txt in ((mujoco.mjtGridPos.mjGRID_TOPLEFT, text_tl), (mujoco.mjtGridPos.mjGRID_TOPRIGHT, text_tr),
                         (mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, text_bl)):
            if txt:
                left, _, right = txt.partition("\t")
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, pos, vp, left, right, self.ctx)
        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def close(self):
        glfw.terminate()


class Clock:
    """Simulated time against wall time, at an adjustable speed (and paused)."""
    SPEEDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)

    def __init__(self, speed=1.0):
        self.i = self.SPEEDS.index(speed)
        self.paused = False
        self.wall = time.perf_counter()

    @property
    def speed(self):
        return self.SPEEDS[self.i]

    def faster(self):
        self.i = min(self.i + 1, len(self.SPEEDS) - 1)

    def slower(self):
        self.i = max(self.i - 1, 0)

    def advance(self):
        """Simulated seconds to advance since the last call."""
        now = time.perf_counter()
        dt, self.wall = min(now - self.wall, 0.1), now
        return 0.0 if self.paused else dt * self.speed
