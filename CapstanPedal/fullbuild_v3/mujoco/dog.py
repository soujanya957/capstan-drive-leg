"""The quadruped in physics, driven from commands: the trained walk policy (rl/policies/walk) for walking and
turning, a body-height offset on top of it, and the scripted jump (rl/scripted_jump.py). No window here:
teleop_dog.py and watch_assembly.py drive it.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rl"))
import envs                                                          # noqa: E402
import quad_model as qm                                              # noqa: E402
import scripted_jump as sj                                           # noqa: E402

H_RANGE = (-0.05, 0.01)          # m of body height the walk policy copes with (dog.py's self-test: +0.02 falls)
CMD_MAX = np.array([0.9, 0.3, 0.8])     # forward m/s, sideways m/s, yaw rad/s (the walk policy's training range)


class Dog:
    def __init__(self, seed=7):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        masses = json.load(open(ROOT / "rl" / "masses.json"))
        pol = ROOT / "rl" / "policies"
        self.venv = VecNormalize.load(str(pol / "walk_vecnorm.pkl"), DummyVecEnv([lambda: envs.WalkEnv(seed=seed, masses=masses)]))
        self.venv.training = False
        self.policy = PPO.load(str(pol / "walk"), device="cpu")
        self.env = self.venv.envs[0]
        self.model, self.data = self.env.model, self.env.data
        self.walk_apply = self.env.apply
        self.cmd = np.zeros(3)
        self.height = 0.0
        self.jumper = None
        self.jump_params = sj.params()
        self.reset()

    def reset(self):
        self.env.reset(seed=None)
        self.env.cmd = np.zeros(3)
        self.jumper = None
        self.env.apply = self._walk_apply

    def _offset(self):
        """Output offsets that lower (h < 0) or raise the body by h at the rest pose: a leg-length change."""
        r0 = 2 * qm.LEG_L * math.sin(math.radians(60.0))
        B0, A0 = sj.ik(0.0, -r0)
        B1, A1 = sj.ik(0.0, -(r0 + self.height))
        off = np.zeros(12)
        off[1::3], off[2::3] = B1 - B0, A1 - A0
        return off

    def _walk_apply(self, target):
        return self.walk_apply(target + self._offset())

    def jump(self):
        if self.jumper is None:
            self.jumper = sj.Jumper(self.env, self.jump_params)
            self.env.apply = self.jumper.apply

    @property
    def mode(self):
        return "jump: " + self.jumper.phase if self.jumper else "walk"

    def step(self):
        """One 50 Hz control step (10 physics steps)."""
        e = self.env
        if self.jumper is not None:
            e.sim(np.zeros(12))
            j = self.jumper
            if j.phase == "land" and j.t - j.t_phase > j.p["LAND_GIVE"] + 0.9:
                self.jumper = None                                   # standing again: back to the walk policy
                e.apply = self._walk_apply
                e.last_a = np.zeros(12)
            return
        e.cmd = np.clip(self.cmd, -CMD_MAX, CMD_MAX)
        obs = self.venv.normalize_obs(e.obs()[None])
        a, _ = self.policy.predict(obs, deterministic=True)
        a = np.clip(a[0], -1, 1)
        e.sim(a)
        e.last_a = a

    def fallen(self):
        R, v, w, grav = self.env.base()
        return grav[2] > -0.5

    def speed(self):
        R, v, w, grav = self.env.base()
        return -v[0], v[1], w[2]


if __name__ == "__main__":                                             # self-test: heights and commands, no window
    for h in (-0.06, -0.05, -0.03, 0.0, 0.02, 0.03):
        d = Dog(seed=3)
        d.height = h
        falls, zs, vs = 0, [], []
        for cmd in ((0, 0, 0), (0.5, 0, 0), (0.9, 0, 0), (0.3, 0, 0.6), (0, 0.25, 0), (-0.2, 0, 0)):
            d.cmd = np.array(cmd, float)
            for k in range(200):
                d.step()
                if k > 100:
                    zs.append(d.data.qpos[2])
                    vs.append(d.speed()[0] - cmd[0])
                if d.fallen():
                    falls += 1
                    d.reset()
                    break
        print(f"height {h:+.2f}: falls {falls}, body z {np.mean(zs):.3f} (stand {qm.STAND_Z:.3f}), "
              f"forward-speed error rms {np.sqrt(np.mean(np.square(vs))):.2f} m/s", flush=True)
    d = Dog(seed=4)
    for k in range(50):
        d.step()
    d.jump()
    x0 = d.data.qpos[0]
    for k in range(200):
        d.step()
    print("jump from walk mode:", d.mode, "fallen" if d.fallen() else "upright", f"moved {-(d.data.qpos[0] - x0):.2f} m")
    for k in range(150):
        d.cmd = np.array([0.5, 0, 0])
        d.step()
    print("walk after the jump:", "fallen" if d.fallen() else "upright", f"{d.speed()[0]:.2f} m/s")
