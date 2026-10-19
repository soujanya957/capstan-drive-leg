"""Train a walk or jump policy for the v3 quadruped with PPO (stable-baselines3), 12 parallel environments.

Run:  python train.py walk [million_steps] [masses.json]        (RL_INIT=policies/walk_v0: fine-tune that policy)
      python train.py jump [million_steps] [masses.json]
Writes policies/<task>.zip, policies/<task>_vecnorm.pkl and a log (tensorboard under runs/).
"""
import json
import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

import envs

HERE = os.path.dirname(os.path.abspath(__file__))


class Progress(BaseCallback):
    def __init__(self, every=200_000):
        super().__init__()
        self.every, self.next = every, every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            ep = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else [0.0]
            ln = [e["l"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else [0.0]
            print(f"{self.num_timesteps / 1e6:6.2f} M steps  return {np.mean(ep):8.2f}  length {np.mean(ln):6.0f}", flush=True)
        return True


class JumpCurriculum(BaseCallback):
    """Raise the jump's target distance by 5 cm whenever 70 % of the last 300 standing jumps land cleanly at
    or beyond it (landed, upright, body never on the ground)."""

    def __init__(self, start=0.05, step=0.05):
        super().__init__()
        self.target, self.step, self.hist = start, step, []

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and not info.get("practice"):                      # standing jumps only
                ok = (info.get("landed") and not info.get("bottomed") and info.get("distance", 0.0) >= self.target
                      and info.get("TimeLimit.truncated", False))           # ran to the end: did not tip over
                self.hist.append(bool(ok))
        if len(self.hist) >= 300 and np.mean(self.hist[-300:]) >= 0.7:
            self.target += self.step
            self.hist = []
            self.training_env.env_method("set_target", self.target)
            print(f"   curriculum: target {self.target:.2f} m", flush=True)
        return True


def make(task, i, masses):
    def f():
        return (envs.WalkEnv if task == "walk" else envs.JumpEnv)(seed=1000 + i, masses=masses)
    return f


def main():
    task = sys.argv[1]
    steps = int(float(sys.argv[2]) * 1e6) if len(sys.argv) > 2 else (40_000_000 if task == "walk" else 15_000_000)
    mfile = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "masses.json")       # the CAD's masses by default
    masses = json.load(open(mfile)) if os.path.exists(mfile) else None
    torch.set_num_threads(2)
    venv = VecNormalize(VecMonitor(SubprocVecEnv([make(task, i, masses) for i in range(12)])), norm_reward=True, clip_obs=10.0)
    policy = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), activation_fn=torch.nn.ELU, log_std_init=-0.7)
    init = os.environ.get("RL_INIT")                                 # fine-tune an earlier policy (its zip, minus .zip)
    if init:
        venv = VecNormalize.load(init + "_vecnorm.pkl", venv.venv)
        venv.training = True
        model = PPO.load(init, env=venv, device="cpu", tensorboard_log=os.path.join(HERE, "runs"))
    else:
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=6144, n_epochs=5, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.003, max_grad_norm=1.0, policy_kwargs=policy, verbose=0, device="cpu",
                    tensorboard_log=os.path.join(HERE, "runs"))
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    ckpt = CheckpointCallback(save_freq=max(1, 5_000_000 // 12), save_path=os.path.join(HERE, "policies", "ckpt"),
                              name_prefix=task, save_vecnormalize=True)
    cbs = [Progress(), ckpt] + ([JumpCurriculum()] if task == "jump" and os.environ.get("JUMP_CURRICULUM") else [])
    model.learn(steps, callback=CallbackList(cbs), tb_log_name=task)
    os.makedirs(os.path.join(HERE, "policies"), exist_ok=True)
    model.save(os.path.join(HERE, "policies", task))
    venv.save(os.path.join(HERE, "policies", f"{task}_vecnorm.pkl"))
    print("saved", task, flush=True)


if __name__ == "__main__":
    main()
