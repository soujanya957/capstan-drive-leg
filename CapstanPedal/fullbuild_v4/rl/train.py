"""Train the v4 walking policy with PPO (stable-baselines3) on parallel CPU environments (envs.WalkEnv).

Run (from fullbuild_v4/rl):  python train.py [million_steps] [--envs=N] [--name=walk]
      RL_INIT=policies/walk python train.py 50     (fine-tune an earlier policy)
Writes policies/<name>.zip, policies/<name>_vecnorm.pkl, policies/<name>.npz (numpy-only, for play.py and
teleop.py), checkpoints every 10 M steps, and tensorboard logs under runs/.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per env process: 18 processes x 20 BLAS threads crawl

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

import envs

HERE = os.path.dirname(os.path.abspath(__file__))
POL = os.path.join(HERE, "policies")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"           # the update; the envs run on the CPU


class Progress(BaseCallback):
    def __init__(self, every=500_000):
        super().__init__()
        self.every, self.next = every, every
        self.e_lin, self.e_yaw = [], []

    def _on_step(self):
        for info in self.locals["infos"]:
            self.e_lin.append(info["e_lin"])
            self.e_yaw.append(info["e_yaw"])
        if self.num_timesteps >= self.next:
            self.next += self.every
            buf = self.model.ep_info_buffer
            ep = np.mean([e["r"] for e in buf]) if buf else 0.0
            ln = np.mean([e["l"] for e in buf]) if buf else 0.0
            vl, vy = np.sqrt(np.mean(self.e_lin)), np.sqrt(np.mean(self.e_yaw))
            self.logger.record("walk/vel_rms", vl)
            self.logger.record("walk/yaw_rms", vy)
            print(f"{self.num_timesteps / 1e6:6.1f} M  return {ep:7.2f}  length {ln:5.0f}  "
                  f"vel err {vl:.3f} m/s  yaw err {vy:.3f} rad/s", flush=True)
            self.e_lin, self.e_yaw = [], []
        return True


def export(model, venv, path):
    """The actor's weights and the observation normalisation, for a numpy forward pass (policy.py)."""
    sd = model.policy.state_dict()
    out = {k.replace(".", "_"): v.cpu().numpy() for k, v in sd.items()
           if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")}
    np.savez(path, obs_mean=venv.obs_rms.mean, obs_var=venv.obs_rms.var, clip_obs=venv.clip_obs,
             epsilon=venv.epsilon, **out)


def make(i):
    def f():
        return envs.WalkEnv(seed=1000 + i)
    return f


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    steps = int(float(args[0]) * 1e6) if args else 200_000_000
    n_env = int(opts.get("envs", max(1, (os.cpu_count() or 4) - 2)))
    name = opts.get("name", "walk")
    torch.set_num_threads(2)
    venv = VecNormalize(VecMonitor(SubprocVecEnv([make(i) for i in range(n_env)])), norm_reward=True, clip_obs=10.0)
    policy = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), activation_fn=torch.nn.ELU, log_std_init=-1.0)
    init = os.environ.get("RL_INIT")
    if init:
        venv = VecNormalize.load(init + "_vecnorm.pkl", venv.venv)
        venv.training = True
        model = PPO.load(init, env=venv, device=DEVICE, tensorboard_log=os.path.join(HERE, "runs"))
    else:
        model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=n_env * 1024 // 4, n_epochs=5, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.003, max_grad_norm=1.0, target_kl=0.02,
                    policy_kwargs=policy, verbose=0, device=DEVICE, tensorboard_log=os.path.join(HERE, "runs"))
    os.makedirs(POL, exist_ok=True)
    ckpt = CheckpointCallback(save_freq=max(1, 10_000_000 // n_env), save_path=os.path.join(POL, "ckpt"),
                              name_prefix=name, save_vecnormalize=True)
    print(f"{n_env} envs, {steps / 1e6:.0f} M steps -> policies/{name}", flush=True)
    model.learn(steps, callback=CallbackList([Progress(), ckpt]), tb_log_name=name)
    model.save(os.path.join(POL, name))
    venv.save(os.path.join(POL, f"{name}_vecnorm.pkl"))
    export(model, venv, os.path.join(POL, f"{name}.npz"))
    print("saved", name, flush=True)


if __name__ == "__main__":
    main()
