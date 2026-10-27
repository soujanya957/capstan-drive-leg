"""A trained policy as numpy only (no torch, no stable-baselines3): the .npz train.py writes (actor and observation normalisation), run deterministically.

    pol = Policy("policies/walk_sgs_s0.npz")
    action = pol(obs)            # obs from env.CpuEnv

A distilled student (distill.py) reads the last 50 frames (env.task_for gives the task that observes them): a history
encoder estimates the teacher's latent, and the policy network reads the newest 3 frames and that estimate. With
look=1 the observation starts with the forward range sensor's readings, which the encoder also reads. A residual
(train_sac.py --base, RL Token) is a frozen base policy plus a correction read from its last hidden layer. A teacher
(train.py --teacher=1) needs the privileged state the robot cannot sense, so it is refused here.
"""
import numpy as np


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class Policy:
    def __init__(self, path):
        z = np.load(path)
        self.kind = str(z["kind"]) if "kind" in z.files else "mlp"
        if self.kind == "teacher":
            raise ValueError(f"{path} is a teacher: it needs the privileged state; distill.py makes its deployable student")
        self.mean, self.var = z["obs_mean"], z["obs_var"]
        self.clip, self.eps = float(z["clip_obs"]), float(z["epsilon"])
        n = sorted({int(k.split("_")[-2]) for k in z.files if k.startswith("mlp_extractor_policy_net") and k.endswith("weight")})
        self.layers = [(z[f"mlp_extractor_policy_net_{i}_weight"], z[f"mlp_extractor_policy_net_{i}_bias"]) for i in n]
        self.out = (z["action_net_weight"], z["action_net_bias"])
        self.squash = "squash" in z.files and bool(z["squash"])           # FastSAC (train_sac.py): tanh output
        if self.kind == "residual":
            k = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("res_") and k.endswith("weight")})
            self.res = [(z[f"res_{i}_weight"], z[f"res_{i}_bias"]) for i in k]
            self.res_scale = float(z["res_scale"])
        if self.kind == "student":
            self.n_hist, self.n_frame = int(z["n_hist"]), int(z["n_frame"])
            self.f_mean, self.f_var = z["frame_mean"], z["frame_var"]
            self.proj = (z["proj_weight"], z["proj_bias"])
            self.look = (z["look_mean"], z["look_var"]) if "look" in z.files and int(z["look"]) else None
            k = sorted({int(k.split("_")[1]) for k in z.files if k.startswith("enc_")})
            self.enc = [(z[f"enc_{i}_weight"], z[f"enc_{i}_bias"]) for i in k]

    def norm(self, x, mean, var):
        return np.clip((x - mean) / np.sqrt(var + self.eps), -self.clip, self.clip)

    def __call__(self, obs):
        x = self.norm(obs[-len(self.mean):], self.mean, self.var)
        if self.kind == "student":
            nl = 0 if self.look is None else len(self.look[0])
            fr = self.norm(obs[nl:].reshape(self.n_hist, self.n_frame), self.f_mean, self.f_var)
            h = elu(fr @ self.proj[0].T + self.proj[1]).ravel()
            if nl:
                h = np.concatenate([h, self.norm(obs[:nl], *self.look)])
            for W, b in self.enc[:-1]:
                h = elu(W @ h + b)
            W, b = self.enc[-1]
            x = np.concatenate([x, W @ h + b])
        for W, b in self.layers:
            x = elu(W @ x + b)
        W, b = self.out
        if self.kind == "residual":
            a_b = np.tanh(W @ x + b)
            h = np.concatenate([x, a_b])
            for W, b in self.res[:-1]:
                h = elu(W @ h + b)
            W, b = self.res[-1]
            return np.clip(a_b + self.res_scale * np.tanh(W @ h + b), -1.0, 1.0)
        return np.tanh(W @ x + b) if self.squash else np.clip(W @ x + b, -1.0, 1.0)
