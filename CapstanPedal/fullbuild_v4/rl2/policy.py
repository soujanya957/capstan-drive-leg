"""A trained policy as numpy only (no torch, no stable-baselines3): the .npz train.py writes (actor and observation normalisation), run deterministically.

    pol = Policy("policies/walk_sgs_s0.npz")
    action = pol(obs)            # obs from env.CpuEnv

"""
import numpy as np


class Policy:
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.var = z["obs_mean"], z["obs_var"]
        self.clip, self.eps = float(z["clip_obs"]), float(z["epsilon"])
        n = sorted({int(k.split("_")[-2]) for k in z.files if k.startswith("mlp_extractor_policy_net") and k.endswith("weight")})
        self.layers = [(z[f"mlp_extractor_policy_net_{i}_weight"], z[f"mlp_extractor_policy_net_{i}_bias"]) for i in n]
        self.out = (z["action_net_weight"], z["action_net_bias"])

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        for W, b in self.layers:
            x = W @ x + b
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))         # ELU
        W, b = self.out
        return np.clip(W @ x + b, -1.0, 1.0)


