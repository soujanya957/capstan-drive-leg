"""Train the v4 walking policy on the GPU: PPO in JAX over thousands of MuJoCo Warp worlds (mjx_walk.py), with
Success-Guided Sampling of the task configurations (Zhang et al. 2026, "A Balanced Data Diet", arXiv 2610.12465).

SGS: a fixed bank of N configurations (commands, friction, mass, strength, gains, shove; mjx_walk.sample_bank),
each with the outcomes of its last H episodes. Every reset draws the next configuration from a softmax (temperature
T) over log w_i, w_i = (p_i + eps)^(k t) (1 - p_i + eps)^(k (1 - t)): most weight on configurations solved about a
fraction t of the time, little on those already mastered or still hopeless, never zero. p_i starts at one
pseudo-outcome of t, so unseen configurations are tried early. --sampler=uniform gives the baseline.

Also: an asymmetric critic (it sees the configuration and the foot contacts; the actor does not), value
bootstrapping at time-outs, a KL-adaptive learning rate.

Run (from fullbuild_v4/rl, conda env capstan-gpu):
    python train_gpu.py [--steps=300] [--envs=8192] [--sampler=sgs] [--bank=base|hard] [--name=walk_gpu] [--init=policies/<name>.pkl]
Writes policies/<name>.npz (for policy.py: play.py, teleop.py), policies/<name>.pkl (to resume) and
logs/<name>.csv, every --save iterations and at the end.
"""
import csv
import os
import pickle
import sys
import time

import jax
import jax.numpy as jp
import numpy as np
import optax

import mjx_walk as W
import quad as Q

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = dict(steps=300.0, envs=8192, cfgs=16384, sampler="sgs", bank="base", name="walk_gpu", init="", seed=0, save=50,
           unroll=24, epochs=5, minibatches=4, gamma=0.99, lam=0.95, clip=0.2, ent=0.004, vf=1.0, lr=1e-3,
           kl=0.01, sgs_t=0.66, sgs_k=5.0, sgs_temp=2.0, sgs_h=16)
ACTOR, CRITIC = (256, 256, 128), (512, 256, 128)


def options():
    o = dict(OPT)
    for a in sys.argv[1:]:
        k, v = a.lstrip("-").split("=", 1)
        o[k] = type(OPT[k])(v)
    return o


# --- networks (plain JAX: lists of (W (out, in), b)) ---------------------------------------------------------
def mlp_init(key, sizes, last_scale):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        s = last_scale if i == len(sizes) - 2 else 1.0
        ps.append((jax.random.normal(k, (b, a)) * jp.sqrt(2.0 / a) * s, jp.zeros(b)))
    return ps


def mlp(ps, x):
    for Wt, b in ps[:-1]:
        x = jax.nn.elu(x @ Wt.T + b)
    Wt, b = ps[-1]
    return x @ Wt.T + b


def normalise(norm, x):
    return jp.clip((x - norm["mean"]) / jp.sqrt(norm["var"] + 1e-8), -10.0, 10.0)


def norm_update(norm, x):
    """Merge a batch (n, d) into the running mean / variance."""
    n, mean, var = x.shape[0], x.mean(0), x.var(0)
    tot = norm["count"] + n
    delta = mean - norm["mean"]
    m2 = norm["var"] * norm["count"] + var * n + delta ** 2 * norm["count"] * n / tot
    return dict(mean=norm["mean"] + delta * n / tot, var=m2 / tot, count=tot)


# --- SGS --------------------------------------------------------------------------------------------------
def sgs_logits(sgs, o):
    n = jp.minimum(sgs["count"], o["sgs_h"]).astype(jp.float32)
    p = (sgs["hist"].sum(1) + o["sgs_t"]) / (n + 1.0)
    eps, k, t = 1e-8, o["sgs_k"], o["sgs_t"]
    w = (p + eps) ** (k * t) * (1.0 - p + eps) ** (k * (1.0 - t))
    return jp.log(w + eps)[:-1], p[:-1]                                # the last row is a dummy for unfinished envs


def sgs_record(sgs, cfg, done, success, o):
    rows = jp.where(done, cfg, sgs["hist"].shape[0] - 1)
    col = sgs["count"][rows] % o["sgs_h"]
    hist = sgs["hist"].at[rows, col].set(success.astype(jp.float32))
    count = sgs["count"].at[rows].add(1)
    return dict(hist=hist.at[-1].set(0.0), count=count.at[-1].set(0))


def main():
    o = options()
    env = W.Walk(o["envs"])
    n, n_cfg = o["envs"], o["cfgs"]
    bank = W.sample_bank(n_cfg, o["seed"], o["bank"])
    key = jax.random.PRNGKey(o["seed"])
    n_obs, n_in = Q.N_OBS, Q.N_OBS + W.N_PRIV
    key, ka, kc = jax.random.split(key, 3)
    params = dict(actor=mlp_init(ka, (Q.N_OBS,) + ACTOR + (12,), 0.01), critic=mlp_init(kc, (n_in,) + CRITIC + (1,), 1.0),
                  log_std=jp.full(12, -1.0))
    norm = dict(mean=jp.zeros(n_in), var=jp.ones(n_in), count=jp.float32(1e-4))
    sgs = dict(hist=jp.zeros((n_cfg + 1, o["sgs_h"])), count=jp.zeros(n_cfg + 1, jp.int32))
    it0 = 0
    if o["init"]:
        ck = pickle.load(open(o["init"], "rb"))
        params, norm, it0 = jax.tree_util.tree_map(jp.asarray, ck["params"]), jax.tree_util.tree_map(jp.asarray, ck["norm"]), ck["it"]
        if ck.get("sgs") is not None and ck["sgs"]["hist"].shape == sgs["hist"].shape:
            sgs = jax.tree_util.tree_map(jp.asarray, ck["sgs"])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=o["lr"]))
    opt_state = tx.init(params)

    # --- the worlds ---
    d = env.make_data()
    key, k1, k2 = jax.random.split(key, 3)
    cfg0 = jax.random.randint(k1, (n,), 0, n_cfg)
    qpos, qvel, aux, fric, mass = jax.vmap(env.reset_slot, in_axes=(None, 0, 0))(bank, cfg0, jax.random.split(k2, n))
    d = d.replace(qpos=qpos, qvel=qvel)
    obs = jax.vmap(env.obs)(d, aux, jax.random.split(key, n))
    priv = jp.zeros((n, W.N_PRIV))
    step_v = jax.vmap(env.step, in_axes=(env.in_axes, 0, 0, 0, None, 0))
    reset_v = jax.vmap(env.reset_slot, in_axes=(None, 0, 0))

    def value(p, nm, ob, pv):
        return mlp(p["critic"], normalise(nm, jp.concatenate([ob, pv], -1)))[..., 0]

    def actor_mean(p, nm, ob):
        return mlp(p["actor"], normalise(dict(mean=nm["mean"][:n_obs], var=nm["var"][:n_obs]), ob))

    def rollout(carry, _, p, nm):
        d, aux, fric, mass, obs, priv, sgs, key = carry
        key, k_a, k_s, k_r, k_c = jax.random.split(key, 5)
        mu = actor_mean(p, nm, obs)
        std = jp.exp(p["log_std"])
        a = mu + std * jax.random.normal(k_a, mu.shape)
        logp = jp.sum(-0.5 * ((a - mu) / std) ** 2 - p["log_std"] - 0.5 * jp.log(2 * jp.pi), -1)
        v = value(p, nm, obs, priv)
        mxb = env.mx.tree_replace({"geom_friction": fric, "body_mass": mass})
        d2, aux2, obs2, priv2, r, done, trunc, info = step_v(mxb, d, aux, a, bank, jax.random.split(k_s, n))
        r = r + o["gamma"] * value(p, nm, obs2, priv2) * trunc            # time-out: bootstrap from the last state
        sgs = sgs_record(sgs, aux["cfg"], done, info["success"], o)
        logits, _ = sgs_logits(sgs, o)
        new_cfg = (jax.random.categorical(k_c, logits / o["sgs_temp"], shape=(n,)) if o["sampler"] == "sgs"
                   else jax.random.randint(k_c, (n,), 0, n_cfg))
        q_r, v_r, aux_r, fric_r, mass_r = reset_v(bank, new_cfg, jax.random.split(k_r, n))
        sel = lambda new, old: jp.where(done.reshape((n,) + (1,) * (old.ndim - 1)), new, old)   # noqa: E731
        d2 = d2.replace(qpos=sel(q_r, d2.qpos), qvel=sel(v_r, d2.qvel), qacc_warmstart=sel(jp.zeros_like(d2.qacc_warmstart), d2.qacc_warmstart),
                        ctrl=sel(jp.zeros_like(d2.ctrl), d2.ctrl))
        aux2 = jax.tree_util.tree_map(sel, aux_r, aux2)
        fric, mass = sel(fric_r, fric), sel(mass_r, mass)
        obs_n = sel(jax.vmap(env.obs)(d2, aux2, jax.random.split(key, n)), obs2)
        priv_n = sel(jp.zeros_like(priv2), priv2)
        tr = dict(obs=obs, priv=priv, a=a, logp=logp, v=v, r=r, done=done,
                  success=info["success"] & done, fallen=info["fallen"] & done, trunc=trunc,
                  rms_lin=jp.where(trunc, info["rms_lin"], 0.0), e_lin=info["e_lin"])
        return (d2, aux2, fric, mass, obs_n, priv_n, sgs, key), tr

    def loss_fn(p, nm, b):
        mu = actor_mean(p, nm, b["obs"])
        std = jp.exp(p["log_std"])
        logp = jp.sum(-0.5 * ((b["a"] - mu) / std) ** 2 - p["log_std"] - 0.5 * jp.log(2 * jp.pi), -1)
        ratio = jp.exp(logp - b["logp"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        pg = -jp.mean(jp.minimum(ratio * adv, jp.clip(ratio, 1 - o["clip"], 1 + o["clip"]) * adv))
        v = value(p, nm, b["obs"], b["priv"])
        vl = jp.mean((v - b["ret"]) ** 2)
        ent = jp.sum(p["log_std"] + 0.5 * jp.log(2 * jp.pi * jp.e))
        # KL(old || new) for diagonal Gaussians with the old std unknown per sample: use the log-ratio estimator
        kl = jp.mean((ratio - 1) - (logp - b["logp"]))
        return pg + o["vf"] * vl - o["ent"] * ent, (pg, vl, kl)

    @jax.jit
    def iteration(params, opt_state, norm, carry, lr):
        carry, tr = jax.lax.scan(lambda c, x: rollout(c, x, params, norm), carry, None, length=o["unroll"])
        d, aux, fric, mass, obs, priv, sgs, key = carry
        v_last = value(params, norm, obs, priv)

        def gae(nxt, t):
            adv_n, v_n = nxt
            nd = 1.0 - t["done"]
            delta = t["r"] + o["gamma"] * v_n * nd - t["v"]
            adv = delta + o["gamma"] * o["lam"] * nd * adv_n
            return (adv, t["v"]), adv
        _, adv = jax.lax.scan(gae, (jp.zeros(n), v_last), tr, reverse=True)
        ret = adv + tr["v"]
        norm_new = norm_update(norm, jp.concatenate([tr["obs"], tr["priv"]], -1).reshape(-1, n_in))
        batch = dict(obs=tr["obs"], priv=tr["priv"], a=tr["a"], logp=tr["logp"], adv=adv, ret=ret)
        batch = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), batch)
        total = batch["obs"].shape[0]
        hp = opt_state[1]
        opt_state = (opt_state[0], hp._replace(hyperparams=dict(hp.hyperparams, learning_rate=lr)))

        def epoch(st, k):
            perm = jax.random.permutation(k, total)
            mbs = jax.tree_util.tree_map(lambda x: x[perm].reshape((o["minibatches"], -1) + x.shape[1:]), batch)

            def mb(st, b):
                p, os_ = st
                (l, aux_l), g = jax.value_and_grad(loss_fn, has_aux=True)(p, norm, b)
                upd, os_ = tx.update(g, os_, p)
                p = optax.apply_updates(p, upd)
                p["log_std"] = jp.maximum(p["log_std"], -3.0)
                return (p, os_), aux_l
            return jax.lax.scan(mb, st, mbs)
        key, k = jax.random.split(key)
        (params, opt_state), (pg, vl, kl) = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k, o["epochs"]))
        carry = (d, aux, fric, mass, obs, priv, sgs, key)
        nd = jp.maximum(tr["done"].sum(), 1)
        nt = jp.maximum(tr["trunc"].sum(), 1)
        _, p_cfg = sgs_logits(sgs, o)
        seen = sgs["count"][:-1] > 0
        stats = dict(reward=tr["r"].mean(), episodes=tr["done"].sum(), success=tr["success"].sum() / nd,
                     fall=tr["fallen"].sum() / nd, rms_lin=tr["rms_lin"].sum() / nt, vel_err=jp.sqrt(tr["e_lin"].mean()),
                     kl=kl.mean(), v_loss=vl.mean(), std=jp.exp(params["log_std"]).mean(),
                     bank_success=jp.sum(jp.where(seen, sgs["hist"][:-1].sum(1) / jp.maximum(jp.minimum(sgs["count"][:-1], o["sgs_h"]), 1), 0)) / jp.maximum(seen.sum(), 1),
                     bank_seen=seen.mean(), bank_mastered=jp.mean(seen & (p_cfg > 0.9)))
        return params, opt_state, norm_new, carry, stats

    carry = (d, aux, fric, mass, obs, priv, sgs, key)
    per_it = n * o["unroll"]
    n_it = int(o["steps"] * 1e6 // per_it)
    os.makedirs(os.path.join(HERE, "policies"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = open(os.path.join(HERE, "logs", f"{o['name']}.csv"), "a", newline="")
    wr = None
    lr = o["lr"]
    print(f"{n} worlds, {n_cfg} configurations ({o['bank']}, {o['sampler']}), {n_it} iterations of {per_it} steps", flush=True)
    t0 = time.time()
    for it in range(it0, it0 + n_it):
        params, opt_state, norm, carry, st = iteration(params, opt_state, norm, carry, jp.float32(lr))
        st = {k: float(v) for k, v in st.items()}
        if st["kl"] > 2 * o["kl"]:                                       # KL-adaptive step size
            lr = max(lr / 1.5, 1e-5)
        elif st["kl"] < o["kl"] / 2:
            lr = min(lr * 1.5, 1e-2)
        steps = (it - it0 + 1) * per_it
        sps = steps / (time.time() - t0)
        row = dict(it=it, steps=steps, sps=sps, lr=lr, **st)
        if wr is None:
            wr = csv.DictWriter(log, fieldnames=list(row))
            if log.tell() == 0:
                wr.writeheader()
        wr.writerow(row)
        log.flush()
        if it % 5 == 0 or it == it0 + n_it - 1:
            print(f"it {it:4d}  {steps / 1e6:6.1f} M  {sps / 1e3:5.0f} k/s  r {st['reward']:+.3f}  success {st['success']:.2f}  "
                  f"fall {st['fall']:.2f}  vel err {st['vel_err']:.2f}  bank {st['bank_success']:.2f} "
                  f"(seen {st['bank_seen']:.2f}, mastered {st['bank_mastered']:.2f})  std {st['std']:.2f}  lr {lr:.1e}", flush=True)
        if (it + 1) % o["save"] == 0 or it == it0 + n_it - 1:
            save(o["name"], params, norm, carry[6], it + 1)
    print("saved", o["name"], flush=True)


def save(name, params, norm, sgs, it):
    to_np = lambda t: jax.tree_util.tree_map(np.asarray, t)              # noqa: E731
    pickle.dump(dict(params=to_np(params), norm=to_np(norm), sgs=to_np(sgs), it=it),
                open(os.path.join(HERE, "policies", f"{name}.pkl"), "wb"))
    a = to_np(params["actor"])
    out = {f"mlp_extractor_policy_net_{2 * i}_weight": Wt for i, (Wt, b) in enumerate(a[:-1])}
    out.update({f"mlp_extractor_policy_net_{2 * i}_bias": b for i, (Wt, b) in enumerate(a[:-1])})
    nm = to_np(norm)
    np.savez(os.path.join(HERE, "policies", f"{name}.npz"), obs_mean=nm["mean"][:Q.N_OBS], obs_var=nm["var"][:Q.N_OBS],
             clip_obs=10.0, epsilon=1e-8, action_net_weight=a[-1][0], action_net_bias=a[-1][1], **out)


if __name__ == "__main__":
    main()
