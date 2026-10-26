"""Train a task on the GPU: PPO in JAX over thousands of MuJoCo Warp worlds, with Success-Guided Sampling (SGS) or
uniform sampling of the task configurations in a fixed bank.

SGS as in Zhang et al. 2026 (arXiv 2610.12465): each configuration keeps its last H outcomes (H = 100); p_i is their
mean (unvisited: p_i = t); w_i = max((p_i + eps)^(k t) (1 - p_i + eps)^(k (1 - t)), eps); the next configuration
is drawn from softmax(log(w_i + eps) / T). Locomotion settings: t = 0.66, k = 5, T = 2, eps = 1e-8. --sgs_prior=1
(--sampler=sgsp) gives this repo's variant: one pseudo-outcome of t, so a configuration that failed
once early on is not frozen out.

--sampler=curr: a per-robot level curriculum (Rudin et al. 2021, "Learning to walk in minutes"). Each robot keeps a
level per terrain type; an episode draws a type uniformly at random and plays it at the robot's level for that type;
success moves that level up one, failure down one; succeeding at the top level sends the robot to a random level, so
it does not forget. --curr_mix of episodes (10 %) draw any configuration uniformly. Unlike SGS, every type keeps the
same share of practice, and every robot practises at the edge of what it can do.

Exploration noise: the policy's action noise (log_std) is learned. With an entropy bonus and no upper bound it grew
all through training in most runs (terrain 0.3 -> 1.2-1.75, rear -> 6.2; actions are clipped to [-1, 1], so the
noise costs the policy little). --std_max caps it; --ent_end anneals the entropy bonus linearly from --ent to that
value over the run. Tried in queue4 (std_max 0.5, ent_end 0): no better, terrain worse; the late slumps in held-out
success were not caused by the noise. Both default off.

--cat=p_max: constraints as terminations (Chane-Sane et al. 2024, CaT) for the pose tasks: an episode ends with
probability up to p_max when a constraint is broken (tasks.Pose.end), which costs the rewards it would have
collected. Only in training worlds; held-out evaluation and the CPU model never end an episode this way.

--teacher=1: the actor also reads the privileged state the critic reads (tasks.Terrain: a height scan of the course,
foot heights, true velocity, friction, mass, motor strength, delays), through a small encoder to a latent of 16
numbers (Lee et al. 2020, "Learning quadrupedal locomotion over challenging terrain"; Kumar et al. 2021, RMA). The
robot has no such sensors, so a teacher is not deployable: distill.py trains a blind student to act like it. A
teacher run also keeps policies/snaps/<name>/<steps>M.pkl at each held-out evaluation, to distill from its best.

Reward tuning: every reward term is logged on its own (logs/<name>.csv, columns r.<term>: its weighted mean per
step), and any weight can be set from the command line or a JSON file:
    python train.py --task=walk --w=slip=-0.5,air=2.0 --name=walk_try
    python train.py --task=walk --w_file=my_weights.json
    python terms.py walk_try                     # which terms dominate, and how they moved

Run (from fullbuild_v4/rl2, conda env capstan-gpu):
    python train.py --task=terrain --sampler=sgs --seed=0 --steps=600
Writes policies/<name>.npz (play.py, teleop.py), policies/<name>.pkl (to resume), logs/<name>.csv and
logs/<name>_bank.json (each configuration's final success estimate, for the per-level tables), and at every
held-out evaluation a snapshot policies/snaps/<name>/<steps>M.npz with logs/<name>_eval.jsonl (progress.py films them).
"""
import csv
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jp
import numpy as np
import optax

import core as C
import env as E

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = dict(task="walk", steps=300.0, envs=8192, cfgs=1040, eval_every=50, eval_n=4160, sampler="sgs", name="", init="", seed=0, bank_seed=0, save=50,
           unroll=24, epochs=5, minibatches=4, gamma=0.99, lam=0.95, clip=0.2, ent=0.004, vf=1.0, lr=1e-3, kl=0.01,
           sgs_t=0.66, sgs_k=5.0, sgs_temp=2.0, sgs_h=100, sgs_eps=1e-8, sgs_prior=0, w="", w_file="", nacon=24, njmax=160,
           std_max=0.0, ent_end=-1.0, curr_mix=0.1, cat=0.0, teacher=0)                 # std_max 0: no cap; ent_end < 0: constant
ACTOR, CRITIC = (512, 256, 128), (512, 256, 128)
ENC, LATENT = (64,), 16                                                 # a teacher's encoder of the privileged state


def options():
    o = dict(OPT)
    for a in sys.argv[1:]:
        k, v = a.lstrip("-").split("=", 1)
        o[k] = type(OPT[k])(v)
    if o["sampler"] == "sgsp":                                         # SGS with one pseudo-outcome of t: no freezing
        o["sgs_prior"] = 1
    if not o["name"]:
        o["name"] = f"{o['task']}_{o['sampler']}_s{o['seed']}"
    w = {}
    if o["w_file"]:
        w.update(json.load(open(o["w_file"])))
    for kv in filter(None, o["w"].split(",")):
        k, v = kv.split("=")
        w[k] = float(v)
    o["weights"] = w
    return o


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


def split_norm(norm, n_obs):
    return (dict(mean=norm["mean"][:n_obs], var=norm["var"][:n_obs]), dict(mean=norm["mean"][n_obs:], var=norm["var"][n_obs:]))


def actor_mean(p, norm, ob, pv):
    """The policy's mean action. A teacher (p has "enc") also reads the privileged state pv; returns (action, latent)."""
    n_obs = ob.shape[-1]
    nm_o, nm_p = split_norm(norm, n_obs)
    x = normalise(nm_o, ob)
    z = mlp(p["enc"], normalise(nm_p, pv)) if "enc" in p else jp.zeros(ob.shape[:-1] + (0,))
    return mlp(p["actor"], jp.concatenate([x, z], -1)), z


def make_evaluate(task, o, act):
    """Held-out evaluation: a different bank (seed bank_seed + 1000), every configuration once, the mean action, first
    episode only. act(params, norm, obs, priv) -> action. Returns evaluate(params, norm, key) -> success per episode."""
    n_ev = o["eval_n"]
    bank_ev = task.bank(n_ev, o["bank_seed"] + 1000)
    env_ev = E.GpuEnv(task, n_ev, o["nacon"], o["njmax"])
    step_ev = jax.vmap(env_ev.step, in_axes=(env_ev.in_axes, 0, 0, 0, None, 0))

    @jax.jit
    def evaluate(params, norm, key):
        k1, k2 = jax.random.split(key)
        d, aux, mp, ob, pv = env_ev.start(bank_ev, jp.arange(n_ev), k1)
        mxb = env_ev.mx.tree_replace(mp)

        def body(c, k):
            d, aux, ob, pv, over, ok = c
            a = act(params, norm, ob, pv)
            d, aux, ob, pv, _, _, done, _, success, _, _ = step_ev(mxb, d, aux, a, bank_ev, jax.random.split(k, n_ev))
            ok = ok | (~over & success)
            return (d, aux, ob, pv, over | done, ok), None
        (*_, ok), _ = jax.lax.scan(body, (d, aux, ob, pv, jp.zeros(n_ev, bool), jp.zeros(n_ev, bool)), jax.random.split(k2, task.EP))
        return ok
    return evaluate, bank_ev


def norm_update(norm, x):
    n, mean, var = x.shape[0], x.mean(0), x.var(0)
    tot = norm["count"] + n
    delta = mean - norm["mean"]
    m2 = norm["var"] * norm["count"] + var * n + delta ** 2 * norm["count"] * n / tot
    return dict(mean=norm["mean"] + delta * n / tot, var=m2 / tot, count=tot)


# --- SGS --------------------------------------------------------------------------------------------------
def sgs_p(sgs, o):
    n = jp.minimum(sgs["count"], o["sgs_h"]).astype(jp.float32)
    wins = sgs["hist"].sum(1)
    if o["sgs_prior"]:
        return (wins + o["sgs_t"]) / (n + 1.0)
    return jp.where(n > 0, wins / jp.maximum(n, 1.0), o["sgs_t"])


def sgs_logits(sgs, o):
    p = sgs_p(sgs, o)
    eps, k, t = o["sgs_eps"], o["sgs_k"], o["sgs_t"]
    w = jp.maximum((p + eps) ** (k * t) * (1.0 - p + eps) ** (k * (1.0 - t)), eps)
    return jp.log(w + eps)[:-1], p[:-1]                                # the last row is a dummy for unfinished envs


def sgs_record(sgs, cfg, done, success, o):
    """Store each finished episode's outcome in its configuration's ring buffer. Several robots often finish the same
    configuration in one step (8,192 robots over ~1,000 configurations): each gets its own slot, ranked among them, so
    every outcome is kept (an earlier version wrote them all to one slot and kept only one)."""
    rows = jp.where(done, cfg, sgs["hist"].shape[0] - 1)
    order = jp.argsort(rows, stable=True)
    srt = rows[order]
    rank = jp.zeros_like(rows).at[order].set(jp.arange(rows.shape[0]) - jp.searchsorted(srt, srt, side="left"))
    rows = jp.where(rank < o["sgs_h"], rows, sgs["hist"].shape[0] - 1)   # more than H at once: the buffer holds H
    col = (sgs["count"][rows] + rank) % o["sgs_h"]
    hist = sgs["hist"].at[rows, col].set(success.astype(jp.float32))
    count = sgs["count"].at[rows].add(1)
    return dict(sgs, hist=hist.at[-1].set(0.0), count=count.at[-1].set(0))


def curr_table(bank):
    """Bank indices by (type, level): table (K, LV, cmax), padded by repetition, and the count per cell."""
    kind, level = np.asarray(bank["kind"]), np.asarray(bank["level"])
    K, LV = int(kind.max()) + 1, int(level.max()) + 1
    cells = [[np.flatnonzero((kind == k) & (level == lv)) for lv in range(LV)] for k in range(K)]
    cmax = max(len(c) for row in cells for c in row)
    assert min(len(c) for row in cells for c in row) > 0, "every (type, level) needs a configuration"
    table = np.array([[np.resize(c, cmax) for c in row] for row in cells], np.int32)
    count = np.array([[len(c) for c in row] for row in cells], np.int32)
    return jp.asarray(table), jp.asarray(count)


def curr_update(lvl, kind, level, done, success, key, LV=10):
    """Move each finished robot's level for the type it just played: up on success, down on failure. Only when the
    episode was played at that level (not a uniform 'mix' episode). Success at the top: a random level."""
    n = lvl.shape[0]
    ar = jp.arange(n)
    cur = lvl[ar, kind]
    up = jp.where(cur >= LV - 1, jax.random.randint(key, (n,), 0, LV), cur + 1)
    new = jp.where(success, up, jp.maximum(cur - 1, 0))
    return lvl.at[ar, kind].set(jp.where(done & (level == cur), new, cur))


def curr_pick(lvl, table, count, n_cfg, mix, key):
    """The next configuration for every robot: a random type at the robot's level for it, or (mix) any."""
    n, K = lvl.shape
    k1, k2, k3, k4 = jax.random.split(key, 4)
    kind = jax.random.randint(k1, (n,), 0, K)
    lv = lvl[jp.arange(n), kind]
    j = jax.random.randint(k2, (n,), 0, count[kind, lv])
    return jp.where(jax.random.uniform(k3, (n,)) < mix, jax.random.randint(k4, (n,), 0, n_cfg), table[kind, lv, j])


def main():
    o = options()
    task = E.make_task(o["task"], o["weights"])
    env = E.GpuEnv(task, o["envs"], o["nacon"], o["njmax"])
    n, n_cfg = o["envs"], o["cfgs"]
    bank = task.bank(n_cfg, o["bank_seed"])
    key = jax.random.PRNGKey(o["seed"])
    n_obs, n_in = C.N_OBS, C.N_OBS + task.N_PRIV
    key, ka, kc = jax.random.split(key, 3)
    key, ke = jax.random.split(key)
    n_lat = LATENT if o["teacher"] else 0
    params = dict(actor=mlp_init(ka, (n_obs + n_lat,) + ACTOR + (12,), 0.01), critic=mlp_init(kc, (n_in,) + CRITIC + (1,), 1.0),
                  log_std=jp.full(12, -1.0))
    if o["teacher"]:
        params["enc"] = mlp_init(ke, (task.N_PRIV,) + ENC + (LATENT,), 1.0)
    norm = dict(mean=jp.zeros(n_in), var=jp.ones(n_in), count=jp.float32(1e-4))
    sgs = dict(hist=jp.zeros((n_cfg + 1, o["sgs_h"])), count=jp.zeros(n_cfg + 1, jp.int32))
    curr = o["sampler"] == "curr"
    if curr:
        c_table, c_count = curr_table(bank)
        bank_kind, bank_level = jp.asarray(bank["kind"]), jp.asarray(bank["level"])
        key, kl = jax.random.split(key)
        sgs["lvl"] = jax.random.randint(kl, (n, c_table.shape[0]), 0, 3)    # every robot starts at levels 0-2
    it0 = 0
    if o["init"]:
        ck = pickle.load(open(o["init"], "rb"))
        params, norm, it0 = jax.tree_util.tree_map(jp.asarray, ck["params"]), jax.tree_util.tree_map(jp.asarray, ck["norm"]), ck["it"]
        if ck.get("sgs") is not None and ck["sgs"]["hist"].shape == sgs["hist"].shape:
            sgs = jax.tree_util.tree_map(jp.asarray, ck["sgs"])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.inject_hyperparams(optax.adam)(learning_rate=o["lr"]))
    opt_state = tx.init(params)
    terms0 = list(task.w)

    # --- the worlds ---
    key, k1, k2 = jax.random.split(key, 3)
    cfg0 = curr_pick(sgs["lvl"], c_table, c_count, n_cfg, o["curr_mix"], k1) if curr else jax.random.randint(k1, (n,), 0, n_cfg)
    cat = jp.full(n, o["cat"], jp.float32)                              # constraints as terminations: training only
    d, aux, mp, obs, priv = env.start(bank, cfg0, k2, cat=cat)
    step_v = jax.vmap(env.step, in_axes=(env.in_axes, 0, 0, 0, None, 0))
    evaluate, bank_ev = make_evaluate(task, o, lambda p, nm, ob, pv: actor_mean(p, nm, ob, pv)[0])

    def value(p, nm, ob, pv):
        return mlp(p["critic"], normalise(nm, jp.concatenate([ob, pv], -1)))[..., 0]

    def rollout(carry, _, p, nm):
        d, aux, mp, obs, priv, sgs, key = carry
        key, k_a, k_s, k_r, k_c = jax.random.split(key, 5)
        mu, _ = actor_mean(p, nm, obs, priv)
        std = jp.exp(p["log_std"])
        a = mu + std * jax.random.normal(k_a, mu.shape)
        logp = jp.sum(-0.5 * ((a - mu) / std) ** 2 - p["log_std"] - 0.5 * jp.log(2 * jp.pi), -1)
        v = value(p, nm, obs, priv)
        mxb = env.mx.tree_replace(mp)
        d2, aux2, obs2, priv2, r, terms, done, trunc, success, fallen, err = step_v(mxb, d, aux, a, bank, jax.random.split(k_s, n))
        r = r + o["gamma"] * value(p, nm, obs2, priv2) * trunc            # time-out: bootstrap from the last state
        sgs = sgs_record(sgs, aux["cfg"], done, success, o)
        logits, _ = sgs_logits(sgs, o)
        if curr:
            k_c, k_u = jax.random.split(k_c)
            sgs = dict(sgs, lvl=curr_update(sgs["lvl"], bank_kind[aux["cfg"]], bank_level[aux["cfg"]], done, success, k_u,
                                                   c_table.shape[1]))
            new_cfg = curr_pick(sgs["lvl"], c_table, c_count, n_cfg, o["curr_mix"], k_c)
        elif o["sampler"].startswith("sgs"):
            new_cfg = jax.random.categorical(k_c, logits / o["sgs_temp"], shape=(n,))
        else:
            new_cfg = jax.random.randint(k_c, (n,), 0, n_cfg)
        d2, aux2, mp, obs_n, priv_n = env.respawn(bank, d2, aux2, mp, obs2, priv2, done, new_cfg, k_r, cat=cat)
        tr = dict(obs=obs, priv=priv, a=a, logp=logp, v=v, r=r, done=done, success=success & done, fallen=fallen & done,
                  err=err, terms=terms)
        return (d2, aux2, mp, obs_n, priv_n, sgs, key), tr

    def loss_fn(p, nm, b, ent_c):
        mu, _ = actor_mean(p, nm, b["obs"], b["priv"])
        std = jp.exp(p["log_std"])
        logp = jp.sum(-0.5 * ((b["a"] - mu) / std) ** 2 - p["log_std"] - 0.5 * jp.log(2 * jp.pi), -1)
        ratio = jp.exp(logp - b["logp"])
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        pg = -jp.mean(jp.minimum(ratio * adv, jp.clip(ratio, 1 - o["clip"], 1 + o["clip"]) * adv))
        v = value(p, nm, b["obs"], b["priv"])
        vl = jp.mean((v - b["ret"]) ** 2)
        ent = jp.sum(p["log_std"] + 0.5 * jp.log(2 * jp.pi * jp.e))
        kl = jp.mean((ratio - 1) - (logp - b["logp"]))
        return pg + o["vf"] * vl - ent_c * ent, (pg, vl, kl)

    log_std_max = float(np.log(o["std_max"])) if o["std_max"] > 0 else 10.0

    @jax.jit
    def iteration(params, opt_state, norm, carry, lr, ent_c):
        carry, tr = jax.lax.scan(lambda c, x: rollout(c, x, params, norm), carry, None, length=o["unroll"])
        d, aux, mp, obs, priv, sgs, key = carry
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
                (l, aux_l), g = jax.value_and_grad(loss_fn, has_aux=True)(p, norm, b, ent_c)
                upd, os_ = tx.update(g, os_, p)
                p = optax.apply_updates(p, upd)
                p["log_std"] = jp.clip(p["log_std"], -3.0, log_std_max)
                return (p, os_), aux_l
            return jax.lax.scan(mb, st, mbs)
        key, k = jax.random.split(key)
        (params, opt_state), (pg, vl, kl) = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k, o["epochs"]))
        carry = (d, aux, mp, obs, priv, sgs, key)
        nd = jp.maximum(tr["done"].sum(), 1)
        _, p_cfg = sgs_logits(sgs, o)
        seen = sgs["count"][:-1] > 0
        stats = dict(reward=tr["r"].mean(), episodes=tr["done"].sum(), success=tr["success"].sum() / nd,
                     fall=tr["fallen"].sum() / nd, err=tr["err"].mean(), kl=kl.mean(), v_loss=vl.mean(),
                     std=jp.exp(params["log_std"]).mean(),
                     bank_success=jp.sum(jp.where(seen, p_cfg, 0)) / jp.maximum(seen.sum(), 1),
                     bank_seen=seen.mean(), bank_mastered=jp.mean(seen & (p_cfg > 0.9)), bank_hopeless=jp.mean(seen & (p_cfg < 0.1)),
                     curr_level=jp.mean(sgs["lvl"]) if curr else jp.float32(0.0))
        stats.update({f"r.{k}": tr["terms"][k].mean() for k in terms0})
        return params, opt_state, norm_new, carry, stats

    carry = (d, aux, mp, obs, priv, sgs, key)
    per_it = n * o["unroll"]
    n_it = int(o["steps"] * 1e6 // per_it)
    os.makedirs(os.path.join(HERE, "policies"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log = open(os.path.join(HERE, "logs", f"{o['name']}.csv"), "a", newline="")
    json.dump({k: v for k, v in o.items()} | dict(weights_used=task.w), open(os.path.join(HERE, "logs", f"{o['name']}_opts.json"), "w"), indent=1)
    wr = None
    lr = o["lr"]
    print(f"{o['task']}: {n} worlds, {n_cfg} configurations ({o['sampler']}), {n_it} iterations of {per_it} steps", flush=True)
    t0 = time.time()
    for it in range(it0, it0 + n_it):
        frac = (it - it0) / max(n_it - 1, 1)
        ent_c = o["ent"] if o["ent_end"] < 0 else o["ent"] + (o["ent_end"] - o["ent"]) * frac
        params, opt_state, norm, carry, st = iteration(params, opt_state, norm, carry, jp.float32(lr), jp.float32(ent_c))
        st = {k: float(v) for k, v in st.items()}
        if not np.isfinite(st["reward"]):
            print("non-finite reward; stopping", flush=True)
            break
        if st["kl"] > 2 * o["kl"]:
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
        if it % 10 == 0 or it == it0 + n_it - 1:
            top = sorted(((k[2:], v) for k, v in st.items() if k.startswith("r.")), key=lambda kv: -abs(kv[1]))[:4]
            print(f"it {it:4d}  {steps / 1e6:6.1f} M  {sps / 1e3:4.0f} k/s  r {st['reward']:+.3f}  success {st['success']:.2f}  "
                  f"fall {st['fall']:.2f}  err {st['err']:.2f}  bank {st['bank_success']:.2f} (mastered {st['bank_mastered']:.2f}, "
                  f"hopeless {st['bank_hopeless']:.2f})  std {st['std']:.2f}  level {st['curr_level']:.1f}  | " + "  ".join(f"{k} {v:+.2f}" for k, v in top), flush=True)
        if it % o["eval_every"] == 0 or it == it0 + n_it - 1:
            ok = np.asarray(evaluate(params, norm, jax.random.PRNGKey(10_000 + it)))
            kind, level = np.asarray(bank_ev["kind"]), np.asarray(bank_ev["level"])
            table = {f"{k},{lv}": float(ok[(kind == k) & (level == lv)].mean()) for k in np.unique(kind) for lv in np.unique(level)}
            with open(os.path.join(HERE, "logs", f"{o['name']}_eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(it=it, steps=steps, success=float(ok.mean()), table=table)) + "\n")
            print(f"   eval {steps / 1e6:6.1f} M  held-out success {ok.mean():.3f}", flush=True)
            snap = os.path.join(HERE, "policies", "snaps", o["name"], f"{steps // 1_000_000:05d}M")
            export(params, norm, o["task"], snap + ".npz")
            if o["teacher"]:
                pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), norm=jax.tree_util.tree_map(np.asarray, norm),
                                 it=it + 1, task=o["task"]), open(snap + ".pkl", "wb"))
        if (it + 1) % o["save"] == 0 or it == it0 + n_it - 1:
            save(o, task, params, norm, carry[5], bank, it + 1)
    print("saved", o["name"], flush=True)


def export(params, norm, task_name, path):
    """The deployable policy: actor weights and observation normalisation, numpy only (policy.py), ~1 MB. A teacher's
    file is marked kind=teacher (it needs the privileged state: policy.py refuses it; distill.py makes its student)."""
    to_np = lambda t: jax.tree_util.tree_map(np.asarray, t)              # noqa: E731
    a = to_np(params["actor"])
    out = {f"mlp_extractor_policy_net_{2 * i}_weight": Wt for i, (Wt, b) in enumerate(a[:-1])}
    out.update({f"mlp_extractor_policy_net_{2 * i}_bias": b for i, (Wt, b) in enumerate(a[:-1])})
    nm = to_np(norm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, obs_mean=nm["mean"][:C.N_OBS], obs_var=nm["var"][:C.N_OBS], clip_obs=10.0, epsilon=1e-8,
             action_net_weight=a[-1][0], action_net_bias=a[-1][1], task=task_name, kind="teacher" if "enc" in params else "mlp", **out)


def save(o, task, params, norm, sgs, bank, it):
    name = o["name"]
    to_np = lambda t: jax.tree_util.tree_map(np.asarray, t)              # noqa: E731
    pickle.dump(dict(params=to_np(params), norm=to_np(norm), sgs=to_np(sgs), it=it, task=o["task"]),
                open(os.path.join(HERE, "policies", f"{name}.pkl"), "wb"))
    export(params, norm, o["task"], os.path.join(HERE, "policies", f"{name}.npz"))
    s = to_np(sgs)
    nseen = np.minimum(s["count"][:-1], o["sgs_h"])
    p = np.where(nseen > 0, s["hist"][:-1].sum(1) / np.maximum(nseen, 1), np.nan)
    kind, level = np.asarray(bank["kind"]), np.asarray(bank["level"])
    table = {f"{k},{lv}": float(np.nanmean(p[(kind == k) & (level == lv)])) if np.any((kind == k) & (level == lv) & (nseen > 0)) else None
             for k in np.unique(kind) for lv in np.unique(level)}
    json.dump(dict(it=it, p=table, visits=np.asarray(s["count"][:-1]).tolist()),
              open(os.path.join(HERE, "logs", f"{name}_bank.json"), "w"))


if __name__ == "__main__":
    main()
