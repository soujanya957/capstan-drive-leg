"""Train a task with FastSAC: off-policy soft actor-critic over thousands of GPU worlds (Seo et al. 2025, FastTD3, and
its SAC variant in "Learning sim-to-real humanoid locomotion in 15 minutes", 2025). Same worlds, tasks, rewards,
held-out evaluation and output files as train.py (PPO), so the two compare directly.

What is different from PPO:
  - Off-policy: every transition goes into a replay buffer on the GPU (1 M by default) and is learned from many
    times; PPO uses each batch for 5 passes and throws it away.
  - The critic learns Q(state, action) (how good this action is here), not V(state); the actor is trained to pick
    actions the critic rates highly. Two critics; the target averages them (--cdq=1: the more pessimistic one).
  - Distributional critic (C51): each critic predicts a distribution over 101 return values, not one number.
  - "Soft": the actor is also paid for its entropy, alpha * H, with alpha tuned so the entropy stays near a
    target (--ent_target). Actions are a tanh-squashed Gaussian, so they stay in [-1, 1].
  - Asymmetric, as in PPO: the critic reads the privileged state, the actor only the robot's sensors.
  - Large batches, several updates per environment step, a soft-updated target critic (tau = 0.1).
  - Settings from the FastSAC paper (arXiv 2512.01996): the target averages the two critics (taking the more
    pessimistic one, --cdq=1, hurts with layer norm), target entropy 0, initial alpha 0.001, pre-tanh std at most 1,
    gamma 0.97, AdamW (weight decay 0.001, beta2 0.95). terrain_sac_s0 and terrain_sac3_* ran with the earlier
    defaults: --cdq=1 --ent_target=-6 --alpha0=0.01 --log_std_max=1 --gamma=0.99 --wd=0 --b2=0.999.
  - n-step returns (--n_step=3): the target adds up 3 real rewards before trusting the critic, so reward reaches
    earlier steps faster. With 1-step targets the first terrain run (terrain_sac_s0) peaked at 33 % vs PPO's 71 %.
  - --look=1: the actor also reads a forward range sensor (hw.LOOK_*, tasks.Terrain.observe), so a deployable policy
    can see the ground ahead. --teacher=1: the actor reads the whole privileged state through an encoder (as
    train.py --teacher=1); not deployable, distill.py turns it into a student (its pkl snapshots are kept).
  - RL Token residual (--base=policies/<run>.pkl; Xu et al. 2026, arXiv 2604.23073): the base policy is frozen; its
    last hidden layer (128 numbers) is the "token" of the situation. A small actor reads the token and the base
    action and outputs a correction, at most --resid_scale per joint: a = clip(a_base + scale * tanh(u)). The actor
    loss adds --anchor * |a - a_base|^2, so it only moves away from the base where the critic says it pays. The
    critics are new. Made for adapting a working policy with little data (on the real robot, or here a sim with
    --shift=strength=0.55:0.7,mass_add=3:5: those bank fields redrawn in that range, for training and the held-out
    test). policy.py runs the result (kind=residual: base and correction in one file).
  - Symmetry augmentation (--sym=1): half of every batch is replaced by its left-right mirror image
    (tasks.Task.mirror), so each transition also teaches the mirrored case and the gait is pushed toward symmetry.

    python train_sac.py --task=terrain --seed=0 --steps=400 --name=terrain_sac_s0

Writes the same files as train.py: policies/<name>.npz (tanh output: policy.py reads squash=1), snapshots and
logs/<name>_eval.jsonl at every held-out evaluation, logs/<name>.csv (reward terms as r.<term>, plus critic loss,
Q, alpha, entropy), logs/<name>_opts.json, policies/<name>.pkl (to resume).
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
import train as Tr

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = dict(task="terrain", steps=400.0, envs=4096, cfgs=1040, chunk=25, updates=4, batch=16384, buffer=1_048_576,
           gamma=0.97, tau=0.1, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4, ent_target=0.0, alpha0=0.001,
           cdq=0, log_std_max=0.0, wd=0.001, b2=0.95, sym=0, look=0, teacher=0, base="", resid_scale=0.3, anchor=1.0,
           shift="",
           v_min=-10.0, v_max=50.0, atoms=101, n_step=3, start=0.5, eval_every=8.0, eval_n=4160, seed=0, bank_seed=0,
           name="", init="", w="", w_file="", nacon=24, njmax=160)
ACTOR, CRITIC = (512, 256, 128), (768, 384, 192)
RESID = (256, 256)
RES = {}                                      # an RL Token run's frozen base (main sets it): fwd(ob) -> (token, a_base)
LOG_STD = [-5.0, 0.0]                         # pre-tanh log std bounds; the upper one is --log_std_max


def options():
    o = dict(OPT)
    for a in sys.argv[1:]:
        k, v = a.lstrip("-").split("=", 1)
        o[k] = type(OPT[k])(v)
    o["name"] = o["name"] or f"{o['task']}_sac_s{o['seed']}"
    w = json.load(open(o["w_file"])) if o["w_file"] else {}
    for kv in filter(None, o["w"].split(",")):
        k, v = kv.split("=")
        w[k] = float(v)
    o["weights"] = w
    return o


def ln_mlp_init(key, sizes):
    """Critic layers with LayerNorm on the hidden layers (gain, bias per layer)."""
    ps = Tr.mlp_init(key, sizes, 1.0)
    return dict(lin=ps, ln=[(jp.ones(b.shape), jp.zeros(b.shape)) for _, b in ps[:-1]])


def ln_mlp(p, x):
    for (Wt, b), (g, c) in zip(p["lin"][:-1], p["ln"]):
        h = x @ Wt.T + b
        h = (h - h.mean(-1, keepdims=True)) / jp.sqrt(h.var(-1, keepdims=True) + 1e-5) * g + c
        x = jax.nn.relu(h)
    Wt, b = p["lin"][-1]
    return x @ Wt.T + b


def actor_dist(p, norm, ob, pv):
    if RES:                                                               # a residual on the frozen base's token
        tok, a_b = RES["fwd"](ob)
        out = Tr.mlp(p, jp.concatenate([tok, a_b], -1))
    elif isinstance(p, dict):                                             # a teacher: also reads the privileged state
        out = Tr.actor_mean(p, norm, ob, pv)[0]
    else:
        nm_o, _ = Tr.split_norm(norm, ob.shape[-1])
        out = Tr.mlp(p, Tr.normalise(nm_o, ob))
    mu, log_std = out[..., :12], jp.clip(out[..., 12:], *LOG_STD)
    return mu, log_std


def sample(p, norm, ob, pv, key):
    """A tanh-squashed Gaussian action and its log-probability."""
    mu, log_std = actor_dist(p, norm, ob, pv)
    u = mu + jp.exp(log_std) * jax.random.normal(key, mu.shape)
    a = jp.tanh(u)
    logp = jp.sum(-0.5 * ((u - mu) / jp.exp(log_std)) ** 2 - log_std - 0.5 * jp.log(2 * jp.pi), -1)
    logp = logp - jp.sum(2 * (jp.log(2.0) - u - jax.nn.softplus(-2 * u)), -1)      # log |d tanh(u) / du|
    if RES:
        a = jp.clip(RES["fwd"](ob)[1] + RES["scale"] * a, -1.0, 1.0)
        logp = logp - 12 * jp.log(RES["scale"])
    return a, logp


def deterministic(p, norm, ob, pv):
    a = jp.tanh(actor_dist(p, norm, ob, pv)[0])
    return jp.clip(RES["fwd"](ob)[1] + RES["scale"] * a, -1.0, 1.0) if RES else a


def load_base(path, scale):
    """The frozen base of an RL Token run: (token, base action) from the observation, with the base's own input
    normalisation. The token is the policy network's last hidden layer."""
    ck = pickle.load(open(path, "rb"))
    bp = jax.tree_util.tree_map(jp.asarray, ck["params"]["actor"])
    nm = jax.tree_util.tree_map(jp.asarray, ck["norm"])

    def fwd(ob):
        x = Tr.normalise(Tr.split_norm(nm, ob.shape[-1])[0], ob)
        for Wt, b in bp[:-1]:
            x = jax.nn.elu(x @ Wt.T + b)
        Wt, b = bp[-1]
        return x, jp.tanh((x @ Wt.T + b)[..., :12])
    RES.update(fwd=jax.jit(fwd), scale=jp.float32(scale), params=bp, norm=nm, n_tok=bp[-1][0].shape[1])


def shift_bank(task, spec):
    """--shift=field=lo:hi,...: redraw those bank fields uniformly in the range (training and held-out banks)."""
    ranges = {k: tuple(map(float, v.split(":"))) for k, v in (kv.split("=") for kv in spec.split(","))}
    bank0 = task.bank

    def bank(n, seed=0):
        b, r = bank0(n, seed), np.random.default_rng(seed + 7777)
        assert set(ranges) <= set(b), f"--shift: no such bank field in {sorted(b)}"
        return dict(b, **{k: jp.asarray(r.uniform(lo, hi, n), jp.float32) for k, (lo, hi) in ranges.items()})
    task.bank = bank


def main():
    o = options()
    task = E.make_task(o["task"], o["weights"])
    task.look = bool(o["look"])
    if o["shift"]:
        shift_bank(task, o["shift"])
    if o["base"]:
        load_base(o["base"], o["resid_scale"])
    n, n_cfg = o["envs"], o["cfgs"]
    env = E.GpuEnv(task, n, o["nacon"], o["njmax"])
    bank = task.bank(n_cfg, o["bank_seed"])
    n_obs, n_priv = task.n_obs, task.N_PRIV
    n_in = n_obs + n_priv
    support = jp.linspace(o["v_min"], o["v_max"], o["atoms"])
    (m_oi, m_os), (m_pi, m_ps), (m_ai, m_as) = (tuple(jp.asarray(x) for x in pair) for pair in task.mirror())
    dz = (o["v_max"] - o["v_min"]) / (o["atoms"] - 1)

    key = jax.random.PRNGKey(o["seed"])
    key, ka, k1, k2 = jax.random.split(key, 4)
    if o["base"]:
        actor = Tr.mlp_init(ka, (RES["n_tok"] + 12,) + RESID + (24,), 0.01)
    elif o["teacher"]:
        ka, ke = jax.random.split(ka)
        actor = dict(actor=Tr.mlp_init(ka, (n_obs + Tr.LATENT,) + ACTOR + (24,), 0.01),
                     enc=Tr.mlp_init(ke, (n_priv,) + Tr.ENC + (Tr.LATENT,), 1.0))
    else:
        actor = Tr.mlp_init(ka, (n_obs,) + ACTOR + (24,), 0.01)
    critics = jax.vmap(lambda k: ln_mlp_init(k, (n_in + 12,) + CRITIC + (o["atoms"],)))(jp.stack([k1, k2]))
    params = dict(actor=actor, critics=critics, target=critics, log_alpha=jp.log(jp.float32(o["alpha0"])))
    norm = dict(mean=jp.zeros(n_in), var=jp.ones(n_in), count=jp.float32(1e-4))
    if o["init"]:
        ck = pickle.load(open(o["init"], "rb"))
        params, norm = jax.tree_util.tree_map(jp.asarray, ck["params"]), jax.tree_util.tree_map(jp.asarray, ck["norm"])
    LOG_STD[1] = o["log_std_max"]
    opt_a, opt_c = (optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(o[k], b2=o["b2"], weight_decay=o["wd"]))
                    for k in ("lr_actor", "lr_critic"))
    opt_l = optax.adam(o["lr_alpha"])
    opts = dict(a=opt_a.init(params["actor"]), c=opt_c.init(params["critics"]), l=opt_l.init(params["log_alpha"]))

    T = o["buffer"] // n                                                 # buffer rows: one per step, one column per world
    buf = dict(obs=jp.zeros((T, n, n_obs)), priv=jp.zeros((T, n, n_priv)), a=jp.zeros((T, n, 12)), r=jp.zeros((T, n)),
               term=jp.zeros((T, n)), done=jp.zeros((T, n)), obs2=jp.zeros((T, n, n_obs)), priv2=jp.zeros((T, n, n_priv)))
    rb = dict(buf=buf, ptr=jp.int32(0), size=jp.int32(0))

    def q_logits(cp, nm, ob, pv, a):
        x = jp.concatenate([Tr.normalise(nm, jp.concatenate([ob, pv], -1)), a], -1)
        return jax.vmap(ln_mlp, in_axes=(0, None))(cp, x)                    # (2, B, atoms)

    def q_value(logits):
        return jp.sum(jax.nn.softmax(logits, -1) * support, -1)

    def project(probs, r, disc, shift):
        """C51: move each atom to r + disc (z - shift) and spread it over the two nearest atoms."""
        tz = jp.clip(r[:, None] + disc[:, None] * (support[None] - shift[:, None]), o["v_min"], o["v_max"])
        b = (tz - o["v_min"]) / dz
        lo, hi = jp.floor(b).astype(jp.int32), jp.ceil(b).astype(jp.int32)
        w_lo = jp.where(lo == hi, 1.0, hi - b)
        w_hi = jp.where(lo == hi, 0.0, b - lo)
        out = jax.vmap(lambda p, l, wl: jp.zeros(o["atoms"]).at[l].add(p * wl))(probs, lo, w_lo)
        return out + jax.vmap(lambda p, h, wh: jp.zeros(o["atoms"]).at[h].add(p * wh))(probs, hi, w_hi)

    def nstep_batch(rb, key):
        """A batch of n-step transitions: from step t of world e, add up to n rewards (stopping at the end of an
        episode), then bootstrap from the state reached, discounted gamma^k, or not at all if the episode really ended."""
        k1, k2 = jax.random.split(key)
        nn = o["n_step"]
        u = jax.random.randint(k1, (o["batch"],), 0, jp.maximum(rb["size"] - nn + 1, 1))
        t = (rb["ptr"] - rb["size"] + u) % T                               # oldest valid row + u
        e = jax.random.randint(k2, (o["batch"],), 0, n)
        B = rb["buf"]
        ret, disc, alive = jp.zeros(o["batch"]), jp.ones(o["batch"]), jp.ones(o["batch"])
        last = t
        for k in range(nn):
            tk = (t + k) % T
            ret = ret + alive * disc * B["r"][tk, e]
            disc = jp.where(alive > 0, disc * o["gamma"], disc)
            last = jp.where(alive > 0, tk, last)
            term_k = jp.where(alive > 0, B["term"][tk, e], 0.0)
            disc = disc * (1 - term_k)
            alive = alive * (1 - B["done"][tk, e])
        b = dict(obs=B["obs"][t, e], priv=B["priv"][t, e], a=B["a"][t, e], r=ret, disc=disc,
                 obs2=B["obs2"][last, e], priv2=B["priv2"][last, e])
        if o["sym"]:                                                          # mirror every other sample
            flip = (jp.arange(o["batch"]) % 2 == 1)[:, None]
            mir = dict(obs=(m_oi, m_os), obs2=(m_oi, m_os), priv=(m_pi, m_ps), priv2=(m_pi, m_ps), a=(m_ai, m_as))
            b.update({k: jp.where(flip, b[k][:, i] * sg, b[k]) for k, (i, sg) in mir.items()})
        return b

    def update(params, opts, norm, b, key):
        k1, k2 = jax.random.split(key)
        alpha = jp.exp(params["log_alpha"])
        a2, logp2 = sample(params["actor"], norm, b["obs2"], b["priv2"], k1)
        tl = q_logits(params["target"], norm, b["obs2"], b["priv2"], a2)
        tq = q_value(tl)
        if o["cdq"]:                                                         # the more pessimistic critic, per sample
            probs = jax.nn.softmax(jp.where((jp.argmin(tq, 0) == 0)[:, None], tl[0], tl[1]), -1)
        else:                                                                # the two critics' average (FastSAC 2025)
            probs = jax.nn.softmax(tl, -1).mean(0)
        target = jax.lax.stop_gradient(project(probs, b["r"], b["disc"], alpha * logp2))

        def critic_loss(cp):
            lg = q_logits(cp, norm, b["obs"], b["priv"], b["a"])
            return -jp.mean(jp.sum(target[None] * jax.nn.log_softmax(lg, -1), -1)) * 2, q_value(lg).mean()
        (lc, q), g = jax.value_and_grad(critic_loss, has_aux=True)(params["critics"])
        upd, opts_c = opt_c.update(g, opts["c"], params["critics"])
        critics = optax.apply_updates(params["critics"], upd)

        def actor_loss(ap):
            a, logp = sample(ap, norm, b["obs"], b["priv"], k2)
            qa = q_value(q_logits(critics, norm, b["obs"], b["priv"], a))
            pen = o["anchor"] * jp.sum((a - RES["fwd"](b["obs"])[1]) ** 2, -1) if RES else 0.0
            return jp.mean(alpha * logp - (qa.min(0) if o["cdq"] else qa.mean(0)) + pen), logp
        (la, logp), g = jax.value_and_grad(actor_loss, has_aux=True)(params["actor"])
        upd, opts_a = opt_a.update(g, opts["a"], params["actor"])
        actor_p = optax.apply_updates(params["actor"], upd)
        ent = -jax.lax.stop_gradient(logp).mean()

        g = jax.grad(lambda la_: -la_ * (o["ent_target"] - ent))(params["log_alpha"])
        upd, opts_l = opt_l.update(g, opts["l"], params["log_alpha"])
        target_p = jax.tree_util.tree_map(lambda t, c: (1 - o["tau"]) * t + o["tau"] * c, params["target"], critics)
        params = dict(actor=actor_p, critics=critics, target=target_p, log_alpha=optax.apply_updates(params["log_alpha"], upd))
        return params, dict(a=opts_a, c=opts_c, l=opts_l), dict(critic_loss=lc, q=q, actor_loss=la, entropy=ent, alpha=alpha)

    step_v = jax.vmap(env.step, in_axes=(env.in_axes, 0, 0, 0, None, 0))
    evaluate, bank_ev = Tr.make_evaluate(task, o, deterministic)
    terms0 = list(task.w)
    start_size = max(int(o["start"] * 1e6) // n, o["n_step"])

    @jax.jit
    def chunk(params, opts, norm, rb, carry):
        def one(c, _):
            params, opts, norm, rb, (d, aux, mp, obs, priv, key) = c
            key, k_a, k_s, k_r, k_c, k_u = jax.random.split(key, 6)
            a, _ = sample(params["actor"], norm, obs, priv, k_a)
            d2, aux2, obs2, priv2, r, terms, done, trunc, success, fallen, err = step_v(env.mx.tree_replace(mp), d, aux, a, bank,
                                                                                        jax.random.split(k_s, n))
            term = (done & ~trunc).astype(jp.float32)                     # a time-out is not an ending: bootstrap
            i = rb["ptr"]
            new = dict(obs=obs, priv=priv, a=a, r=r, term=term, done=done.astype(jp.float32), obs2=obs2, priv2=priv2)
            rb = dict(buf={k: rb["buf"][k].at[i].set(v) for k, v in new.items()}, ptr=(i + 1) % T, size=jp.minimum(rb["size"] + 1, T))
            norm = Tr.norm_update(norm, jp.concatenate([obs, priv], -1))
            d2, aux2, mp, obs2, priv2 = env.respawn(bank, d2, aux2, mp, obs2, priv2, done, jax.random.randint(k_c, (n,), 0, n_cfg), k_r)

            def learn(st, k):
                p, op = st
                kb, ku = jax.random.split(k)
                p, op, info = update(p, op, norm, nstep_batch(rb, kb), ku)
                return (p, op), info
            (p_new, op_new), info = jax.lax.scan(learn, (params, opts), jax.random.split(k_u, o["updates"]))
            ready = rb["size"] >= start_size
            params = jax.tree_util.tree_map(lambda x, y: jp.where(ready, x, y), p_new, params)
            opts = jax.tree_util.tree_map(lambda x, y: jp.where(ready, x, y), op_new, opts)
            info = jax.tree_util.tree_map(lambda x: x.mean(), info)
            st = dict(reward=r.mean(), done=done.sum(), success=(success & done).sum(), fall=(fallen & done).sum(), err=err.mean(),
                      **{f"r.{k}": terms[k].mean() for k in terms0}, **info)
            return (params, opts, norm, rb, (d2, aux2, mp, obs2, priv2, key)), st
        (params, opts, norm, rb, carry), st = jax.lax.scan(one, (params, opts, norm, rb, carry), None, length=o["chunk"])
        nd = jp.maximum(st["done"].sum(), 1)
        out = {k: v.mean() for k, v in st.items() if k not in ("done", "success", "fall")}
        out.update(episodes=st["done"].sum(), success=st["success"].sum() / nd, fall=st["fall"].sum() / nd)
        return params, opts, norm, rb, carry, out

    key, k0, k1 = jax.random.split(key, 3)
    carry = env.start(bank, jax.random.randint(k0, (n,), 0, n_cfg), k1) + (key,)
    per_it = n * o["chunk"]
    n_it = int(o["steps"] * 1e6 // per_it)
    ev_every = max(1, int(o["eval_every"] * 1e6 // per_it))
    os.makedirs(os.path.join(HERE, "policies"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump(dict(o, algo="fastsac", weights_used=task.w), open(os.path.join(HERE, "logs", f"{o['name']}_opts.json"), "w"), indent=1)
    log = open(os.path.join(HERE, "logs", f"{o['name']}.csv"), "w", newline="")
    wr = None
    print(f"{o['task']} FastSAC: {n} worlds, {n_it} iterations of {per_it} steps, {o['updates']} updates of {o['batch']} per step, "
          f"buffer {T * n}, {o['n_step']}-step returns", flush=True)
    t0 = time.time()
    for it in range(n_it):
        params, opts, norm, rb, carry, st = chunk(params, opts, norm, rb, carry)
        st = {k: float(v) for k, v in st.items()}
        if not np.isfinite(st["critic_loss"]):
            print("non-finite loss; stopping", flush=True)
            break
        steps = (it + 1) * per_it
        row = dict(it=it, steps=steps, sps=steps / (time.time() - t0), **st)
        if wr is None:
            wr = csv.DictWriter(log, fieldnames=list(row))
            wr.writeheader()
        wr.writerow(row)
        log.flush()
        if it % 10 == 0 or it == n_it - 1:
            top = sorted(((k[2:], v) for k, v in st.items() if k.startswith("r.")), key=lambda kv: -abs(kv[1]))[:3]
            print(f"it {it:4d}  {steps / 1e6:6.1f} M  {row['sps'] / 1e3:4.0f} k/s  r {st['reward']:+.3f}  success {st['success']:.2f}  "
                  f"fall {st['fall']:.2f}  Q {st['q']:+.2f}  critic {st['critic_loss']:.3f}  alpha {st['alpha']:.4f}  "
                  f"entropy {st['entropy']:+.1f}  | " + "  ".join(f"{k} {v:+.2f}" for k, v in top), flush=True)
        if it % ev_every == 0 or it == n_it - 1:
            ok = np.asarray(evaluate(params["actor"], norm, jax.random.PRNGKey(10_000 + it)))
            kind, level = np.asarray(bank_ev["kind"]), np.asarray(bank_ev["level"])
            table = {f"{k},{lv}": float(ok[(kind == k) & (level == lv)].mean()) for k in np.unique(kind) for lv in np.unique(level)}
            with open(os.path.join(HERE, "logs", f"{o['name']}_eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(it=it, steps=steps, success=float(ok.mean()), table=table)) + "\n")
            print(f"   eval {steps / 1e6:6.1f} M  held-out success {ok.mean():.3f}", flush=True)
            snap = os.path.join(HERE, "policies", "snaps", o["name"], f"{steps // 1_000_000:05d}M")
            export(params, norm, task, snap + ".npz")
            if o["teacher"]:
                pickle.dump(dict(params=dict(actor=jax.tree_util.tree_map(np.asarray, params["actor"])), task=o["task"], algo="fastsac",
                                 norm=jax.tree_util.tree_map(np.asarray, norm)), open(snap + ".pkl", "wb"))
    export(params, norm, task, os.path.join(HERE, "policies", f"{o['name']}.npz"))
    pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, params), norm=jax.tree_util.tree_map(np.asarray, norm), task=o["task"],
                     algo="fastsac"),
                open(os.path.join(HERE, "policies", f"{o['name']}.pkl"), "wb"))
    print("saved", o["name"], flush=True)


def export(params, norm, task, path):
    """policy.py format: the actor's mean head only, tanh output (squash=1). A teacher is marked kind=teacher
    (policy.py refuses it: it needs the privileged state)."""
    a = jax.tree_util.tree_map(np.asarray, params["actor"])
    kind, extra = "mlp", {}
    if isinstance(a, dict):
        a, kind = a["actor"], "teacher"
    if RES:                                     # the correction's layers; the base's go in the usual keys
        extra = {f"res_{i}_{n}": x for i, (Wt, b) in enumerate(a) for n, x in (("weight", Wt[:12] if i == len(a) - 1 else Wt),
                                                                             ("bias", b[:12] if i == len(a) - 1 else b))}
        a, kind = jax.tree_util.tree_map(np.asarray, RES["params"]), "residual"
        norm = RES["norm"]
        extra["res_scale"] = float(RES["scale"])
    out = {f"mlp_extractor_policy_net_{2 * i}_weight": Wt for i, (Wt, b) in enumerate(a[:-1])}
    out.update({f"mlp_extractor_policy_net_{2 * i}_bias": b for i, (Wt, b) in enumerate(a[:-1])})
    nm = jax.tree_util.tree_map(np.asarray, norm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, obs_mean=nm["mean"][:task.n_obs], obs_var=nm["var"][:task.n_obs], clip_obs=10.0, epsilon=1e-8,
             action_net_weight=a[-1][0][:12], action_net_bias=a[-1][1][:12], task=task.name, kind=kind, squash=1,
             look=int(task.look), **out, **extra)


if __name__ == "__main__":
    main()
