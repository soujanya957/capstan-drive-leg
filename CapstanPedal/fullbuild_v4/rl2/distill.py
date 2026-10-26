"""Teacher -> student: train a blind policy to act like a teacher that sees the course (train.py --teacher=1).

The teacher reads the privileged state through an encoder to a latent z (16 numbers: what the ground ahead looks
like, how slippery it is, how heavy the robot is ...). The student cannot see any of that. It reads the last
N_LONG = 50 observation frames (1 s of what its own sensors gave: joints, IMU, flow, range) and estimates z from them
(an "adaptation module", Kumar et al. 2021, RMA), then acts through a copy of the teacher's policy network. Lee et
al. 2020 trained ANYmal over rough terrain this way.

Training is imitation, not RL: no reward. DAgger (Ross et al. 2011): the student drives the robot (early on the
teacher drives some steps, --beta of the run, falling to 0), so it visits the states its own mistakes lead to, and
at every step the teacher says what it would have done there. Loss = |a_student - a_teacher|^2 + z_w |z_hat - z|^2.

    python distill.py --teacher=terrain_teacher_s0 --name=terrain_student_s0 --steps=200
        --snap=best (default): the teacher snapshot with the best held-out success; --snap=last: policies/<teacher>.pkl

Writes policies/<name>.npz (a student: policy.py runs it, numpy only, ~2 MB; env.task_for gives it the 50-frame
observation), snapshots policies/snaps/<name>/<steps>M.npz with logs/<name>_eval.jsonl (pick_best.py, progress.py),
logs/<name>.csv.
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
OPT = dict(teacher="", snap="best", name="", steps=200.0, envs=4096, cfgs=1040, unroll=24, epochs=4, minibatches=4,
           lr=5e-4, beta=0.2, z_w=1.0, eval_every=50, eval_n=4160, save=50, seed=0, bank_seed=0, nacon=24, njmax=160)
PROJ, ENC = 16, (256, 64)                     # per-frame projection, then the history encoder to the latent


def options():
    o = dict(OPT)
    for a in sys.argv[1:]:
        k, v = a.lstrip("-").split("=", 1)
        o[k] = type(OPT[k])(v)
    o["name"] = o["name"] or o["teacher"].replace("teacher", "student")
    return o


def load_teacher(o):
    snaps = os.path.join(HERE, "policies", "snaps", o["teacher"])
    path = os.path.join(HERE, "policies", f"{o['teacher']}.pkl")
    if o["snap"] == "best":
        ev = [json.loads(line) for line in open(os.path.join(HERE, "logs", f"{o['teacher']}_eval.jsonl"))]
        for e in sorted(ev, key=lambda e: -e["success"]):
            p = os.path.join(snaps, f"{e['steps'] // 1_000_000:05d}M.pkl")
            if os.path.exists(p):
                path = p
                print(f"teacher {o['teacher']}: snapshot {e['steps'] / 1e6:.0f} M, held-out success {e['success']:.3f}", flush=True)
                break
    ck = pickle.load(open(path, "rb"))
    assert "enc" in ck["params"], f"{path} is not a teacher (train.py --teacher=1)"
    return jax.tree_util.tree_map(jp.asarray, ck["params"]), jax.tree_util.tree_map(jp.asarray, ck["norm"]), ck["task"]


def student_act(sp, frame_norm, obs_norm, ob):
    """The student: (action, latent estimate) from the 50-frame observation."""
    fr = Tr.normalise(frame_norm, ob.reshape(ob.shape[:-1] + (C.N_LONG, C.N_FRAME)))
    Wt, b = sp["proj"]
    h = jax.nn.elu(fr @ Wt.T + b).reshape(ob.shape[:-1] + (C.N_LONG * PROJ,))
    z = Tr.mlp(sp["enc"], h)
    x = Tr.normalise(obs_norm, ob[..., -C.N_OBS:])
    return Tr.mlp(sp["actor"], jp.concatenate([x, z], -1)), z


def main():
    o = options()
    tp, tnorm, task_name = load_teacher(o)
    task = E.make_task(task_name)
    task.obs_hist = C.N_LONG
    n, n_cfg = o["envs"], o["cfgs"]
    env = E.GpuEnv(task, n, o["nacon"], o["njmax"])
    bank = task.bank(n_cfg, o["bank_seed"])
    obs_norm, _ = Tr.split_norm(tnorm, C.N_OBS)
    frame_norm = dict(mean=obs_norm["mean"][-C.N_FRAME:], var=obs_norm["var"][-C.N_FRAME:])   # the newest frame's statistics

    def teacher(ob, pv):
        return Tr.actor_mean(tp, tnorm, ob[..., -C.N_OBS:], pv)

    key = jax.random.PRNGKey(o["seed"])
    key, k1, k2 = jax.random.split(key, 3)
    sp = dict(proj=Tr.mlp_init(k1, (C.N_FRAME, PROJ), 1.0)[0], enc=Tr.mlp_init(k2, (C.N_LONG * PROJ,) + ENC + (Tr.LATENT,), 1.0),
              actor=tp["actor"])                                         # the teacher's policy network, fine-tuned
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(o["lr"]))
    opt_state = tx.init(sp)
    step_v = jax.vmap(env.step, in_axes=(env.in_axes, 0, 0, 0, None, 0))
    evaluate, bank_ev = Tr.make_evaluate(task, o, lambda p, nm, ob, pv: student_act(p, frame_norm, obs_norm, ob)[0])

    def rollout(carry, _, sp, beta):
        d, aux, mp, obs, priv, key = carry
        key, k_s, k_r, k_c, k_b = jax.random.split(key, 5)
        a_s, _ = student_act(sp, frame_norm, obs_norm, obs)
        a_t, z_t = teacher(obs, priv)
        a = jp.where((jax.random.uniform(k_b, (n,)) < beta)[:, None], a_t, a_s)
        d2, aux2, obs2, priv2, r, terms, done, trunc, success, fallen, err = step_v(env.mx.tree_replace(mp), d, aux, a, bank,
                                                                                    jax.random.split(k_s, n))
        new_cfg = jax.random.randint(k_c, (n,), 0, n_cfg)
        d2, aux2, mp, obs2, priv2 = env.respawn(bank, d2, aux2, mp, obs2, priv2, done, new_cfg, k_r)
        tr = dict(obs=obs, a_t=a_t, z_t=z_t, r=r, done=done, success=success & done, fallen=fallen & done, err=err)
        return (d2, aux2, mp, obs2, priv2, key), tr

    def loss_fn(sp, b):
        a, z = student_act(sp, frame_norm, obs_norm, b["obs"])
        la = jp.mean(jp.sum((a - b["a_t"]) ** 2, -1))
        lz = jp.mean(jp.sum((z - b["z_t"]) ** 2, -1))
        return la + o["z_w"] * lz, (la, lz)

    @jax.jit
    def iteration(sp, opt_state, carry, beta):
        carry, tr = jax.lax.scan(lambda c, x: rollout(c, x, sp, beta), carry, None, length=o["unroll"])
        batch = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), dict(obs=tr["obs"], a_t=tr["a_t"], z_t=tr["z_t"]))
        total = batch["obs"].shape[0]

        def epoch(st, k):
            perm = jax.random.permutation(k, total)
            mbs = jax.tree_util.tree_map(lambda x: x[perm].reshape((o["minibatches"], -1) + x.shape[1:]), batch)

            def mb(st, b):
                p, os_ = st
                (_, (la, lz)), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
                upd, os_ = tx.update(g, os_, p)
                return (optax.apply_updates(p, upd), os_), (la, lz)
            return jax.lax.scan(mb, st, mbs)
        key = carry[-1]
        key, k = jax.random.split(key)
        (sp, opt_state), (la, lz) = jax.lax.scan(epoch, (sp, opt_state), jax.random.split(k, o["epochs"]))
        nd = jp.maximum(tr["done"].sum(), 1)
        stats = dict(loss_a=la.mean(), loss_z=lz.mean(), reward=tr["r"].mean(), episodes=tr["done"].sum(),
                     success=tr["success"].sum() / nd, fall=tr["fallen"].sum() / nd, err=tr["err"].mean())
        return sp, opt_state, carry[:-1] + (key,), stats

    key, k0, k1 = jax.random.split(key, 3)
    carry = env.start(bank, jax.random.randint(k0, (n,), 0, n_cfg), k1) + (key,)
    per_it = n * o["unroll"]
    n_it = int(o["steps"] * 1e6 // per_it)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    json.dump(o, open(os.path.join(HERE, "logs", f"{o['name']}_opts.json"), "w"), indent=1)
    log = open(os.path.join(HERE, "logs", f"{o['name']}.csv"), "w", newline="")
    wr = None
    print(f"distil {o['teacher']} -> {o['name']}: {n} worlds, {n_it} iterations of {per_it} steps", flush=True)
    t0 = time.time()
    for it in range(n_it):
        beta = max(0.0, 1.0 - it / max(o["beta"] * n_it, 1))
        sp, opt_state, carry, st = iteration(sp, opt_state, carry, jp.float32(beta))
        st = {k: float(v) for k, v in st.items()}
        steps = (it + 1) * per_it
        row = dict(it=it, steps=steps, sps=steps / (time.time() - t0), beta=beta, **st)
        if wr is None:
            wr = csv.DictWriter(log, fieldnames=list(row))
            wr.writeheader()
        wr.writerow(row)
        log.flush()
        if it % 10 == 0 or it == n_it - 1:
            print(f"it {it:4d}  {steps / 1e6:6.1f} M  {row['sps'] / 1e3:4.0f} k/s  beta {beta:.2f}  action loss {st['loss_a']:.4f}  "
                  f"latent loss {st['loss_z']:.4f}  success {st['success']:.2f}  fall {st['fall']:.2f}", flush=True)
        if it % o["eval_every"] == 0 or it == n_it - 1:
            ok = np.asarray(evaluate(sp, tnorm, jax.random.PRNGKey(10_000 + it)))
            kind, level = np.asarray(bank_ev["kind"]), np.asarray(bank_ev["level"])
            table = {f"{k},{lv}": float(ok[(kind == k) & (level == lv)].mean()) for k in np.unique(kind) for lv in np.unique(level)}
            with open(os.path.join(HERE, "logs", f"{o['name']}_eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(it=it, steps=steps, success=float(ok.mean()), table=table)) + "\n")
            print(f"   eval {steps / 1e6:6.1f} M  held-out success {ok.mean():.3f}", flush=True)
            export(sp, obs_norm, frame_norm, task_name, os.path.join(HERE, "policies", "snaps", o["name"], f"{steps // 1_000_000:05d}M.npz"))
        if (it + 1) % o["save"] == 0 or it == n_it - 1:
            export(sp, obs_norm, frame_norm, task_name, os.path.join(HERE, "policies", f"{o['name']}.npz"))
    print("saved", o["name"], flush=True)


def export(sp, obs_norm, frame_norm, task_name, path):
    """A student for policy.py: the policy network in the same keys as a plain policy, plus the history encoder."""
    sp = jax.tree_util.tree_map(np.asarray, sp)
    a = sp["actor"]
    out = {f"mlp_extractor_policy_net_{2 * i}_weight": Wt for i, (Wt, b) in enumerate(a[:-1])}
    out.update({f"mlp_extractor_policy_net_{2 * i}_bias": b for i, (Wt, b) in enumerate(a[:-1])})
    out.update({f"enc_{i}_weight": Wt for i, (Wt, b) in enumerate(sp["enc"])})
    out.update({f"enc_{i}_bias": b for i, (Wt, b) in enumerate(sp["enc"])})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, kind="student", n_hist=C.N_LONG, n_frame=C.N_FRAME, obs_mean=np.asarray(obs_norm["mean"]), obs_var=np.asarray(obs_norm["var"]),
             frame_mean=np.asarray(frame_norm["mean"]), frame_var=np.asarray(frame_norm["var"]), proj_weight=sp["proj"][0],
             proj_bias=sp["proj"][1], clip_obs=10.0, epsilon=1e-8, action_net_weight=a[-1][0], action_net_bias=a[-1][1],
             task=task_name, **out)


if __name__ == "__main__":
    main()
