"""Smoke test: every task builds, resets and steps on the GPU without NaNs; prints steps/s and reward terms."""
import sys
import time

import jax
import jax.numpy as jp
import numpy as np

import core as C
import env as E

n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
for name in (sys.argv[1].split(",") if len(sys.argv) > 1 else E.T.TASKS):
    task = E.make_task(name)
    env = E.GpuEnv(task, n)
    bank = task.bank(512, 0)
    d = env.make_data()
    key = jax.random.PRNGKey(0)
    cfg = jp.arange(n) % 512
    q, v, aux, mp = jax.vmap(env.reset_slot, in_axes=(None, 0, 0))(bank, cfg, jax.random.split(key, n))
    mp, dp = E.split(mp)
    d = d.replace(qpos=q, qvel=v, **dp)
    aux, obs = jax.vmap(env.first_obs, in_axes=(0, 0, None, 0))(d, aux, bank, jax.random.split(key, n))
    step = jax.jit(jax.vmap(env.step, in_axes=(env.in_axes, 0, 0, 0, None, 0)))
    mxb = env.mx.tree_replace(mp)
    t0 = time.time()
    tot = {}
    falls = 0
    for i in range(60):
        a = jnp_a = jp.zeros((n, 12))
        d, aux, obs, priv, r, terms, done, trunc, succ, fallen, err = step(mxb, d, aux, a, bank, jax.random.split(jax.random.PRNGKey(i), n))
        if i == 0:
            jax.block_until_ready(r)
            t1 = time.time()
        falls += int(fallen.sum())
        for k, x in terms.items():
            tot[k] = tot.get(k, 0) + float(x.mean())
    jax.block_until_ready(r)
    dt = time.time() - t1
    print(f"{name}: obs {obs.shape} priv {priv.shape} finite {bool(jp.all(jp.isfinite(obs)))} r {float(r.mean()):+.3f} "
          f"falls {falls} | {n * 59 * C.DECIM / dt / 1e3:.0f}k phys steps/s ({n * 59 / dt / 1e3:.1f}k policy steps/s) compile {t1 - t0:.0f}s")
    print("   z", np.round(np.asarray(d.qpos[:4, 2]), 3), "terms", {k: round(v / 60, 3) for k, v in tot.items()})
