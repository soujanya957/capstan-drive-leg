"""A task on a model: thousands of worlds on the GPU (GpuEnv: MuJoCo Warp through MJX, the "physics" model), or one
world on the CPU in the full model (CpuEnv: every mesh, rotors on their joints; what play.py and teleop.py use).
Both step a policy action the same way: hold the old targets for the control delay, then the new ones, through the
PD loop and motor model at 500 Hz for 10 physics steps, then the task scores the step and senses the next frame.
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np

import core as C
import tasks as T


DATA_KEYS = ("mocap_pos", "mocap_quat")          # per-world values that live in the data, not the model


def split(mp):
    return {k: v for k, v in mp.items() if k not in DATA_KEYS}, {k: v for k, v in mp.items() if k in DATA_KEYS}


def make_task(name, weights=None):
    return T.TASKS[name](weights)


def task_for(path):
    """The task a policy file was trained on, observing as the policy expects (a distilled student reads N_LONG frames)."""
    z = np.load(path)
    task = make_task(str(z["task"]))
    task.obs_hist = int(z["n_hist"]) if "n_hist" in z.files else C.N_HIST
    return task


class GpuEnv:
    def __init__(self, task, n_env, nacon=24, njmax=160):
        from mujoco import mjx
        self.mjx = mjx
        self.task, self.n = task, n_env
        m = self.m = C.build_spec("physics", task.extras).compile()
        self.L = C.Layout(m, task.extras)
        task.bind(self.L)
        self.mx = mjx.put_model(m, impl="warp")
        self.nacon, self.njmax = nacon, njmax
        probe = task.bank(len(T.TERRAINS), 0)
        keys = [k for k in task.mparams({k: v[0] for k, v in probe.items()}) if k not in DATA_KEYS]
        self.in_axes = jax.tree_util.tree_map(lambda x: None, self.mx).tree_replace({k: 0 for k in keys})

    def make_data(self):
        mjx, m = self.mjx, self.m
        return jax.vmap(lambda _: mjx.make_data(m, impl="warp", naconmax=self.nacon * self.n, njmax=self.njmax))(jp.arange(self.n))

    def reset_slot(self, bank, i, key):
        return self.task.reset({k: v[i] for k, v in bank.items()}, i, key)

    def first_obs(self, d, aux, bank, key):
        cfg = {k: v[aux["cfg"]] for k, v in bank.items()}
        return self.task.first_obs(C.state(self.L, d.qpos, d.qvel, d.site_xpos), aux, cfg, key)

    def first_priv(self, d, aux, bank):
        cfg = {k: v[aux["cfg"]] for k, v in bank.items()}
        return self.task.priv_now(C.state(self.L, d.qpos, d.qvel, d.site_xpos), aux, cfg)

    def respawn(self, bank, d, aux, mp, obs, priv, done, cfg, key, **aux_set):
        """Start a new episode in the worlds that are done (all n), on configurations cfg (n bank rows); the others
        keep going. aux_set: values to put in the new episodes' aux (train.py: cat)."""
        n = self.n
        k_r, k_o = jax.random.split(key)
        q, v, aux_r, mp_r = jax.vmap(self.reset_slot, in_axes=(None, 0, 0))(bank, cfg, jax.random.split(k_r, n))
        aux_r = dict(aux_r, **aux_set)
        mp_r, dp_r = split(mp_r)
        sel = lambda new, old: jp.where(done.reshape((n,) + (1,) * (old.ndim - 1)), new, old)   # noqa: E731
        d = d.replace(qpos=sel(q, d.qpos), qvel=sel(v, d.qvel), qacc_warmstart=sel(jp.zeros_like(d.qacc_warmstart), d.qacc_warmstart),
                      ctrl=sel(jp.zeros_like(d.ctrl), d.ctrl), **{k: sel(x, getattr(d, k)) for k, x in dp_r.items()})
        aux_f, obs_f = jax.vmap(self.first_obs, in_axes=(0, 0, None, 0))(d, aux_r, bank, jax.random.split(k_o, n))
        priv_f = jax.vmap(self.first_priv, in_axes=(0, 0, None))(d, aux_f, bank)
        return (d, jax.tree_util.tree_map(sel, aux_f, aux), jax.tree_util.tree_map(sel, mp_r, mp), sel(obs_f, obs),
                sel(priv_f, priv))

    def start(self, bank, cfg, key, **aux_set):
        """Every world's first episode: (d, aux, mp, obs, priv)."""
        d = self.make_data()
        k1, k2 = jax.random.split(key)
        q, v, aux, mp = jax.vmap(self.reset_slot, in_axes=(None, 0, 0))(bank, cfg, jax.random.split(k1, self.n))
        aux = dict(aux, **aux_set)
        mp, dp = split(mp)
        d = d.replace(qpos=q, qvel=v, **dp)
        aux, obs = jax.vmap(self.first_obs, in_axes=(0, 0, None, 0))(d, aux, bank, jax.random.split(k2, self.n))
        return d, aux, mp, obs, jax.vmap(self.first_priv, in_axes=(0, 0, None))(d, aux, bank)

    def step(self, mx, d, aux, action, bank, key):
        L, task, mjx = self.L, self.task, self.mjx
        cfg = {k: v[aux["cfg"]] for k, v in bank.items()}
        a = jp.clip(action, -1.0, 1.0)
        target = C.targets(jp, jp.asarray(L.default), a)
        qvel, aux = task.begin(aux, cfg, d.qvel)
        d = d.replace(qvel=qvel)
        delay = aux["sens"]["delay"]

        def sub(d, i):
            pos, vel = d.qpos[L.qa], d.qvel[L.va]
            pos, vel = pos.at[2::3].add(pos[1::3]), vel.at[2::3].add(vel[1::3])
            tgt = jp.where(i < delay, aux["prev_target"], target)
            tau = C.torque(jp, pos, vel, tgt, aux["kp"], aux["kd"], aux["strength"], aux["w0"], L.isA)
            d = mjx.step(mx, d.replace(ctrl=d.ctrl.at[L.act].set(tau)))
            return d, jp.abs(tau)
        d, taus = jax.lax.scan(sub, d, jp.arange(C.DECIM))
        aux = dict(aux, prev_target=target)
        s = C.state(L, d.qpos, d.qvel, d.site_xpos)
        return (d,) + task.end(s, aux, cfg, a, taus.mean(0), key)


class CpuEnv:
    """One world in the full model. reset(cfg) takes one configuration (a row of a bank), step(action) returns
    (obs, reward, terms, done, success, fallen, err)."""

    def __init__(self, task, kind="full"):
        self.task = task
        self.m = C.build_spec(kind, task.extras).compile()
        self.d = mujoco.MjData(self.m)
        self.L = C.Layout(self.m, task.extras)
        task.bind(self.L)
        self.cpu = jax.devices("cpu")[0]
        L = self.L

        def post(qpos, qvel, site_xpos, aux, cfg, a, tau, key):
            return task.end(C.state(L, qpos, qvel, site_xpos), aux, cfg, a, tau, key)

        def first(qpos, qvel, site_xpos, aux, cfg, key):
            return task.first_obs(C.state(L, qpos, qvel, site_xpos), aux, cfg, key)
        self._reset = jax.jit(task.reset)
        self._begin = jax.jit(task.begin)
        self._post = jax.jit(post)
        self._first = jax.jit(first)
        self.key = jax.random.PRNGKey(0)

    def _k(self):
        self.key, k = jax.random.split(self.key)
        return k

    def reset(self, cfg, seed=None):
        with jax.default_device(self.cpu):
            if seed is not None:
                self.key = jax.random.PRNGKey(seed)
            cfg = {k: jp.asarray(v) for k, v in cfg.items()}
            qpos, qvel, aux, mp = self._reset(cfg, jp.int32(0), self._k())
            m, d = self.m, self.d
            mp, dp = split(mp)
            for k, v in mp.items():
                getattr(m, k)[:] = np.asarray(v)
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:] = np.asarray(qpos), np.asarray(qvel)
            for k, v in dp.items():
                getattr(d, k)[:] = np.asarray(v)
            mujoco.mj_forward(m, d)
            self.aux, self.cfg = aux, cfg
            self.aux, obs = self._first(d.qpos, d.qvel, d.site_xpos, aux, cfg, self._k())
            return np.asarray(obs)

    def step(self, action):
        L, m, d = self.L, self.m, self.d
        with jax.default_device(self.cpu):
            a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
            target = C.targets(np, L.default, a)
            qvel, aux = self._begin(self.aux, self.cfg, d.qvel)
            d.qvel[:] = np.asarray(qvel)
            delay, prev = int(aux["sens"]["delay"]), np.asarray(aux["prev_target"])
            kp, kd, st, w0 = (np.asarray(aux[k]) for k in ("kp", "kd", "strength", "w0"))
            tau_sum = np.zeros(12)
            for i in range(C.DECIM):
                pos, vel = d.qpos[L.qa].copy(), d.qvel[L.va].copy()
                pos[2::3] += pos[1::3]
                vel[2::3] += vel[1::3]
                tau = C.torque(np, pos, vel, prev if i < delay else target, kp, kd, st, w0, L.isA)
                d.ctrl[L.act] = tau
                mujoco.mj_step(m, d)
                tau_sum += np.abs(tau)
            aux = dict(aux, prev_target=jp.asarray(target))
            out = self._post(d.qpos, d.qvel, d.site_xpos, aux, self.cfg, jp.asarray(a), jp.asarray(tau_sum / C.DECIM), self._k())
            self.aux, obs, _, r, terms, done, _, success, fallen, err = out
            return (np.asarray(obs), float(r), {k: float(v) for k, v in terms.items()}, bool(done), bool(success),
                    bool(fallen), float(err))
