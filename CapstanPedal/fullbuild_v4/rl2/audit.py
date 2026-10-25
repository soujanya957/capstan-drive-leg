"""Is a policy good enough for the robot? Plays it on the full CPU model under named conditions (nominal, worst-case
sensors, weak robot, slippery, shoved, flow sensor dead) and reports quality, not just pass / fail:

  walk:             rms speed and turn error, falls, swing height, peak torque vs the motor limit,
                    action jerk, cost of transport
  rear, handstand:  success, time to get up, how steady the hold is, falls
  balance:          success, board tilt, how far the robot wanders on the board
  terrain:          success per terrain type at three levels

    python audit.py <name> [--n=20]          -> logs/<name>_audit.json
"""
import json
import sys
from pathlib import Path

import jax.numpy as jp
import numpy as np

import core as C
import env as E
import play
import tasks as T

HERE = Path(__file__).resolve().parent
MASS = 18.73

# per-episode overrides: robot / world (cfg) and sensor errors (aux["sens"])
WORST_SENS = dict(delay=int(round(C.hw.DELAY_MS[1] / 1000 / C.DT)), flow_drop=0.3, flow_scale=C.hw.FLOW_SCALE,
                  gyro_bias=np.full(3, C.hw.GYRO_BIAS), tilt_bias=np.full(2, C.hw.TILT_BIAS))
NOMINAL = dict(fric=1.0, mass_add=0.0, strength=1.0, kp_s=1.0, kd_s=1.0, w0_s=1.0)
CONDS = {
    "nominal": (dict(NOMINAL), dict(delay=int(round(8 / 1000 / C.DT)), flow_drop=0.05, flow_scale=0.0)),
    "worst sensors": (dict(NOMINAL), WORST_SENS),
    "weak robot": (dict(NOMINAL, strength=0.75, w0_s=C.hw.BATTERY_SAG[0], mass_add=2.0), {}),
    "slippery": (dict(NOMINAL, fric=0.4), {}),
    "all of the above": (dict(NOMINAL, strength=0.75, w0_s=C.hw.BATTERY_SAG[0], mass_add=2.0, fric=0.4), WORST_SENS),
    "flow sensor dead": (dict(NOMINAL), dict(flow_drop=1.0)),
}


def st(ce):
    d = ce.d
    return C.state(ce.L, jp.asarray(d.qpos), jp.asarray(d.qvel), jp.asarray(d.site_xpos))


def apply(ce, cfg_over, sens_over):
    ce.cfg = dict(ce.cfg, **{k: jp.asarray(v, jp.float32) for k, v in cfg_over.items() if k in ce.cfg})
    sens = dict(ce.aux["sens"])
    for k, v in sens_over.items():
        sens[k] = jp.asarray(v, sens[k].dtype)
    ce.aux = dict(ce.aux, sens=sens)


def walk_audit(ce, pol, n):
    out = {}
    for cname, (cfg_o, sens_o) in CONDS.items():
        acc = dict(lin=[], yaw=[], fell=0, swing=[], tau=[], jerk=[], cot=[])
        for ep in range(n):
            bank = ce.task.bank(1, 900 + ep)
            cfg = play.row(bank, 0)
            cfg.update(push_t=np.int32(10 ** 6), **{k: np.float32(v) for k, v in cfg_o.items()})
            obs = ce.reset(cfg, seed=900 + ep)
            apply(ce, {}, sens_o)
            # mid-run shove in every episode, sideways, 0.5 m/s
            cmd, a_prev, el, ey, work, dist, fell = np.zeros(3), None, [], [], 0.0, 0.0, False
            peak_z = np.zeros(4)
            t = 0
            for seconds, target in play.WALK_SCRIPT:
                for _ in range(int(seconds * 50)):
                    cmd += np.clip(np.array(target) - cmd, -0.04, 0.04)
                    ce.aux = dict(ce.aux, cmd=jp.asarray(cmd, jp.float32), t=jp.int32(0))
                    if t == 600:
                        ce.d.qvel[1] += 0.5
                    a = pol(obs)
                    obs, r, terms, done, ok, fallen, err = ce.step(a)
                    d, L = ce.d, ce.L
                    s = st(ce)
                    v = np.asarray(s["v"])
                    if t > 75 and not (580 < t < 680):                     # skip the start and the shove
                        el.append((-v[0] - cmd[0]) ** 2 + (v[1] - cmd[1]) ** 2)
                        ey.append((float(s["w"][2]) - cmd[2]) ** 2)
                    feet = np.asarray(s["feet"])
                    low = feet[:, 2] < C.G.FOOT_R + 0.01
                    peak_z = np.where(low, 0.0, np.maximum(peak_z, feet[:, 2] - C.G.FOOT_R))
                    if t % 25 == 0:
                        acc["swing"].append(float(np.max(peak_z)))
                    tau = d.ctrl[L.act]
                    acc["tau"].append(float(np.max(np.abs(tau)) / C.M.TAU))
                    w = np.asarray(s["vel"])
                    work += float(np.sum(np.abs(tau * w))) / 50
                    dist += float(np.linalg.norm(np.asarray(s["v_world"])[:2])) / 50
                    if a_prev is not None:
                        acc["jerk"].append(float(np.mean(np.abs(a - a_prev))))
                    a_prev = a
                    fell |= fallen
                    t += 1
                    if fallen:
                        break
                if fell:
                    break
            acc["fell"] += fell
            acc["lin"].append(np.sqrt(np.mean(el)) if el else np.nan)
            acc["yaw"].append(np.sqrt(np.mean(ey)) if ey else np.nan)
            acc["cot"].append(work / (MASS * 9.81 * max(dist, 1e-3)))
        out[cname] = dict(fell=acc["fell"] / n, rms_lin=float(np.nanmean(acc["lin"])), rms_yaw=float(np.nanmean(acc["yaw"])),
                          swing_cm=100 * float(np.median(acc["swing"])), tau_peak=float(np.percentile(acc["tau"], 99)),
                          jerk=float(np.mean(acc["jerk"])), cot=float(np.mean(acc["cot"])))
        o = out[cname]
        print(f"{cname:<17} fell {o['fell']:5.0%}  rms speed {o['rms_lin']:.3f} m/s  turn {o['rms_yaw']:.3f} rad/s  "
              f"swing {o['swing_cm']:4.1f} cm  torque p99 {o['tau_peak']:.0%} of limit  jerk {o['jerk']:.3f}  CoT {o['cot']:.2f}", flush=True)
    return out


def episodes(ce, pol, n, level, cfg_o, sens_o, task):
    bank = task.bank(400, 5151)
    lv = np.asarray(bank["level"])
    idx = np.flatnonzero(lv == level)[:n]
    res = dict(ok=0, fell=0, rise=[], wobble=[], extra=[])
    for j, i in enumerate(idx):
        cfg = play.row(bank, int(i))
        cfg.update({k: np.float32(v) for k, v in cfg_o.items()})
        obs = ce.reset(cfg, seed=5000 + j)
        apply(ce, {}, sens_o)
        fz, rise = [], None
        for t in range(task.EP):
            obs, r, terms, done, ok, fallen, err = ce.step(pol(obs))
            s = st(ce)
            if task.name in ("rear", "handstand"):
                f = -float(s["R"][2, 0]) * task.UP
                fz.append(f)
                if rise is None and f >= task.F_OK and t >= int(cfg["t_on"]):
                    rise = (t - int(cfg["t_on"])) / 50
            elif task.name == "balance":
                n_ = np.asarray(s["board_R"])[:, 2]
                fz.append(float(np.degrees(np.arccos(np.clip(n_[2], -1, 1)))))
                res["extra"].append(float(np.linalg.norm(np.asarray(s["p"] - s["board_p"])[:2])))
            if done:
                res["ok"] += ok
                res["fell"] += fallen
                break
        if task.name in ("rear", "handstand"):
            if rise is not None:
                res["rise"].append(rise)
            res["wobble"].append(float(np.degrees(np.std(np.arcsin(np.clip(fz[-100:], -1, 1))))) if len(fz) >= 100 else np.nan)
        else:
            res["wobble"].append(float(np.mean(fz)))
    m = len(idx)
    return dict(success=res["ok"] / m, fell=res["fell"] / m, rise_s=float(np.mean(res["rise"])) if res["rise"] else None,
                wobble_deg=float(np.nanmean(res["wobble"])), wander_cm=100 * float(np.mean(res["extra"])) if res["extra"] else None)


def skill_audit(ce, pol, n, task):
    out = {}
    for cname, (cfg_o, sens_o) in CONDS.items():
        for level in (0, 5, 9):
            o = episodes(ce, pol, n, level, cfg_o, sens_o, task)
            out[f"{cname} / level {level}"] = o
            extra = (f"rise {o['rise_s']:.2f} s  hold wobble {o['wobble_deg']:.1f} deg" if task.name != "balance"
                     else f"board tilt {o['wobble_deg']:.1f} deg  wander {o['wander_cm']:.1f} cm")
            print(f"{cname:<17} level {level}  success {o['success']:5.0%}  fell {o['fell']:5.0%}  {extra}", flush=True)
    return out


def terrain_audit(ce, pol, n, task):
    bank = task.bank(2000, 5151)
    kind, level = np.asarray(bank["kind"]), np.asarray(bank["level"])
    out = {}
    for k, kn in enumerate(T.TERRAINS):
        row = []
        for lv in (1, 4, 7):
            idx = np.flatnonzero((kind == k) & (level == lv))[:n]
            ok = 0
            for j, i in enumerate(idx):
                obs = ce.reset(play.row(bank, int(i)), seed=6000 + j)
                for _ in range(task.EP):
                    obs, r, terms, done, s_ok, fallen, err = ce.step(pol(obs))
                    if done:
                        ok += s_ok
                        break
            row.append(ok / max(len(idx), 1))
        out[kn] = row
        print(f"{kn:<12} level 1 {row[0]:4.0%}   level 4 {row[1]:4.0%}   level 7 {row[2]:4.0%}", flush=True)
    return out


def main():
    name = sys.argv[1]
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[2:] if a.startswith("--"))
    n = int(opts.get("n", 20))
    pol, task = play.load(name)
    ce = E.CpuEnv(task)
    print(f"audit {name} ({task.name}), {n} episodes per cell, full CPU model")
    if task.name == "walk":
        out = walk_audit(ce, pol, n)
    elif task.name == "terrain":
        out = terrain_audit(ce, pol, n, task)
    else:
        out = skill_audit(ce, pol, n, task)
    json.dump(out, open(HERE / "logs" / f"{name}_audit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
