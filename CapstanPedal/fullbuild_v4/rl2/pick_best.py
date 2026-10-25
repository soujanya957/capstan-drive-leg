"""Pick the deployable snapshot of a run instead of trusting the last one. Training swings: a terrain run's held-out
success moved +-15 points between evaluations 40 M steps apart, and the last iterate is often not the best.

  1. shortlist: the K snapshots with the best GPU held-out success (logs/<name>_eval.jsonl, 4,160 episodes each),
     plus the last one;
  2. selection: each plays a SELECTION bank on the full CPU model (seed 3000, not the 2000 that play.py eval uses);
  3. the winner is copied to policies/<name>_best.npz, then play.py eval <name>_best gives the unbiased test number.

    JAX_PLATFORMS=cpu python pick_best.py <name> [--k=4] [--n=120]     -> logs/<name>_select.json
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import env as E
import play
import policy as P

HERE = Path(__file__).resolve().parent


def score(path, task, ce, bank, n):
    pol = P.Policy(path)
    ok = fell = 0
    for i in range(n):
        obs = ce.reset(play.row(bank, i), seed=3000 + i)
        for _ in range(task.EP):
            obs, r, terms, done, s_ok, fallen, err = ce.step(pol(obs))
            if done:
                ok += s_ok
                fell += fallen
                break
    return ok / n, fell / n


def main():
    name = sys.argv[1]
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[2:] if a.startswith("--"))
    k, n = int(opts.get("k", 4)), int(opts.get("n", 120))
    ev = [json.loads(line) for line in open(HERE / "logs" / f"{name}_eval.jsonl")]
    snaps = {int(p.stem[:-1]): p for p in (HERE / "policies" / "snaps" / name).glob("*M.npz")}
    by_gpu = sorted(ev, key=lambda e: -e["success"])
    picks = []
    for e in by_gpu:
        m = int(e["steps"] // 1_000_000)
        if m in snaps and m not in picks:
            picks.append(m)
        if len(picks) == k:
            break
    last = max(snaps)
    if last not in picks:
        picks.append(last)
    task = E.make_task(str(np.load(snaps[last])["task"]))
    ce = E.CpuEnv(task)
    bank = task.bank(n, 3000)
    gpu = {int(e["steps"] // 1_000_000): e["success"] for e in ev}
    rows = []
    for m in picks:
        s, f = score(snaps[m], task, ce, bank, n)
        rows.append(dict(steps_M=m, gpu=gpu.get(m), cpu_select=s, fell=f))
        print(f"{name} {m:4d} M   GPU held-out {gpu.get(m, float('nan')):.3f}   CPU selection {s:.3f}   fell {f:.3f}", flush=True)
    best = max(rows, key=lambda r: (r["cpu_select"], -r["fell"], r["steps_M"]))
    shutil.copy(snaps[best["steps_M"]], HERE / "policies" / f"{name}_best.npz")
    print(f"{name}: best snapshot {best['steps_M']} M -> policies/{name}_best.npz")
    json.dump(dict(rows=rows, best=best), open(HERE / "logs" / f"{name}_select.json", "w"), indent=1)


if __name__ == "__main__":
    main()
