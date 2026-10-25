"""Reward-term report for a training run: what each term contributed per step, early and late, and its share of
the total. Use it to see which term the robot is really optimising before you change a weight.

    python terms.py <name> [<name2>]          (logs/<name>.csv; a second run is shown alongside, e.g. after a change)
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    rows = list(csv.DictReader(open(HERE / "logs" / f"{name}.csv")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def window(x, a, b):
    n = len(x)
    return x[int(a * n):max(int(b * n), int(a * n) + 1)].mean()


def report(name):
    d = load(name)
    terms = [k for k in d if k.startswith("r.")]
    late = {k: window(d[k], 0.9, 1.0) for k in terms}
    tot = sum(abs(v) for v in late.values()) or 1.0
    print(f"\n{name}: {d['steps'][-1] / 1e6:.0f} M steps, success {window(d['success'], 0.9, 1):.2f}, "
          f"held-in bank {window(d['bank_success'], 0.9, 1):.2f}")
    print(f"{'term':<14}{'first 10 %':>12}{'last 10 %':>12}{'share':>8}   (weighted reward per step, before the x0.1 scale)")
    for k in sorted(terms, key=lambda k: -abs(late[k])):
        print(f"{k[2:]:<14}{window(d[k], 0, 0.1):>+12.3f}{late[k]:>+12.3f}{abs(late[k]) / tot:>8.0%}")
    return late


if __name__ == "__main__":
    for n in sys.argv[1:]:
        report(n)
