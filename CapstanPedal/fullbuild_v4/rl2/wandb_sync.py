"""Send runs to Weights & Biases from their logs (train.py and distill.py write CSVs, not wandb, so training needs
no account and no network). Run it next to the training, on cake, once logged in (`wandb login`):

    python wandb_sync.py terrain_teacher_s0 terrain_student_s0 terrain_blind_s0 --follow   # live: every 30 s until each run says "saved"
    python wandb_sync.py terrain_uniform_s0                                                 # a finished run, at once

Each run becomes a wandb run of that name (project capstan-rl2, group = the task): every CSV column against
env steps (reward, success, falls, every reward term r.<term>, losses, noise), the held-out success and its
per-terrain-type table, the run's options as its config, and the final CPU evaluation if play.py made one.
Rerunning a name resumes the same wandb run and only sends rows it has not sent.
"""
import csv
import json
import sys
import time
from pathlib import Path

import wandb

import tasks as T

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"


class Run:
    def __init__(self, name, project):
        opts = json.load(open(LOGS / f"{name}_opts.json")) if (LOGS / f"{name}_opts.json").exists() else {}
        kind = "distill" if "teacher" in opts and isinstance(opts["teacher"], str) else opts.get("task", "")
        self.name = name
        self.run = wandb.init(project=project, name=name, id=name, resume="allow", group=opts.get("task", kind) or None,
                              job_type="distill" if kind == "distill" else ("teacher" if opts.get("teacher") else "ppo"),
                              config={k: v for k, v in opts.items() if not isinstance(v, dict)}, reinit="create_new")
        self.run.define_metric("steps")
        self.run.define_metric("*", step_metric="steps")
        self.sent = int(self.run.summary.get("_rows", 0))
        self.sent_ev = int(self.run.summary.get("_evals", 0))
        self.done = False

    def poll(self):
        rows = list(csv.DictReader(open(LOGS / f"{self.name}.csv"))) if (LOGS / f"{self.name}.csv").exists() else []
        for r in rows[self.sent:]:
            self.run.log({("train/" + k if k not in ("steps", "it") else k): float(v) for k, v in r.items() if v != ""})
        self.sent = len(rows)
        ev = [json.loads(x) for x in open(LOGS / f"{self.name}_eval.jsonl")] if (LOGS / f"{self.name}_eval.jsonl").exists() else []
        for e in ev[self.sent_ev:]:
            out = {"steps": float(e["steps"]), "eval/held_out_success": e["success"]}
            by_kind = {}
            for key, v in e["table"].items():
                k, _ = key.split(",")
                by_kind.setdefault(int(k), []).append(v)
            if len(by_kind) > 1:
                out.update({f"eval/{T.TERRAINS[k]}": sum(v) / len(v) for k, v in by_kind.items()})
            self.run.log(out)
        self.sent_ev = len(ev)
        self.run.summary.update({"_rows": self.sent, "_evals": self.sent_ev})
        out = LOGS / f"{self.name}.out"
        self.done = out.exists() and "saved " in out.read_text()[-2000:]
        for f in (f"{self.name}_cpu_eval.json", f"{self.name}_best_cpu_eval.json"):
            if (LOGS / f).exists():
                self.run.summary[f.replace(f"{self.name}_", "").replace(".json", "")] = json.load(open(LOGS / f))["success"]
        return self.done


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict((a[2:].split("=") + ["1"])[:2] for a in sys.argv[1:] if a.startswith("--"))
    project = opts.get("project", "capstan-rl2")
    runs = {}
    while True:
        for n in names:
            if n not in runs and (LOGS / f"{n}.csv").exists():
                runs[n] = Run(n, project)
            if n in runs and not runs[n].done and runs[n].poll() and "follow" in opts:
                runs[n].run.finish()
        if "follow" not in opts or all(n in runs and runs[n].done for n in names):
            break
        time.sleep(30)
    for r in runs.values():
        if not r.done or "follow" not in opts:
            r.run.finish()
    for r in runs.values():
        print(f"{r.name}: {r.run.url}")


if __name__ == "__main__":
    main()
