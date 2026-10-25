# Experiment batches

One run per line: `<name> <train.py options>`. Run a batch on cake with `sh run_queue.sh experiments/<batch>.list`
(it skips runs whose policy already exists, so a list can be extended while it runs). Results: `../sgs-lessons.md`.

| Batch | Date | What it tested |
|---|---|---|
| `queue1.list` | 2026-10-24 | First rl2 runs: walk, rear, handstand, balance; terrain with uniform, SGS and SGS-with-prior, 3 seeds each |
| `queue2.list` | 2026-10-24 | walk2: walk retrained with flow-sensor failures (`hw.FLOW_DEAD`) |
| `queue3.list` | 2026-10-24 | After the Codex review: SGS bookkeeping fix (3+3 seeds), balance with corrected sensors, handstand reward reshaped |
| `queue4.list` | 2026-10-24 | Noise cap + entropy anneal, level curriculum (`--sampler=curr`), constraints as terminations (`--cat`) |
