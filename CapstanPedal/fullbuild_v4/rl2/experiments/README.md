# Experiment batches

One run per line: `<name> <train.py options>`. Run a batch on cake with `sh run_queue.sh experiments/<batch>.list`
(it skips runs whose policy already exists, so a list can be extended while it runs). Results: `../sgs-lessons.md`.

| Batch | Date | What it tested |
|---|---|---|
| `queue1.list` | 2026-10-24 | First rl2 runs: walk, rear, handstand, balance; terrain with uniform, SGS and SGS-with-prior, 3 seeds each |
| `queue2.list` | 2026-10-24 | walk2: walk retrained with flow-sensor failures (`hw.FLOW_DEAD`) |
| `queue3.list` | 2026-10-24 | After the Codex review: SGS bookkeeping fix (3+3 seeds), balance with corrected sensors, handstand reward reshaped |
| `queue4.list` | 2026-10-24 | Noise cap + entropy anneal, level curriculum (`--sampler=curr`), constraints as terminations (`--cat`) |
| `queue5.list` | 2026-10-26 | Terrain teacher–student: a teacher that sees a height scan of the course, its blind student (`distill.py`), and a blind PPO control with the same critic |
| `queue6.list` | 2026-10-26 | FastSAC (`train_sac.py`) vs PPO at about equal wall-clock (150 M vs 400 M steps): terrain 1-step vs 3-step returns, 3+3 seeds with PPO controls, handstand, walk |
| `queue7.list` | 2026-10-26 | FastSAC with the paper's settings (averaged critics, entropy target 0, std cap 1, gamma 0.97, AdamW): terrain, 3 seeds. 88.2 % vs PPO's 70.7 % (sgs-lessons.md, Experiment 6) |
