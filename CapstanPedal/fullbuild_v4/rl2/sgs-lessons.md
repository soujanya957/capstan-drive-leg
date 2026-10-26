# RL on the v4 quadruped: what we did, what worked, and lessons from SGS

The running record of the learned-control work on Capstan v4: every skill trained so far, what worked and what
didn't, and what we learned trying SGS (Zhang et al. 2026, *A Balanced Data Diet*,
[arXiv 2610.12465](https://arxiv.org/abs/2610.12465)). Updated as experiments finish. Training runs on cake
(RTX 4090); results below are from there. Last updated 2026-10-24.

## Summary so far

**Setup.**
- `rl/` (2026-10-23): first version. Flat walking with PPO on thousands of GPU worlds, sensors idealised. Used for
  Experiments 1–2 (SGS vs uniform on walking).
- `rl2/` (2026-10-24): the second version. Policies see only what the built robot will measure (`hw.py`): pi3hat IMU,
  moteus encoders, and a new optical-flow + time-of-flight pair for ground speed and height (added to `../BOM.md`,
  "Sensing"), with their rates, noise, biases, dropouts and a 4–20 ms control delay. Five tasks: walk, terrain,
  rear, handstand, balance. Every result is checked on held-out scenarios in the full CPU model, not on the GPU
  training model.
- Tools added along the way: per-term reward logging (`terms.py`), stress audits (`audit.py`), best-snapshot
  selection (`pick_best.py`), training-progress films (`progress.py`), the all-skills showreel (`showreel.py`, also
  in Instagram story and post sizes).
- An independent Codex review (`codex-review.md`) found two silent bugs (SGS bookkeeping, balance sensors measuring
  through the board) and the simulator-perfect terrain steering. Both bugs fixed and retrained.

**Deployable policies** (best snapshot, full CPU model, 200 held-out episodes; ~1 MB each in `policies/`):

| Skill | Policy | Held-out success | Notes |
|---|---|---|---|
| Walk on joystick commands | `walk2_best` | 100 % | trained with flow-sensor failures; dead flow sensor: 0 % falls (walk: 75 %) |
| Rear up on the hind legs | `rear_best` | 100 % | no falls in the weak-robot audit |
| Handstand on the front legs | `handstand2_best` | 82 % | ~60° target, smooth margin reward; weak robot only 10–30 % |
| Balance on a board on a ball | `balance2_best` | 39.5 % | 100 % at level 0 (damped ball), 0 % at level 9; on cake only |
| Blind terrain course | `terrain_uniform_s0_best` | 75.5 % | stairs, slopes, obstacles; gaps past level 0 never crossed |
| Blind terrain course, distilled | `terrain_student_s0_best` | 77.0 % | student of a teacher that saw the course; stable, no late slump; reads 1 s of sensor history (~2 MB) |

**What worked.**
- **Checkpoint selection** was the biggest single gain: terrain 50.5 → 73.5 % (mean of 3 seeds), handstand2
  10 → 82 %, rear 96 → 100 %. Training curves swing ±15 points; the last iterate is often not the best.
- **Training with sensor failures** (walk2): dead-flow falls 75 % → 0 %.
- **Fixing the sensor model** (balance): the old policy scores 0 % under the corrected sensors; retrained 39.5 %.
- **Reshaping the handstand reward** (target above the success bar, smooth margin term): best 35 → 82 %.
- **Plain uniform sampling** of the scenario bank.
- **Teacher–student** (Experiment 5): no gain in peak success for a blind robot (77.0 % vs 77.0 %), but the student
  does not slump: its last snapshot scores 74.5 % where the blind PPO run's last scores 55.5 %.

**What didn't.**
- **SGS**: 14 runs over three experiments; never beat uniform, and lost by 15–20 points on terrain (details below).
- **Capping the action noise** (`--std_max`, `--ent_end`): terrain 73.5 → 58.5 %, handstand 82 → 66 %, rear 100 →
  98 %. The noise growth came with the late slumps but did not cause them.
- **Constraints as terminations** (`--cat`, 85 % torque limit) on handstand: 0 %. The working handstand exceeds 85 %
  torque on 41 % of its steps, so the rule forbade the skill.
- **Level curriculum** (`--sampler=curr`, up on success, down on failure): +3 points over its uniform control,
  within seed noise; robots settled at level ~1. Balance with the curriculum: 35 % (39.5 % without), still rising.

**Open problems and next steps** (in order of likely payoff):
1. Why training success falls after ~70 M steps on terrain, handstand and balance (learning rate, the value
   network, what the reward pays for late in training).
2. Curriculum promotion by distance covered, not success.
3. ~~Teacher–student training for terrain~~ done (Experiment 5). What it showed: gaps need a sensor that looks
   ahead. A forward-facing multi-zone range sensor (e.g. VL53L5CX, 8×8 zones) and a student that reads a few
   of the teacher's scan points is the next step there.
4. Terrain test closer to the real robot: a delayed, noisy operator (~0.3 s, 10 Hz, deadband, limits), a ±0.6 m
   corridor, a fixed-distance finish.
5. Constraints with limits measured from a working policy, switched on gradually; a longer sensor history and wider
   motor-strength ranges for the weak-robot cases.

## What SGS is

A fixed **bank** of task configurations (scenarios). Each remembers its last H pass/fail outcomes. When an episode
ends, the next configuration is drawn with weight w = (p+ε)^(κt) (1−p+ε)^(κ(1−t)), p = its success rate, softmax
with temperature T. Most practice goes to scenarios solved about a fraction t of the time; mastered and hopeless
ones get less. Paper's locomotion settings: t = 0.66, κ = 5, T = 2, ε = 1e-8, H = 100, bank 104,000.

## Experiment 1: flat walking, base bank (rl/, 2026-10-23)

- Velocity tracking on flat ground; bank of 16,384 scenarios (commands, friction, mass, motor strength, gains, shove).
- SGS vs uniform, one seed each, 300 M steps: **no difference**. Both reached ~99 % of the bank; 90 % at 170 M vs 174 M.
- Why: nearly every scenario is learnable and they are close relatives. Nothing for SGS to steer around.

## Experiment 2: flat walking, hard bank (rl/, 2026-10-23)

- Wider ranges: faster commands (to 1.6 m/s), slicker floors (0.25), heavier (+4 kg), weaker motors (65 %), bigger shoves.
- 400 M steps each: **no difference**. Both plateaued at 83 % of the bank; held-out success 86.3 % (SGS) vs 86.7 % (uniform);
  the two runs solved the same scenarios (0.12 % disagreement). SGS ran ~5 % slower per step.
- SGS did change the practice mix as designed: impossible scenarios 12 % → 2 % of episodes, the learnable middle 9 % → 31 %.
- Why it still didn't pay: the middle band was small (9 % of the bank), and its scenarios are near neighbours of easy ones
  (1.1 m/s on 0.5 grip teaches most of 1.2 m/s on 0.45). The unsolved 17 % is a physics limit, not a training-time limit.
  PPO also learns from every step's reward, not just pass/fail, so "mastered" episodes still teach.

## Implementation lessons

- **The paper's SGS freezes at small scale.** With no prior, a scenario that fails once has p = 0 and its weight falls to the
  ε floor: about 2,000x less likely than a p ≈ 0.66 scenario. At the start every scenario fails, so whichever scenarios
  succeed first take nearly all the samples, and the rest are almost never revisited. In a 16k-scenario bank with
  8,192 robots, 85 % of the bank stayed frozen as "hopeless" while the policy succeeded on 100 % of what it was sampling.
  The paper avoids this by scale: 104k scenarios over ~1M robots (0.1 per robot) revisits everything often.
  Fixes: a bank of ~0.1 scenarios per robot (we use 1,040 for 8,192 robots), or one pseudo-outcome of t as a prior
  (`--sampler=sgsp`), which keeps a single early failure from sinking a scenario.
- **SGS's own success estimate is not a fair metric.** It lags the policy (each estimate is the last H outcomes, some old),
  and it lags more for scenarios SGS rarely picks. Compare samplers on a held-out bank, every scenario equally, mean
  action: `train.py` does this every 50 iterations (`logs/<name>_eval.jsonl`).
- **Sampled-episode success runs above t.** With κ = 5, T = 2 the weights are asymmetric: a p = 0.95 scenario is only ~3x
  less likely than p = 0.66, a p = 0.04 one ~50x. SGS mostly drops the impossible part and keeps revisiting easy ones.
- **Cost:** the categorical draw over the bank each step cost ~5 % of throughput at 16k scenarios.

## Experiment 3: terrain course, paper-style bank (rl2/, 2026-10-24)

- 8 terrain types (flat, rough, slope, stairs up, stairs down, gaps, obstacles, mixed) × 10 difficulty levels, a blind
  policy with the robot's real sensors (rl2/hw.py). 1,040 scenarios, 400 M steps, three samplers × three seeds:
  uniform, SGS as in the paper (H = 100, no prior), SGS with prior (H = 16).
- First look (60 M steps, one SGS run): held-out success 35 %, with a clear learnable-but-hard band: flat ~80 %,
  stairs down and rough falling with level, slopes and stairs up past level 3 near zero, gaps zero. This is the kind of
  bank where the paper reports SGS gains.
- First results (2026-10-24, 400 M steps, held-out success on the full CPU model, 200 episodes):
  uniform 60.5 / 57.0 / 34.0 % (mean 50.5), SGS-with-prior 44.0 / 33.5 / 55.0 % (44.2), paper SGS 25.5 / 33.0 / 39.0 % (32.5).
- **These SGS numbers are void: the SGS bookkeeping had a bug** (found by an independent Codex review). When several
  robots finished the same configuration in the same step, every one advanced the configuration's counter but only
  one outcome was written (a JAX scatter with duplicate indices keeps one). With 8,192 robots over 1,040
  configurations (~8 per configuration) this happened constantly: success estimates mixed fresh and stale outcomes.
  Paper-SGS also starved its bank: median visits per configuration ~22 vs ~980 for uniform. Fixed in `sgs_record`
  (each simultaneous finish gets its own slot; more than H at once are dropped, since the buffer holds H), unit-tested,
  and the six SGS runs are being retrained (`terrain_sgs2_*`, `terrain_sgsp2_*`). Uniform runs are unaffected.
- Even uniform was unstable: every terrain run's held-out success swung +-15 points between evaluations 40 M steps
  apart (4,160 episodes each, so not sampling noise), and uniform seed 2 slid from 63 % at 157 M to 28 % at 393 M.
  Lesson: keep snapshots and pick the deployable one on a separate selection set (`pick_best.py`), never the last
  iterate by default. The learned action noise also grew all run (0.3 -> 1.2-1.75), but it is not the cause:
  capping it at 0.5 with the entropy bonus fading to zero (queue4, `terrain_fix_*`) kept the noise at 0.32 and the
  swings stayed (seed 2: best 59.5 %, last 32 %), while every result got worse (best checkpoints 58.5 % mean vs
  73.5 %). Training success itself falls after ~70 M steps with steady noise. Cause still open.
- Gaps past level 0 were never crossed by any run: a blind policy cannot know where the gap is.
- Results after the fix (2026-10-24, same task, 400 M steps, 3 seeds each):

  | sampler | final, full CPU model | held-out GPU: final / best / average over training | level 0 → level 9 (final) |
  |---|---|---|---|
  | uniform | 60.5 / 57.0 / 34.0 % (mean 50.5) | 48.5 / 70.4 / 48.2 % | 64 → 31 % |
  | SGS, paper settings (H = 100) | 22.5 / 36.0 / 32.5 % (mean 30.3) | 31.1 / 37.1 / 27.2 % | 68 → 17 % |
  | SGS with prior (H = 16) | 33.0 / 42.0 / 27.0 % (mean 34.0) | 33.1 / 56.2 / 36.5 % | 54 → 17 % |

  **Fixing the bug changed almost nothing: uniform still wins, by 15–20 points, on every seed.** Paper SGS before and
  after the fix: 32.5 vs 30.3 % (CPU), 0.28 vs 0.27 average held-out. So the bug was real but was not why SGS lost.
- By terrain (held-out, final, mean of seeds): uniform learned slopes (57 %) and stairs up (51 %); paper SGS 8 % and 6 %,
  SGS with prior 21 % and 22 %. SGS was as good or better on flat (74 vs 63 %) and stairs down (53 vs 52 %).
- Why: **SGS starved the terrains it judged hopeless.** Slopes and stairs up start at ~0 % success for every sampler.
  Uniform kept sending 1 in 8 robots there anyway and eventually broke through; SGS's weight for a p ≈ 0 configuration is
  ~50x smaller than for a p ≈ 0.66 one, so it practised them rarely and never got the first successes. During training
  73–81 % of the bank sat below 10 % success under paper SGS, 49–53 % with the prior, 26–28 % under uniform. The paper's
  premise is that the hopeless set is genuinely out of reach; here it was "not yet learned", and a pass/fail sampler
  cannot tell the two apart. The prior softens this but does not remove it.
- Also: SGS's training episodes succeed less (~10 % vs ~35 % under uniform), so fewer complete demonstrations of a good
  traverse reach PPO. PPO learns from dense per-step reward, not only from the pass/fail signal SGS steers by.

## Process lessons

- **Unit-test the sampler's bookkeeping against a sequential reference.** The SGS bug produced plausible-looking
  numbers and survived two experiments; a six-line test with duplicate configurations exposes it.
- **A second reviewer pays.** The Codex review also caught the balance sensors measuring through the board (range to
  the floor, flow relative to the world). Both bugs were silent: training ran, curves rose.
- **Compare samplers on held-out evaluation, several seeds, and the full CPU model.** One seed would have produced
  the opposite conclusion (uniform seed 2: 34 %; SGS-with-prior seed 2: 55 %).

## When to use SGS here

Not for this robot's tasks as they stand. Three experiments, 14 SGS runs: none beat
uniform sampling, and on the one bank built to suit it, it lost by 15–20 points.

Where it might still pay:
- **A bank with a real impossible fraction** (physically infeasible combinations) that is large, and a learnable
  middle that is not reachable from the easy cases. Here the "hopeless" share shrank under uniform training, so it
  was not impossible.
- **Much bigger banks per robot**, as in the paper (~0.1 configurations per robot, ~1 M robots), where every
  configuration is still revisited often enough for its estimate to move off zero.
- **With a floor**: mix SGS with uniform (e.g. 50/50), so no configuration drops below a minimum practice rate. Not
  tried yet; cheapest next experiment if SGS is revisited.

What did help, more than any sampler: **checkpoint selection.** Picking each uniform run's best snapshot on a separate
selection set raised the final terrain policies from 60.5 / 57.0 / 34.0 % to 75.5 / 74.5 / 70.5 % (CPU, 200 held-out
episodes). The training curve swings ±15 points; the last iterate is often not the best.
With best checkpoints the ranking holds: uniform 73.5 %, SGS with prior 60.8 % (57.0 / 61.5 / 64.0, all picked at
88–98 M steps, so it learns early and then degrades), paper SGS 34.0 % (30.5 / 31.5 / 40.0).

## Experiment 4: a level curriculum instead (rl2/, queue4, 2026-10-24)

- `--sampler=curr` (Rudin et al. 2021): each robot keeps a level per terrain type, up one on success, down one on
  failure, types drawn uniformly, 10 % of episodes uniform over the whole bank. Same noise cap as `terrain_fix_*`.
- Best checkpoints 64.5 / 55.0 / 65.5 % (mean 61.7 %) vs 58.5 % for uniform with the same settings: +3 points,
  within seed noise. Better on gaps (19 vs 9 % held-out), worse on stairs up (20 vs 30 %).
- The mean level stayed at 0.7-1.2: up-on-success / down-on-failure settles where a robot succeeds half the time,
  which on this course is level ~1. Promotion by distance covered (as Rudin et al. do) would push robots higher.
- Neither SGS nor this curriculum beat plain uniform sampling with the original settings (73.5 %).
- Same batch, other skills (best snapshots, CPU): handstand3 (noise cap) 66 % vs 82 %; handstand3c (noise cap +
  constraint termination at 85 % torque) 0 %; balance3 (curriculum + cap) 35 % vs 39.5 %, mean level 2.6 and
  rising when it stopped; rear2 (cap) 98 % vs 100 %. Deployable policies unchanged.

## Experiment 5: teacher–student on terrain (rl2/, queue5, 2026-10-26)

- **Teacher** (`train.py --teacher=1`): PPO whose policy also reads the privileged state through an encoder to 16
  numbers. That is a height scan of the course (10 points from 0.3 m behind to 1.05 m ahead × 3 across), each
  foot's height over the ground, the true velocity, friction, added mass, motor strength, gains and delay. The
  critic reads the same (before, it had 4 heights ahead and the course type and level).
- **Student** (`distill.py`, Lee et al. 2020, RMA): blind. It reads 1 s of its own sensors (50 frames) and
  estimates the teacher's 16 numbers from them, then acts through a copy of the teacher's policy network. It is
  trained by imitation with DAgger: the student drives (the teacher drives some steps over the first 20 %),
  and the loss is the difference from the teacher's action and latent. No reward. 200 M steps, ~20 min.
- **Control**: blind PPO with the same new critic, same seed, 400 M steps (`terrain_blind_s0`).

| | Held-out, GPU (4,160) | Best snapshot, CPU (200) | Last snapshot, CPU (200) | Gaps (GPU) |
|---|---|---|---|---|
| Teacher (sees the course) | **80.9 %** at 59 M, 44.9 % at the end | not deployable | – | **52 %** |
| Student (blind, distilled) | 78.3 % at 103 M, 63–78 % over the whole run | **77.0 %** | **74.5 %** | 12 % |
| Blind PPO control | 70.7 % at 108 M, 55 % at the end | **77.0 %** | 55.5 % | 11 % |
| Earlier blind best (`terrain_uniform_s0`) | 71.5 % | 75.5 % | – | 14 % |

Per terrain type (GPU held-out, best snapshots): teacher / student / blind: flat .95/.97/.96, rough .96/.98/.91,
slope .93/.94/.88, stairs up .94/.86/.74, stairs down .55/.72/.72, gaps .52/.12/.11, obstacles .78/.83/.77,
mixed .83/.62/.56.

What we learned:
- **Seeing the course is worth ~10 points, and nearly all of it is gaps** (52 % vs 11 %; the teacher crosses gaps
  to level 7) **and mixed courses**. A blind robot cannot know a gap is there until a foot falls into it, so the
  student cannot copy that. Its latent error stays high (8.8, from 20), because it cannot predict the scan ahead.
- **The student keeps what can be felt**: stairs up 86 % vs 74 % blind, rough, slope and obstacles a little better.
- **Distillation is stable where PPO is not.** The teacher slumped like every PPO run (81 → 45 %) and the blind
  control too (71 → 55 %). The student, trained on its best snapshot, stayed between 63 and 78 % for the whole run,
  and its final snapshot is nearly as good as its best. On the stress audit (last snapshots, 24 cells × 10
  episodes), student 76 % vs blind 59 %. Supervised learning has no moving target, which is a hint for the late slumps:
  they come from the RL loop, not from the network or the task.
- **Peak blind success did not move**: 77.0 % for both on the CPU test (200 episodes, ±3 points). The blind control
  with the new critic also beat the earlier blind best (75.5 %): a scan in the critic may help PPO itself.
- Practical: `terrain_student_s0_best` is the new deployable terrain policy (same score, robust late, ~2 MB, numpy;
  policy.py runs it on 50 frames of history). Teacher snapshots are on cake only.
- Next: a forward-looking range sensor so the student gets part of the scan (gaps); 3 seeds of teacher and student;
  distilling from a teacher trained longer at a low learning rate.

