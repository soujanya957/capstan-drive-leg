# Codex review of rl2, 2026-10-24

An independent review by OpenAI Codex (codex-cli, read-only) of the code and the first full set of results, and the
follow-up discussion with Claude. Links point into this directory.

## Round 1: review

# Independent review

**I would not treat any policy as cleared for the real robot yet.** The walking results are encouraging in the modeled conditions, but the terrain controller uses simulator state to create commands, and the balance sensor model does not represent what a downward sensor would see above the board. The code and logs support useful research conclusions; they do not yet support a hardware readiness claim.

## 1. Verification

| Finding | Evidence and consequence |
|---|---|
| **Terrain steering uses exact simulator position and yaw.** | [tasks.py:319](tasks.py:319)–[321](tasks.py:321) computes sideways and turn commands from `s["p"][1]` and yaw. Those commands enter the actor observation through [core.py:229](core.py:229). The listed hardware has no absolute lateral position or course heading source ([README.md:24](README.md:24)–[36](README.md:36)). The policy therefore receives steering information the proposed robot cannot generate as written. |
| **Terrain “success” does not require staying on the course.** | [tasks.py:315](tasks.py:315)–[318](tasks.py:318) checks forward displacement, posture, and height, but no lateral corridor or continuous terrain contact. Walking beside obstacles could count. The logs do not record paths, so this is a vulnerability in the test, **not proof** that the trained policies exploited it. The goal also varies from 2.04 to 6.12 m with sampled speed ([tasks.py:279](tasks.py:279)–[281](tasks.py:281)); success does not mean traversing the entire roughly 8 m course. |
| **Balance range and flow are modeled against the floor, through the board.** | `Balance` inherits the flat-ground `ground()` value of zero ([tasks.py:79](tasks.py:79)–[80](tasks.py:80)), while [core.py:218](core.py:218)–[224](core.py:224) synthesizes downward range and flow from that value and torso velocity. On the proposed board, the downward sensors would generally see the *moving board surface*: range is shorter and optical flow is relative to the board. This is a material sim-to-real error for the task with the weakest sensor robustness. Flow also omits sensor-offset rotational motion and line-of-sight geometry during large body tilts. |
| **The SGS formula is close to the paper, but the comparison is not a faithful replication.** | The kernel, 100-outcome window and parameters in [train.py:91](train.py:91)–[104](train.py:104) match the paper’s stated locomotion settings. The paper uses 104,000 locomotion configurations and emphasizes diverse starts and goals; these runs use 1,040 configurations, a different terrain task, and a heavily shaped per-step reward ([paper, §§3.2–3.3 and Appendix A.1](https://arxiv.org/html/2610.12465v1)). `sgsp` adds a prior and uses a 16-outcome window, so it is explicitly a variant. Also, [train.py:107](train.py:107)–[112](train.py:112) writes simultaneous outcomes for the same configuration into the same circular-buffer cell: `count` adds all visits, while colliding `.set` writes retain only one outcome. That makes the estimated success rate inaccurate when episodes finish together. |
| **Some evaluation labels overstate what was tested.** | The 200-episode CPU terrain evaluation has only about 2–3 trials per type × level cell ([play.py:40](play.py:40)–[68](play.py:68)); individual cells are too noisy for strong claims. The walk audit uses a scripted command sequence, resets task time to zero each step, and computes its own error ([audit.py:68](audit.py:68)–[105](audit.py:105)). Its fall and RMS figures are useful, but they are not the `Walk` eight-second success test. The named sensor stresses also use assumed bounds from [hw.py:14](hw.py:14)–[55](hw.py:55), which have not been measured on this build. |

I did **not** find an obvious sign error in the pose success threshold or a PPO time-out bootstrap error. The handstand problem looks primarily like reward and robustness behavior, rather than a broken success predicate.

## 2. Reading the four claims

1. **Terrain — mostly agree, with two qualifications.** Final CPU means are **50.5% uniform, 44.2% `sgsp`, 32.5% SGS** across three seeds, as reported in the `terrain_*_cpu_eval.json` logs. The ordering is descriptive, not a strong general ranking with only three seeds; uniform seed 2 finishes at 34%, versus `sgsp` seed 2 at 55%. The late swings are too large to explain as 4,160-episode binomial noise alone: uniform seed 2’s GPU held-out success drops from about **0.63 at 157M** to **0.28 at 393M** (`terrain_uniform_s2_eval.jsonl`). But “gaps physically impossible” is unsupported. High-level gaps were unsolved **by these policies and this test**. The brief’s speed hypothesis is false: [tasks.py:279](tasks.py:279) samples 0.3–0.9 m/s independently of level. One notable sampler diagnostic: final bank logs show median visits per configuration of about **980 uniform, 21–24 SGS, and 667–781 `sgsp`**. The SGS run heavily starved much of its small bank.

2. **Handstand — partly agree.** The pose kernel in [tasks.py:364](tasks.py:364) is broad near the 0.766 success threshold, while the terminal success term contributes little to logged reward. That plausibly encourages a near-threshold pose; the reward-term average alone cannot establish the actual average \(f_z\), so the brief’s inferred \(f_z \approx 0.76\) is only a hypothesis. CPU success is **14% handstand versus 96% rear**. The weak-robot audit contradicts “both fail with no falls”: rear has **0 falls in 30 trials** across levels 0, 5, 9, while handstand falls in **14 of 30** (`rear_audit.json`, `handstand_audit.json`). The handstand also falls frequently with dead flow.

3. **Balance — agree, within the modeled stresses.** CPU held-out success is **44% over 200 episodes**. The audit has **0/30 successes** for worst sensors and **0/30** for dead flow; slippery gives **2/30** (`balance_cpu_eval.json`, `balance_audit.json`). Ten trials per level are imprecise, and the incorrect board sensor model means these figures cannot be projected directly to hardware.

4. **Walk2 — agree on the tested failure, not on broad readiness.** Dead-flow audit falls improve from **75% for walk** to **0% for walk2**, with walk2 RMS linear error **0.107 m/s**, across the audit’s scripted trials (`walk_audit.json`, `walk2_audit.json`). Other audited conditions show no falls and similar errors. This is persuasive evidence that training with dead-flow episodes fixed that particular simulated failure. It remains a 10-episode audit per condition using assumed sensor and motor models.

## 3. Ranked improvements

GPU costs below use the logged throughput as a rough guide: about **14–15 minutes per 300M walk/pose steps**, **28 minutes per 400M terrain steps**, and **38 minutes per 500M balance steps**, plus compilation and CPU evaluation.

| Rank | Change | Why and expected effect | GPU cost; measure |
|---|---|---|---|
| **1** | In [core.py:209](core.py:209)–[232](core.py:232), generate range and flow from the sensor’s body-frame pose and the first surface its ray sees. For `Balance`, add a board-surface intersection and subtract board motion from flow; flag invalid readings when geometry or range is out of bounds. Retrain balance. | Removes a large, task-specific observation mismatch. The likely effect is a more honest and more transferable balance policy; reported simulator success may initially decrease. | Roughly **40–60 minutes**. Check simulated sensor traces against bench measurements on a moving board, then compare CPU success and dead-flow/worst-sensor audits. |
| **2** | Replace the terrain command construction in [tasks.py:319](tasks.py:319)–[321](tasks.py:321) with commands available from the actual operator interface or onboard estimator. For a straight-course baseline, ramp forward speed and set lateral/yaw commands to zero. Retrain with uniform sampling. | Removes the simulator’s position/yaw steering oracle. Expect lower simulated success at first, but the result will test a controller that can actually be run. | Roughly **30–45 minutes**. Compare paired CPU episodes with oracle steering disabled; log lateral displacement, falls, and completion. |
| **3** | Tighten terrain completion in [tasks.py:315](tasks.py:315)–[324](tasks.py:324): require a defined corridor, record the surface traversed, and report full-course completion separately from the speed-dependent goal. Add path and exit reasons to [play.py:40](play.py:40)–[68](play.py:68). | Prevents an off-course route or short goal from being presented as terrain traversal. Expect a more credible, possibly lower headline rate. | **No GPU time for re-evaluation**; about **30–45 minutes** if retraining against the new objective. Measure on paired seeds, with at least tens of trials per terrain × level of interest. |
| **4** | Select policies by held-out **CPU and stress performance**, rather than saving the last iterate as the deployable choice. Use the existing snapshots from [train.py:306](train.py:306)–[315](train.py:315); evaluate several candidates with [play.py:40](play.py:40) and [audit.py:186](audit.py:186). | Late policy drift is substantial for terrain and balance. A selected checkpoint may improve performance immediately, though it does not repair model errors. | **Zero GPU time**. Use a separate selection set and final test set; compare completion, fall rate, and stress results, not just one GPU success number. |
| **5** | For handstand, sharpen or reshape `pose` near the required angle and increase feedback for sustained hold in [tasks.py:335](tasks.py:335)–[373](tasks.py:373). Include dead-flow and weaker-motor episodes; inspect angle trajectories before choosing weights. | Targets the observed near-threshold failures and severe weak-robot/dead-flow fragility. Expect more robust holds, with possible increased motor demand that must be checked. | Roughly **15–30 minutes per 300M run**. Measure the fraction of the final two seconds above the angle threshold, falls, rise time, peak torque, and CPU success under weak/dead-flow conditions. |

Separately, if SGS experiments continue, fix the duplicate-index history update in [train.py:107](train.py:107)–[112](train.py:112) and test its buffer against a sequential reference. Given uniform’s current advantage, I would not spend today’s retraining budget on SGS tuning.

## 4. One change to make and retrain today

**Remove terrain’s simulator-derived lateral and yaw commands, then retrain one uniform terrain policy.** Edit [tasks.py:319](tasks.py:319)–[321](tasks.py:321) so the straight-course trial supplies forward speed with zero lateral and yaw command. A 400M run appears to fit comfortably within 1–2 GPU hours. Evaluate it on paired CPU courses, recording lateral exits as well as success. This gives the quickest answer to the most consequential question for terrain deployment: whether the policy can traverse the course without steering information supplied by the simulator.

## Round 2: Claude's reply (summary)

Accepted and fixed: the SGS duplicate-index write, the balance sensors measuring through the board, checkpoint
selection. Disputed: zeroing the terrain steering commands; the command is the human operator's joystick, a
legitimate input, but the simulated operator is too perfect. Proposed: a delayed, noisy operator, a corridor and a
fixed-distance success test (next session), and a handstand reward aimed above the success bar.

## Round 2: Codex

**Operator argument:** I accept it. A human’s joystick is a legitimate actor input even when the human uses visual information unavailable onboard. The current simulated operator is still an idealized controller. Your delayed, noisy model is a useful next test; specify its update rate and command limits as well as delay, and evaluate continuous corridor occupancy. A fixed finish distance will make completion easier to compare across commanded speeds.

**Budget and order:** The listed GPU runs total **221 minutes (3 h 41 min)** before overhead, so they do not fit a 2.5-hour budget. I would validate the SGS fix with one short run, then prioritize three SGS seeds (84 min), balance (38 min), and a handstand run (15 min): **137 minutes** nominal, leaving little margin. Defer the three `sgsp` seeds. Do checkpoint selection on CPU, using separate selection and final evaluation sets. The old SGS comparison is invalidated; the old uniform runs remain a valid baseline for the unchanged terrain task.

**Fix review:** The SGS ranking gives distinct slots for same-configuration completions **until more than `H` finish in one step**. Beyond `H`, modulo wraps and duplicate scatter writes return, so a highly concentrated sampler can still lose outcomes. Your six-completion test covers the normal case; test an `H+1` batch too. The balance change fixes the major floor-versus-board error, but `surface()` selects the board from the torso center rather than the sensor ray, and `ground_v` omits board rotation at the viewed point. Those are remaining sensor-model approximations, especially near an edge or during tilt.

**Handstand:** Narrowing the pose kernel is sensible. I would be cautious about moving the target to `0.94` (~70°) when the observed valid pose is near 50°: it may demand a pose limited by body clearance or motor strength. The proposed 1-per-step hold reward has a sharp jump at `F_OK`, which can encourage hovering at the threshold. Inspect angle, body-clearance, torque, and weak-robot traces before committing to the higher target; reward sustained margin above the bar rather than only crossing it.

## Outcome

- `sgs_record`: simultaneous finishes ranked into their own slots, more than H at once dropped; unit tests for both.
- `Balance.surface`: range and flow see the board under the torso. Remaining approximations: the torso centre stands in
  for the sensor ray, and the board's rotation at that point is ignored.
- Handstand: target 0.87 (~60 deg), kernel 0.05, smooth `margin` term above the bar (not the 70 deg first proposed).
- `pick_best.py`: snapshot selection on a separate CPU selection set.
- Retraining: balance2, terrain SGS x3 and SGS-with-prior x3 with the fix, handstand2 (`queue3.list`).
- Next session: operator model, corridor, fixed-distance terrain success.

## Results of the retraining (2026-10-24)

- SGS fix: paper SGS 30.3 % mean (was 32.5 % with the bug), SGS with prior 34.0 %, uniform 50.5 % (final policies,
  CPU, 200 episodes). Best checkpoints: uniform 73.5 %, SGS with prior 60.8 %, SGS 34.0 %. The bug was real but was
  not why SGS lost; see `sgs-lessons.md`.
- Balance sensors: the old policy scores 0 % under the corrected model; balance2 39.5 % (100 % at level 0, 0 % at level 9).
- Checkpoint selection: terrain uniform 50.5 → 73.5 %, rear 96 → 100 %, handstand 14 → 35 %, handstand2 10 → 82 %.
- Handstand2 (~60° target, smooth margin): 82 % at 78 M steps, near-vertical hold; 10 % at 299 M. Not a reward
  trade-off (the pose term held steady in training): the action noise grew 0.23 -> 0.54 and the mean action decayed.
  Same in terrain (0.3 -> 1.2-1.75) and rear (-> 6.2). Tried --std_max / --ent_end (queue4): handstand3 peaked at
  63 % and still slid; terrain got worse. The noise growth was a symptom; the cause is still open.
- Constraints as terminations (handstand3c, --cat=0.25): 0 %. The 85 % torque limit was below what a handstand
  needs (handstand2 exceeds it on 41 % of steps), so the policy learned not to try.
