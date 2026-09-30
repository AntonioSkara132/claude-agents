# Follow-ups on the stroke-grouped retrieval model: progress, timing, history, dough size, residual, scaling

Dates: 2026-09-25 to 2026-09-30. Follow-up to `RESULT_STROKE_GROUPED_RETRIEVAL_HEAD.md`.

Code: `dom_retrieval` commits `c0e1c2a`, `43e634c`, `c64ff6c`, `b34072a` (pushed). Every new option is
**off by default**. Usage and a summary table: `dom_retrieval/docs/RUNNING.md` section 7.

Baseline: the current best config (`configs/deformpath_best_stroke_grouped.yaml`, runs `/workspace/runs/traj_head/`).
All numbers are test-set means over seeds 7/17/27 (125–147 test segments per seed), from
`scripts/evaluate_deployment.py`.

**Bug found afterwards (fixed in `dom_retrieval` `3f0b988`):** `augment_workspace_translation` added
the scene offset (up to 0.3 m) in metres to the already-normalized target (position std 0.026–0.042 m).
So the target's start moved only ~1 cm while the cloud and pose moved 0.3 m. The trajectory loss's start
term and `predicted_start` therefore pulled the start head toward different places. Every run here, and
the current best, used `workspace_translation_m: 0.3`. Shape errors are unaffected (start-relative);
start and raw errors need re-measuring after retraining.

Retrained current best (same config, runs `/workspace/runs/augfix/`, 3 seeds):

| | Before fix | After fix | Seeds after fix |
|---|---|---|---|
| Measured pose: start / shape / raw | 16.9 / 22.4 / 26.3 mm | **15.7 / 21.6 / 24.7 mm** | raw 26.2 / 24.0 / 23.8 |
| Measured pose: orientation, motion size | 6.2°, 81% | 6.7°, 82% | |
| Pose masked: start / shape / raw | **25.1 / 26.5 / 30.5 mm** | 26.9 / 26.4 / 31.7 mm | |
| Chained, raw: measured / masked | 25.8 / **29.1** mm | **23.8** / 30.3 mm | |

- Shape is unchanged for seeds 17 and 27 (22.3 / 20.9 mm), as expected. The mean moves only because
  seed 7's stroke-grouping stage now picked a trained head (it kept epoch 0 before).
- With a measured pose, the start improves 1.2–2.2 mm on seeds 17/27. With the pose masked, the start is
  1.8 mm worse: correctly shifted 0.3 m scenes are harder to place from the dough alone than the buggy,
  nearly unshifted targets.
- New best measured-pose checkpoint: `augfix/cond_augfix_seed27_out/vinn_rbf/checkpoint.pt` (raw 23.8,
  start 14.5, shape 20.9 mm, 6.3°).
- The TL;DR numbers below predate the fix; they compare variants that all had the bug.

## TL;DR

| Variant | Measured pose: start / shape / raw | Motion size | Pose masked: start / shape / raw | Verdict |
|---|---|---|---|---|
| Baseline (stroke-grouped retrieval) | 16.9 / 22.4 / 26.3 mm | 81% | 25.1 / 26.5 / 30.5 mm | — |
| **+ episode progress** (`progress_input`) | 15.9 / **20.8** / **24.9** mm | 83% | 26.7 / **24.5** / 31.1 mm | **helps** |
| + outlier-robust cloud scaling (`encoder.robust_normalization`) | 17.7 / 21.4 / 26.0 mm | 83% | 25.7 / 26.0 / 30.8 mm | small gain |
| + previous stroke, recorded (`previous_stroke_input`) | 17.2 / 20.9 / 25.4 mm* | 86% | 24.2 / 21.2 / 29.5 mm* | only with true history |
| + per-waypoint residual (`cartesian_residual`) | 26.9 / 22.2 / 31.9 mm | 81% | 31.0 / 26.6 / 35.7 mm | worse |
| Fixed real-time steps (`data.step_seconds`) | see section 2 | | | no gain |
| Dough size as input (diagnostic) | see section 4 | | | no gain |

\*On top of the progress model; its baseline is the progress row.

## 1. Episode progress

Inputs: seconds since the episode started and the stroke (segment) number. Both are known live. A
zero-initialized branch is added to the retrieval embedding (proprioceptive encoder); memory and
checkpoints carry the field. Kept segments are contiguous within every episode, so stroke numbers have
no gaps; 6 episodes' annotations start a stroke or so after frame 0.

- Quick check before training (reweighting retrieval by progress similarity): −0.36 mm test shape.
- Trained, in the diagnostic stage: time + stroke number 21.91 → 21.34 mm; time alone only 21.76 mm.
- Full training, 3 seeds: see the TL;DR table. Diversity (predicted / true shape spread) rises from
  9.2 / 18.9 to 10.7 / 18.9 mm.
- Best single checkpoints: `/workspace/runs/progress_head/cond_prog_seed27_out/vinn_rbf/checkpoint.pt`
  (shape 19.5 mm, motion 84%) and `cond_prog_seed17_out` (raw 23.9 mm).
- Regression: with the pose masked, start error is 1.6 mm worse. The anchor network reads the same
  embedding, which now carries progress. Chained, pose masked: 30.3 against 29.1 mm raw.
- Not yet simulated.

## 2. Timing: fixed real-time steps and predicted duration

Segments are stretched onto 32 waypoints (0.17–8.4 s in training, median 0.53 s), so the model never
predicts timing.

- **Predicted duration:** retrieval-weighted neighbour durations give ~10% median error. The simulator
  now uses it by default. On chunk06 the result matches true-duration timing (offset 8.1 vs 8.6 mm).
- **An oracle given the true duration** cut shape error by ~2 mm (20.9 → 18.8 mm) and raised motion to
  86%. Adding duration similarity to the stroke-grouping targets recovered none of it
  (22.34 / 22.39 / 22.50 / 22.95 mm at weight 0 / 0.5 / 1 / 2).
- **Fixed real-time steps:** 32 steps at 1/30 s, continuing past the segment end into the recording.
  Scored in real time over each segment's true duration against the segment model spread over its
  predicted duration: measured pose shape 22.4 vs 21.9 mm, raw 27.6 vs 26.0 mm (start 19.3 vs
  15.9 mm); masked shape 25.5 vs 26.3 mm, raw 33.6 vs 32.9 mm; motion 81 vs 79%. No gain.

## 3. History: the previous stroke

Input: the preceding contiguous stroke of the same episode, as 8 start-relative waypoints × 2 tools,
plus a has-previous flag. It's a zero-initialized branch, added on top of the progress model.
`evaluate_deployment`'s chained test gained `measured_own` / `masked_own` modes, where the previous
stroke is the model's own previous prediction, as in a live run.

- **With the recorded (human) previous stroke:** masked shape 24.5 → 21.2 mm, motion 86%, diversity
  12.6 mm. What just happened does predict what comes next.
- **With the model's own previous strokes:** chained raw 27.4 vs 24.7 mm (measured pose) and 31.5 vs
  30.3 mm (masked), worse than no history. Its own strokes are smaller and averaged, unlike its training
  inputs.
- **Caveat:** the offline chained test pairs the model's stroke with dough shaped by the human's stroke.
  A closed-loop simulation is needed to judge it, or training on its own strokes (scheduled sampling).

## 4. Dough size

- **The max-radius dough size is ruined by stray points.** The nodbscan clouds have points up to
  ~290 mm from an ~80 × 87 × 15 mm dough.
- **The robust size shrinks as expected:** the 80th-percentile radius around the median point falls in
  45 of 46 episodes (median −17 mm first → last) and correlates −0.71 with elapsed time.
- **It adds nothing for choosing strokes.** Reweighting by size: −0.1 mm. As a trained input: worse
  alone (22.58 vs 21.91 mm), and marginal on top of progress (21.16 vs 21.34 mm test, validation
  22.94 vs 22.85). Progress already carries it.

## 5. Per-waypoint residual

`cartesian_residual` with a per-waypoint, position-only residual on the retrieved path. Same config
otherwise.

- Shape barely moves (22.2 vs 22.4 mm) and motion size is unchanged (81%).
- The start is 10 mm worse on every seed (26.7 / 27.4 / 26.8 mm), because the residual also shifts
  the first waypoint and undoes the anchor network. Orientation is 7.1° vs 6.2°.
- It repeats the old sparse-data result (RBF + residual 21.44 vs 21.29 mm without).

## 6. Outlier-robust point-cloud scaling

`normalize_cloud` centres on the mean and scales by the farthest point, and PointNet then max-pools.
One stray point 290 mm away shrinks the dough's normalized radius 3.4× (0.275 → 0.081). The option
centres on the median, scales by the 90th-percentile radius and clamps points beyond 2 radii. It is
unaffected by the stray point (0.553 → 0.552).

- Measured pose: shape 22.4 → 21.4 mm (seeds 21.3 / 21.8 / 21.1 vs 23.8 / 22.4 / 20.9), motion 81 → 83%,
  diversity 9.2 → 9.6 mm, but start 16.9 → 17.7 mm. Raw 26.3 → 26.0 mm, orientation 6.2 → 6.3°.
- Pose masked: shape 26.5 → 26.0 mm, raw 30.5 → 30.8 mm; chained masked 29.1 → 29.4 mm.
- Most of the gain is on seed 7, whose baseline stroke-grouping stage had kept the untouched head.
- A small, cheap improvement to the dough-shape input. Not yet combined with progress.

## 7. ACT regression policy, for comparison

The DeformPath session's ACT work log: `DeformPath/docs/act_regression_2026-09-24.md`
(branch `colab-imports`, not pushed). On the 151 validation segments that all ACT runs cover
(18-frame windows, dataset-mean shape floor 41.5 mm):

| ACT run | Start | Shape | Raw | Motion size |
|---|---|---|---|---|
| Run 1, point cloud only | 32.2 mm | 31.0 mm | 34.6 mm | **70%** |
| Run 3, anchored on previous pose | 9.9 mm | 31.5 mm | 33.9 mm | 47% |
| Run 5, anchored + 4-pose history (their best) | 10.9 mm | 30.7 mm | 33.3 mm | 53% |
| Run 1's stroke started from run 5's start | 10.9 mm | 31.0 mm | 33.2 mm | 70% |

- Stroke shape is the same across ACT runs. Anchoring only fixed the start, and it shrank the strokes.
  Run 1's strokes with a known start match the best run on error while staying bold.
- Relative to its own floor, ACT's shape error is 75–78% of the floor. Ours is 76%
  (20.9 / 27.5 mm). Different data export, split and horizon, so this is only a rough comparison.
- Figures for run 5 were added to the shared results page, section "Fourth run…".

## Next

1. Simulate all 13 calibrated episode-18 chunks, not just chunk06, for the current best, the progress
   model and ACT run 1.
2. Build a closed-loop simulation (simulated dough → point cloud → next stroke). It is needed to judge
   history, bold strokes and feeding back the model's own poses or strokes.
3. Combine progress with robust scaling (3 seeds, ~15 minutes).
