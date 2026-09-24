# Stroke-grouped retrieval head: new best model, and "mask the pose" beats "feed back the predicted pose"

Date: 2026-09-24. Follow-up to the dense-anchor model (`dense_anchor` seed 27: dense
1024-point dataset, frozen InfoNCE-pretrained PointNet, one anchor network, dough-relative
tool pose masked 50% in training). All numbers below are offline test-set errors on the
same 125 test segments (9 held-out episodes) unless stated otherwise.

## TL;DR

- **New best checkpoint:** `/workspace/runs/traj_head/cond_traj_seed27_out/vinn_rbf/checkpoint.pt`
  (`dom_retrieval` commit `663f7d7`, config `/workspace/runs/traj_head/cond_traj_seed27.yaml`).
  Tool pose available: start 15.7 / shape 20.9 / raw 24.9 mm / orientation 5.8°,
  against 17.5 / 22.9 / 28.2 mm / 6.1° for the previous best.
- **What changed:** a post-hoc retrieval-head stage (`retrieval_head_stage`) teaches retrieval to
  put together training segments whose *strokes* were similar, not only segments whose dough
  looked similar. The targets mix stroke similarity with 0.5 × centered-Chamfer dough similarity,
  so the dough keeps its role.
- **Deployment finding (applies to all models):** in a compounding chained test over every test
  episode (up to 19 hops), feeding the model its own predicted endpoint as the next tool pose is
  *worse than masking the pose*. For every checkpoint tested, mask the pose when no measured pose
  exists. Under masking, the new model is also the best: 28.4 mm raw against 33.1 mm.
- **Simulation (episode 18, pc137–154):** roughly on par with the previous best. The per-particle
  offset is slightly better with fixed orientation and slightly worse with recorded orientation.
- **The motion is still smaller than the real one** (about 82% of the true size). Stroke grouping
  improves *which* strokes get retrieved, not the hedging between them.

## 1. Why the motion was too small (context)

Predicted strokes reach about 81% of the true motion size. The prediction is a retrieval-weighted
average of training trajectories, and averaging genuinely different strokes shrinks it. Fixes
tried on seed 27:

| Approach | Shape error | Motion size |
|---|---|---|
| Weighted average (previous best) | 22.9 mm | 81% |
| Sharper retrieval (bandwidth ×0.5 → ×0.05) | 23.4 → 30.5 mm | 84 → 101% |
| 3 modes + post-hoc learned selector | 23.6 mm | 86% |
| Time-aligned averaging (DTW barycenter) | 24.1 mm | 82% |
| Single most-aligned trajectory among the top 16 | 24.6 mm | 90% |
| *Best single retrieved trajectory, in hindsight* | *13.8 mm* | *97%* |

Sharper retrieval degrades toward the 27.5 mm dataset-mean-trajectory floor. The last row matters
most: a nearly exact stroke is almost always retrieved, but picking it needs information the
model doesn't use. In the simulator, the top-16 pick did worse than the average on the one
calibrated segment (per-particle offset 12.9 against 10.0 mm).

## 2. Stroke-grouped retrieval: offline diagnostic

`/workspace/runs/traj_retrieval/traj_contrastive.py`. Starting from the seed-27 checkpoint, only
the pose branch and projection of the retrieval embedding are fine-tuned. The loss is
soft-target InfoNCE over the whole memory, with same-episode items excluded. Targets are
`softmax(-d / τ)`, where d is the start-relative path distance, row-median scaled, optionally
plus 0.5 × centered Chamfer. The epoch is selected on validation retrieval shape error, with the
untouched head competing as epoch 0. 3 seeds each:

| Variant | Test shape error | Seeds | Best stroke in top 16 | Dough reliance* |
|---|---|---|---|---|
| Previous best (no stage) | 22.9 mm | — | 18.3 mm | 2.3 mm |
| Stroke targets only, τ=0.1 | 20.7 mm | 20.6 / 20.7 / 20.8 | 16.9 mm | 2.1 mm |
| Stroke targets only, τ=0.05 | 21.0 mm | 20.6 / 20.7 / 21.7 | 16.6 mm | **1.6 mm ↓** |
| **Stroke + 0.5 Chamfer, τ=0.1** | **21.1 mm** | 21.1 / 20.9 / 21.4 | 16.6 mm | **2.9 mm** |
| Stroke + 0.5 Chamfer, τ=0.05 | 21.0 mm | 21.1 / 21.0 / 20.9 | 16.7 mm | 2.4 mm |
| Unfrozen PointNet | 21.5–21.8 mm | — | 16.5–17.4 mm | overfits by epoch 300 (23–24 mm) |

\*Dough reliance is the increase in shape error when each query's dough cloud is swapped for
another query's while its tool pose is kept. It checks that the dough still matters. Stroke-only
targets let the model lean on the pose. The Chamfer mix keeps or increases the dough's role at
the same accuracy, so it was chosen. Pose-masked shape error is unchanged (about 26.2 mm) in
every variant.

## 3. Integrated model (`dom_retrieval` 663f7d7)

`training/trainer.py::fit_retrieval_head` runs after `train_policy` when
`retrieval_head_stage.enabled` is set (relative_temperature 0.1, chamfer_weight 0.5, 300 epochs,
lr 3e-3). The whole encoder is then frozen and the policy is refit, so the anchor network adapts
to the new embedding. There is a unit test
(`test_retrieval_head_stage_changes_only_pose_and_projection`), and the suite passes
(2 skipped).

Test set, 125 segments (`eval_modes.py`):

| | Seed 7 | Seed 17 | Seed 27 | Mean | Previous mean |
|---|---|---|---|---|---|
| Pose: start / shape / raw (mm) | 18.8 / 23.8 / 28.8 | 16.1 / 22.4 / 25.2 | **15.7 / 20.9 / 24.9** | 16.9 / 22.4 / 26.3 | 19.9 / 24.3 / 29.3 |
| Pose: orientation | 6.6° | 6.3° | 5.8° | 6.2° | 6.7° |
| Masked: start / shape / raw (mm) | 25.1 / 26.4 / 31.0 | 24.8 / 26.9 / 30.0 | 25.5 / 26.2 / 30.5 | 25.1 / 26.5 / 30.5 | 27.8 / 26.4 / 32.8 |

- **Seed 7** is identical to its previous checkpoint: the stage kept epoch 0, because validation
  didn't improve.
- **Seed 17** improves most (raw 31.0 → 25.2 mm). Its previous retrieval was nearly uniform
  (max weight 0.01), and the stage fixed that.

## 4. Compounding chained test: mask the pose, don't feed back predictions

`/workspace/runs/traj_head/chained_multi.py`. For every run of contiguous test segments (all 9
test episodes, up to 19 hops), segment k's tool-pose input is segment k−1's *chained* predicted
last waypoint, so errors compound. Same segments, three input modes (raw error):

| Checkpoint | Measured pose | Chained (own prediction) | Pose masked |
|---|---|---|---|
| Previous seed 7 (= new seed 7) | 27.9 mm | 39.4 mm (+42%) | 29.7 mm |
| Previous seed 17 | 30.1 mm | 34.0 mm (+13%) | 31.1 mm |
| Previous seed 27 | 27.1 mm | 37.8 mm (+39%) | 33.1 mm |
| New seed 17 | 24.9 mm | 38.9 mm (+56%) | 29.2 mm |
| **New seed 27** | **24.6 mm** | 45.2 mm (+84%) | **28.4 mm** |

- The single-pair chained test we'd been using (episode 18, pc137→pc154→pc172: 19.3 → 23.9 mm)
  understated the problem. Over full episodes, feeding back predicted poses compounds to +13–84%.
- For every checkpoint, **masking the pose beats feeding back the predicted pose.** Deployment
  rule: give the model a measured tool pose when one exists, otherwise mask it. Never feed back
  its own predictions.
- Stroke grouping makes retrieval lean more on the pose, so bad pose inputs hurt it more (+84%).
  Under both deployment-valid modes (measured or masked), it is the best model.

## 5. MPM simulation (TaichiDough, episode 18 chunk06 = pc137–154, open loop, 0.534 s)

This segment is in the validation split for both models. Physics:
E=8803, ν=0.487, viscosity 30.4, plastic 0.863–1.067, floor retention 0.7, friction 0.9.
Offline on this one segment, the new model is *worse* (shape 17.0 against 11.5 mm; raw 21.1
against 18.7 mm) but moves more (86% against 81% of the true size).

| Condition | Path length (t1 / t2) | Tool error (t1 / t2) | Final Chamfer | Per-particle offset |
|---|---|---|---|---|
| Recorded demonstration | 191 / 185 mm | — | 0 | 0 |
| Previous best, fixed orientation | 154 / 138 mm | 16 / 21 mm | 3.37 mm | 10.0 mm |
| Previous best, recorded orientation | 148 / 144 mm | 17 / 21 mm | **2.78 mm** | **8.4 mm** |
| New, fixed orientation | 163 / 150 mm | 20 / 22 mm | 3.24 mm | 8.6 mm |
| New, recorded orientation | 157 / 155 mm | 21 / 21 mm | 3.17 mm | 8.8 mm |
| Tools held still | 0 / 0 | 46 / 39 mm | 3.21 mm | 13.2 mm |

- The new model's paths are longer (closer to the demonstration's).
- It ends closer to the recorded dough with fixed orientation (8.6 against 10.0 mm), and slightly
  further with recorded orientation (8.8 against 8.4 mm).
- Neither reproduces the deep horseshoe cavity.

Figure: `stroke_grouped_retrieval_sim_analysis.png`. Runs are in
`/workspace/TaichiDough/runs/dom_retrieval_sim_traj/` and `.../dom_retrieval_sim_top16/`.

## Caveats / open

- The simulator covers one calibrated segment, and it's in the validation split. Other episode-18
  chunks (01–13) are available for a broader test.
- Validation has only 49 segments. Seed 7's stage selected "no change" on validation even though
  the offline diagnostic improved every seed.
- Hedging (about 82% motion size) remains. The top-16 hindsight ceiling (16.6 mm) points to
  selection information, e.g. the previous stroke, as the next lever.

## Running the model elsewhere

Copy the whole run folder (`cond_traj_seed27_out/`: `checkpoint.pt` plus `dataset.pt`, which is
hash-checked by `load_experiment`) and the `dom_retrieval` repository at `663f7d7` or later.
Inputs must match training: 1024 real points in the DeformPath2 camera frame
(`Deformapth2_camera1024_interpolated_nodbscan` preprocessing), and the tool pose
`[x,y,z,qx,qy,qz,qw]` × 2 in the same frame, or `model.encoder.pose_masked = True` when there is
no measured pose.
