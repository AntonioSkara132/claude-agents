# Result: joint training + contrastive loss, clean PointNet, real DeformPath data

Follow-up to `RESULT_DOM_RETRIEVAL_REAL_DATA_TRAINING.md`. The clean PointNet run there used
frozen encoder / `loss.contrastive: 0.0` (`configs/deformpath.yaml` defaults). Its `direct`
policy plateaued almost immediately (`history.json`: train loss flat at 0.79 normalized from
epoch 2), and `nearest`'s retrieval was the worst performer of the six methods, consistent with
never training the embedding space for retrieval relevance. Tried two follow-up configs to test
that hypothesis, both still PointNet only (no PointMAE leakage concerns), same 611-motion /
451-41-119 split:

- `freeze_encoder: false`, `loss.contrastive: 0.1`, `epochs: 20`
- Same, `epochs: 200`

No `dom_retrieval` source files changed; only YAML config (`freeze_encoder`/`loss.contrastive`
are existing, tested knobs — `training/trainer.py`'s joint-training + contrastive path already
has test coverage: `test_joint_training_updates_encoder`,
`test_joint_hard_nearest_requires_retrieval_objective`).

## Results (average waypoint error, mm)

| Method | Frozen, 20ep (baseline) | Joint+contrastive, 20ep | Joint+contrastive, 200ep |
|---|---:|---:|---:|
| Direct | 36.52 | 36.55 | **32.22** |
| Nearest | 44.87 | 42.08 | 39.09 |
| Nearest + residual | 38.76 | 40.25 | 37.92 |
| Cartesian | 35.72 | 35.22 | 34.07 |
| Cartesian + residual | 36.39 | 34.80 | 34.55 |
| Latent + residual | 35.83 | 35.58 | **33.67** |

Every method improved 3-13% over the frozen baseline at 200 epochs; `direct` is now the overall
best performer (0.000541 m² test MSE), ahead of every retrieval-based method.

## Why `direct` didn't change at 20 epochs but did at 200

`training/trainer.py::pretrain_encoder` always trains the encoder jointly with a temporary
`direct`-style regression head for `pretrain_epochs`, regardless of `freeze_encoder` — so
`direct`'s own encoder training was already happening in both the "frozen" and "joint" 20-epoch
runs. `freeze_encoder` only changes what happens during each method's own `epochs` budget, which
is why retrieval methods moved at 20 epochs and `direct` only moved once given a bigger `epochs`
budget (200) than its pretraining stage already had (20).

## Per-method convergence differs a lot — flat epoch counts are wasting compute

Checked `history.json` for the 200-epoch run:

- `direct`: best validation epoch was **115** — genuinely still improving for over half the
  budget.
- `cartesian_residual`: best validation epoch was **24** — validation MSE only got noisier/worse
  after that (classic overfitting; train loss kept falling). The trainer's best-checkpoint
  selection (`train_policy`'s `if val < best: best, best_state = ...`) already protects the
  final result, so its 34.55mm here is effectively a ~25-epoch result, not a 200-epoch one.

So a flat `epochs: 200` for every method is significant wasted GPU time on methods that already
converged by epoch ~25; only `direct` (and likely other non-retrieval-shortcut methods) benefit
from the larger budget. Per-method epoch budgets (or a patience-based stopping rule reusing the
existing best-tracking) would get the same result for a fraction of the compute.

## Where this leaves things

Best method (`direct`, 32.2mm) is still close to the segments' own ~30mm mean net displacement —
real, measurable progress from the frozen baseline, but not a solved task. Two directions flagged
for next: (1) the retrieval mechanism (`nearest`/`cartesian`/latent variants) isn't beating plain
regression on this data even with joint+contrastive training, which raises the question of
whether retrieval is adding value here at all vs. being pure added complexity; (2) the underlying
data-diversity ceiling (~46 independent recording groups behind 611 `mode: segments` motions)
flagged earlier is still unaddressed and may be the larger lever.

Full artifacts: `/workspace/runs/deformpath_cuda_joint_seed7/` (20ep) and
`/workspace/runs/deformpath_cuda_joint200_seed7/` (200ep) on this host only.
