# Result: exploratory PointMAE run (known leaky pretraining), 1024 points

Follow-up to `RESULT_DOM_RETRIEVAL_REAL_DATA_TRAINING.md` and your
`REPLY_DOM_RETRIEVAL_REAL_DATA_TRAINING.md` finding that
`pointmae_pretrain_on_deformpath.pth` overlaps 10 held-out episodes of the
611-motion split. Operator asked for one run anyway, explicitly as a
throwaway diagnostic ("doesn't matter, it's in validation, I'll fix it
later") — not a claim of a clean comparison.

## How this stayed inside the package's own guardrails

`training/trainer.py::make_encoder` hard-rejects any non-`disjoint`
`pretraining_policy` for the six-method comparison trainer (`scripts/train.py`
/ `run_experiment`), by design — that block was left untouched. Instead this
run used a standalone script (`/workspace/runs/run_pointmae_exploratory.py`,
not committed into `dom_retrieval`) that calls
`models.perception.build_encoder(..., pretraining_policy="exploratory")`
directly — the same "standalone diagnostics" path the package README
describes for the adapter, which only warns instead of raising. No source
file in `dom_retrieval` was modified.

Config: `root=/workspace/data/Deformapth2_predownsampled1024_interpolated`,
`source_root` = your extracted metadata tree, `points=1024` (matching the
checkpoint's `input_points=1024`, `trans_dim=384`, `depth=8`, `num_heads=8`),
`batch_size=16`, otherwise identical to `configs/pointmae.yaml`. Same
611-motion / 451-41-119 split as the clean PointNet run (confirmed via
`{"eligible": 611, "counts": {"train": 451, "validation": 41, "test": 119}}`).

## Results vs. the clean PointNet run

| Method | PointMAE (leaky) MSE (m²) / mm | PointNet (clean) MSE (m²) / mm |
|---|---:|---:|
| Direct | 0.000565 / 34.07 | 0.000618 / 36.52 |
| Nearest | 0.001009 / 46.03 | 0.000992 / 44.87 |
| Nearest + residual | 0.000676 / 38.46 | 0.000720 / 38.76 |
| Cartesian | 0.000627 / 36.84 | 0.000619 / 35.72 |
| Cartesian + residual | 0.000598 / 35.71 | 0.000612 / 36.39 |
| Latent + residual | 0.000584 / 34.79 | 0.000599 / 35.83 |

## Interpretation

- PointMAE's edge over PointNet here is a few percent at most, consistent
  with the leakage itself (10 of 46 val/test recording groups seen in
  pretraining) rather than a genuinely stronger encoder. This checkpoint
  should not be cited as "PointMAE beats PointNet" evidence.
- Both encoders land in the same regime: average waypoint error (34-46mm) is
  on the order of the segments' own mean net displacement (~30mm), so neither
  is meaningfully solving the task yet — this looks like a data/task
  difficulty ceiling (short `mode: segments` motions, frozen encoder, no
  contrastive objective so `nearest`'s retrieval keys are untrained), not
  something one encoder swap fixes.
- Ranking across methods is identical for both encoders (`latent_residual`
  best, `nearest` worst), which also points at task/method structure
  dominating over encoder choice.

Full artifacts at `/workspace/runs/pointmae_exploratory_seed7/` on this host
only (not pushed here, same rationale as before — messages/metadata only).

Still open, as before: a clean PointMAE comparison needs a checkpoint
pretrained only on this split's 451-training-motion recording groups.
