# Dough-relative anchor, generative policies, and a conditional-prior CVAE (new best)

Date: 2026-09-30. Follow-up to `RESULT_FOLLOWUPS_PROGRESS_HISTORY_NORMALIZATION.md`.

Code: `dom_retrieval` commits `941523f`/`41cd043`, `f195867`, `50fb984`, `6ccd0d1`, `295f301`, `4358ef2`,
`d5b214a`, `e99e02d` (all pushed). Every new option is **off by default**; usage is in `docs/RUNNING.md` §7.1.
Results page: https://claude.ai/artifact/3YZH4bojmHjFZDJfCtrd4r (version 21).

All runs: augmentation-units fix (`3f0b988`), `workspace_translation_m: 0.1` (the user's rule; ±0.3 m made the
start loss 7–10 std in normalized units and early stopping fired at epoch ~15), robust cloud scaling, seeds
7/17/27, test-set means (119–147 segments per seed) from `scripts/evaluate_deployment.py`.

## TL;DR

| 3-seed mean | Measured pose: start / shape / raw | Orient. | Motion size | Spread (truth ~20 mm) | Masked raw | Chained: measured / masked / fed own pose |
|---|---|---|---|---|---|---|
| Retrieval + dough-relative anchor | 12.9 / 21.7 / 22.4 mm | 5.9° | 85% | 11.8 mm | 31.1 mm | 21.8 / 30.0 / 38.3 mm |
| CVAE, KL 0.1, fixed N(0, I) prior | 14.9 / 23.8 / 25.2 mm | 6.1° | 89% | 7.8 mm | **29.7 mm** | — / **28.5** / 29.2 mm |
| **CVAE, KL 0.1, conditional prior** | **11.6 / 21.4 / 22.3 mm** | **5.8°** | **93%** | **13.6 mm** | 30.4 mm | **21.5** / 28.9 / **28.7** mm |
| ACT (+ anchor, min 150 epochs) | 21.3 / 26.1 / 28.8 mm | 7.4° | 86% | 8.3 mm | 30.9 mm | — / 28.6 / 28.3 mm |
| Diffusion Policy (1 seed) | 18.2 / 28.6 / 31.8 mm | 8.5° | 70% | 5.5 mm | 35.4 mm | — / 34.1 / — mm |

**New best: the CVAE with a conditional prior.** Checkpoint used for the simulator segment:
`/workspace/runs/cvae_cprior/cvae_seed27_out/cvae/checkpoint.pt` (raw 22.8, start 10.5, shape 21.9 mm, 5.7°).
Retrieval's best single checkpoint (`/workspace/runs/rel_anchor/cond_rel_t01_seed27_out`, raw 21.9 mm) is
still competitive with a measured pose; it degrades much more when fed its own predicted pose (38 vs 29 mm).

## 1. Dough-relative anchor (`path_start_relative_to_dough`)

The start head predicted absolute positions, so it had to learn "start = dough centroid + offset" over the
whole workspace. Under correct ±0.3 m augmentation the loss started at ~200 and early stopping fired at epoch
13–15. Now the head predicts each tool's start relative to the dough centroid (point mean) and the centroid is
added back; the absolute centroid is dropped from its input. Translating the scene translates the start exactly
(unit test, pose measured and masked). Starting loss ~11, runs train 133–242 epochs.
Retrieval, 3 seeds: measured start / shape / raw 14.2 / 22.3 / 24.2 → **12.9 / 21.7 / 22.4 mm**, 6.3 → 5.9°;
pose masked essentially unchanged (30.8 → 31.1 mm raw). Also implemented for the generative policies.

## 2. Generative policies (added to dom_retrieval by AntonioSkara132, `eb12045`)

One seed each first (27), same data, split, frozen InfoNCE PointNet and anchor settings as retrieval.
- **CVAE:** at the config's `kl_weight: 0.001` it collapsed to one motion for every input (spread 0.3 mm): the
  decoder relied on the latent code, which is 0 at test time. `kl_weight: 0.1` fixed it (spread 6.7 mm).
- **Diffusion Policy:** worse than the dataset-mean path on shape (28.6 vs 27.5 mm) after 725 epochs. Not
  pursued with 437 training strokes.
- **ACT:** the most varied strokes of the first round (spread 14.1 mm) but less accurate. With the anchor its
  validation plateaus after ~16 epochs and patience 50 stopped it at 129–168 epochs; `early_stopping.min_epochs`
  (`50fb984`, ACT config 150) did not help: validation rises after ~130 epochs (overfitting), start stays ~21 mm.
  Diagnosed causes: 2.74 M trainable parameters (CVAE 0.39 M) including raw point-group tokens learned from
  437 strokes, and `policy_learning_rate: 1e-4` for everything including the start head (CVAE 1e-3).

## 3. Generalization and "one typical stroke"

- The fixed-prior CVAE did not overfit but underfit: training-segment shape error ≈ test (26.7 vs 25.0 mm on
  seed 27, close to the 27.5 mm mean path); its predictions varied a third as much as real strokes (7 vs
  21 mm), and giving every segment its own average path cost only 2.5 mm.
- Seed differences came from the split: seed 7 kept improving to 393–467 epochs (shape 22.9 mm, spread 11 mm),
  seeds 17/27 plateaued at 52/97. `early_stopping.smoothing_window` (`6ccd0d1`: stop on the median of the last N
  validation losses, model selection still on the raw minimum) confirmed their plateau was real, not noise.

## 4. Conditional prior (`cvae.conditional_prior`, `295f301`)

A prior network predicts p(z | dough, pose); the KL is taken against it and prediction decodes at its mean.
- Per seed, prediction spread 5.0–11.1 → 11.5–12.8 mm, and that variation is now worth 5.3–6.5 mm of shape
  error (was 1.8–4.0). All seeds train 358–451 epochs and land within ~1 mm.
- Converged: smoothed validation gained only 5–7% in the last 100 epochs, training loss flat. Train/test shape
  gap 1.5–3.8 mm (it fits its training strokes, 18–20 mm, without copying them).
- All 125 seed-27 test strokes from a common start (page figure): recorded 72 / 69 mm size, 20–21 mm spread;
  conditional prior 71 / 63 mm, 11–12 mm; fixed prior 69 / 57 mm, 6–7 mm; retrieval 65 / 56 mm, 10–12 mm.
- Sampling from the prior (seed 27, 16 samples): mean prediction shape 21.9 mm, one random sample 22.8 mm,
  best of 16 in hindsight 16.8 mm; samples for one input differ by ~6.7 mm.

## 5. Simulation (TaichiDough, episode 18 pc137–154, seed 27, open loop, fixed orientation, recorded timing)

| Condition | Path length t1/t2 | Tool error t1/t2 | Displacement | Chamfer | Per-particle offset |
|---|---|---|---|---|---|
| recorded | 191 / 185 mm | — | 13.1 mm | 0 | 0 |
| CVAE cond. prior, pose measured | **188 / 165 mm** | 20.5 / 13.8 mm | 18.7 mm | 4.13 mm | 12.1 mm |
| CVAE cond. prior, pose masked | 198 / 173 mm | 34.5 / 20.3 mm | **13.2 mm** | 3.58 mm | **11.0 mm** |
| CVAE fixed prior, measured / masked | 174–197 / 157–173 mm | 29–33 / 14–16 mm | 18.4 mm | 3.86–4.04 mm | 13.2 / 14.2 mm |
| ACT, measured / masked | 165–167 / 152–154 mm | 35–36 / 19–20 mm | 9.3–9.9 mm | 3.22–3.25 mm | 10.2 / 11.6 mm |
| retrieval, measured / masked | 178–183 / 149–165 mm | 21–26 / 14 mm | 16–22 mm | 3.56–3.71 mm | 16.5 / 13.0 mm |
| tools held still | 0 | 46 / 39 mm | 0.2 mm | **3.21 mm** | 13.2 mm |

The conditional-prior CVAE gives the closest tool motion so far, but no policy carves the recorded cavity and
none beats holding still on Chamfer. In the three look-alike runs (CVAE measured/masked, retrieval masked)
most error is a shared ~2 cm constant start offset carried through the whole stroke. One segment only.

## 6. Simulator-based selection among samples

- `sim/batch_simulate.py` (`e99e02d`): one persistent process, setup once (~35 s), then **~2.4 s per stroke**
  (physics alone; 24 k particles, 48³ grid, 2,669 steps for 0.534 s ≈ 4× slower than real time). Reproduces the
  per-run results exactly. Parallel processes: physics of 17 strokes in ~13 s GPU time, but per-process CPU
  setup dominates (~2 min for 17).
- 16 conditional-prior samples + mean on chunk06: simulated offset 10.5–15.8 mm (mean 12.1, hold 13.2); best of
  17 by outcome is 1.6 mm better than the mean. **Offline imitation accuracy is anti-correlated with the
  simulated outcome here** (Spearman −0.68 shape, −0.76 raw): every sample pushes more dough than the
  demonstration (15–23 vs 13.1 mm) and the gentler ones end closer. A learned outcome surrogate would therefore
  have to be trained on simulated outcomes, not on trajectory similarity.
- The samples for one input form a 3–6 mm bundle around one loop (same start: the start head is deterministic);
  none explores the human's downward tool-1 sweep. Figure:
  `/workspace/runs/sample_select/chunk06/samples_same_input.png`.
- TaichiDough's differentiable MPM differentiates particle state and material/contact parameters, not tool
  poses, and contact-rich backward passes are unqualified (334 steps without contact verified; CUDA failures
  near step 9,770). Gradient-based latent optimization would need a tool-pose adjoint first.

## Follow-up results (same day)

All 3 seeds (7/17/27), same test splits; differences under ~1 mm are treated as ties (run-to-run noise ~±1 mm,
and these splits were also used for earlier decisions, so there is no untouched test set yet).

| Model | Measured: start / shape / raw | Pose masked raw | Chained masked / fed own pose |
|---|---|---|---|
| CVAE + conditional prior (previous best) | 11.6 / 21.4 / 22.3 mm | 30.4 mm | 28.9 / 28.7 mm |
| + `cvae.joint_start` (start decoded with the stroke) | 12.5 / 20.7 / 22.1 mm | 28.4 mm | 27.3 / 31.4 mm |
| + joint start + `fixed_scale_m: 0.05` (**new best config**, `configs/deformpath_cvae_best.yaml`) | 12.2 / 20.9 / 22.2 mm | 28.6 mm | 27.1 / 28.6 mm |
| + `path_start_from_motion` (separate start head reads the stroke) | 16.4 / 22.4 / 24.3 mm | 28.5 mm | 26.9 / 31.4 mm |
| Chamfer nearest neighbour, no network (k = 5) | start = measured pose / 28.6 / 28.6 mm | 31.1 mm | — |
| Dataset-mean stroke | — / 27.5 / 27.5 mm | — | — |

- **Joint start** is the only change that improves the pose-masked mode on every seed (~2 mm); measured pose ties.
  Samples now get their own starts (~5 mm apart) and more stroke variety (9.1 vs 6.7 mm), but best-of-16 does not
  improve. Start from the stroke costs 4–5 mm of measured start: dropped.
- **Fixed-scale clouds** (dough size visible to the PointNet): tied on every metric (±0.5 mm, no consistent
  direction). The user chose to keep it (size is a meaningful quantity for the task).
- **Chamfer nearest neighbour** by start-cloud similarity is worse than the dataset-mean stroke (k = 1: 39 mm). The
  learned models (~21 mm) do real work beyond geometric similarity; without a pose, though, the learned start is
  no better than a neighbour's start offset from the dough centroid (24.8 vs 24.5 mm).
- **Prior coverage** (16 samples per test segment, seed 27): samples vary one family of loop strokes; atypical
  recorded strokes (long sweeps, reversed direction, bigger loops) never appear. Pose masked, a sample starts within
  1 cm of the recorded start in only 10% of segments. With a measured pose the start is still ~12–14 mm off because
  training adds ±2 cm pose noise; placing the start at the measured pose is the untested fix.
- **Adversarial review** of the programme: no untouched test set, sub-mm differences reported as rankings, pre-fix
  "no gain" verdicts not re-run, simulation on one non-held-out segment. Proposed: lock an evaluation protocol
  (frozen held-out episodes, paired per-segment comparisons, pose-masked/chained as primary) and a multi-chunk
  simulator benchmark before further variants.

Docs: `dom_retrieval/docs/RUNNING.md` 7.1 (`37515d6`); analysis scripts in `dom_retrieval/scripts/analysis/`.

## Next

- Start at the measured pose when available (no retraining); prior temperature / lower KL for coverage.
- Locked evaluation protocol and multi-chunk (13 calibrated chunks) simulator benchmark.
