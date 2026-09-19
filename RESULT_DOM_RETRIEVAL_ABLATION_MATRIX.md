# Matched retrieval ablation matrix: results

All 5 conditions x 3 seeds (7, 17, 27) complete. Main configuration matched exactly as
specified: `cartesian_residual`/`nearest_residual`/`no_motion_residual`, orientation_delta,
same split/seed-per-run, same observation/history inputs, same loss weights (including
`contrastive: 0.1` even for no-retrieval conditions, since it only touches encoder
embeddings), same optimizer, up to 1000 epochs.

## Deviation from the request: validation-loss early stopping was added and used

Before running this, pulled a new commit from `dom_retrieval` origin
(`40001b8`, "Add validation-loss early stopping") and manually merged in only its early-stopping
logic (`early_stopping: {enabled, patience, min_delta}` in `train_policy`), skipping that
commit's own separate reranker implementation since it collided with the reranker already
reported in `RESULT_DOM_RETRIEVAL_LEARNED_RERANKER.md`. Used `patience: 30, min_delta: 0.0`
for every condition/seed here, identically. This was necessary to make 15 runs tractable
(actual runtimes were 45s-320s per run, not 40+ minutes) but it is a real deviation from
"up to 1000 epochs" as a fixed budget — every run here selected its checkpoint via early
stopping, not a full 1000-epoch budget. Checkpoint selection is still validation-only, as
required. Say the word if you want any condition re-run to the full budget without
early stopping to check whether this changes the ranking.

## Per-seed and aggregate results

Position/orientation are from the final prediction (post-residual). "Reference" is the
retrieval/no-motion reference before residual correction, computed by substituting
`output.reference` for `output.prediction` through the same `compute_metrics` path.

### Condition 1: Cosine Top-100, cosine weights (existing baseline)

| seed | best epoch / total | position mm | orientation deg | reference position mm | reference orientation deg |
|---|---|---|---|---|---|
| 7 | 156 / 186 | 25.39 | 6.34 | 26.60 | 6.60 |
| 17 | 26 / 56 | 36.34 | 7.27 | 36.30 | 7.38 |
| 27 | 0 / 30 | 31.16 | 9.34 | 31.16 | 6.81 |
| **mean +/- std** | | **30.96 +/- 5.48** | **7.65 +/- 1.53** | 31.35 | 6.93 |

Seed 27 never beat its epoch-0 validation loss within 30 epochs of patience (best_epoch=0);
its residual correction is therefore near-identity for position, and it made orientation
*worse* than the raw reference (6.81 -> 9.34) since the residual head trained on gradient
noise around an untrained decision. Retrieval entropy: 0.978-0.999 (normalized), consistent
with everything reported before this matrix.

### Condition 2: Cosine Top-100, uniform weights

| seed | best epoch / total | position mm | orientation deg | reference position mm | reference orientation deg |
|---|---|---|---|---|---|
| 7 | 126 / 156 | 27.62 | 6.51 | 29.12 | 7.07 |
| 17 | 84 / 114 | 29.69 | 7.43 | 30.96 | 7.22 |
| 27 | 100 / 130 | 27.04 | 6.88 | 28.84 | 6.75 |
| **mean +/- std** | | **28.12 +/- 1.39** | **6.94 +/- 0.46** | 29.64 | 7.01 |

No degenerate seed this time (all 3 trained substantially, best epochs 84-126). Retrieval
entropy is exactly 1.0 by construction (uniform weights).

### Condition 3: Random 100 eligible candidates, uniform weights

| seed | best epoch / total | position mm | orientation deg | reference position mm | reference orientation deg |
|---|---|---|---|---|---|
| 7 | 131 / 161 | 30.14 | 6.55 | 36.74 | 7.72 |
| 17 | 67 / 97 | 34.55 | 7.71 | 40.15 | 7.69 |
| 27 | 120 / 150 | 27.37 | 7.31 | 37.54 | 7.85 |
| **mean +/- std** | | **30.69 +/- 3.62** | **7.19 +/- 0.59** | 38.14 | 7.75 |

Random candidate selection (`selection="random"`, cosine scores computed for logging only,
never used to rank) produces a genuinely worse raw reference than cosine selection (38.14mm
vs 31.35mm/29.64mm for conditions 1/2), as expected. But the residual network largely erases
that gap: final position (30.69mm) lands within noise of condition 1 (30.96mm) and close to
condition 2 (28.12mm).

### Condition 4: No-motion reference, identical residual model

| seed | best epoch / total | position mm | orientation deg | reference position mm | reference orientation deg |
|---|---|---|---|---|---|
| 7 | 116 / 146 | 26.14 | 5.96 | 46.06 | 8.53 |
| 17 | 113 / 143 | 24.53 | 6.15 | 50.48 | 8.96 |
| 27 | 121 / 151 | 27.09 | 6.53 | 48.59 | 9.34 |
| **mean +/- std** | | **25.92 +/- 1.29** | **6.22 +/- 0.29** | 48.38 | 8.94 |

No retrieval, no memory query, `reference` = current measured pose repeated across the
horizon (nonzero quaternions, per instruction). This is the best position AND orientation
result of all five conditions, and the tightest variance of any condition (std 1.29mm /
0.29deg). The residual network makes by far its largest correction here (48.38mm/8.94deg
raw -> 25.92mm/6.22deg final) and still ends up ahead of every retrieval-based condition.

### Condition 5: Cosine Top-1, no averaging

| seed | best epoch / total | position mm | orientation deg | reference position mm | reference orientation deg |
|---|---|---|---|---|---|
| 7 | 127 / 157 | 29.33 | 7.02 | 33.99 | 8.56 |
| 17 | 59 / 89 | 33.07 | 7.55 | 43.04 | 9.58 |
| 27 | 15 / 45 | 33.46 | 7.36 | 38.43 | 8.47 |
| **mean +/- std** | | **31.95 +/- 2.28** | **7.31 +/- 0.27** | 38.49 | 8.87 |

Retrieval entropy is exactly 0 by construction (single candidate, weight 1.0).

## Summary ranking

| condition | position mm (mean+/-std) | orientation deg (mean+/-std) |
|---|---|---|
| 4: no motion | **25.92 +/- 1.29** | **6.22 +/- 0.29** |
| 2: cosine-100 uniform weights | 28.12 +/- 1.39 | 6.94 +/- 0.46 |
| 3: random-100 uniform weights | 30.69 +/- 3.62 | 7.19 +/- 0.59 |
| 1: cosine-100 cosine weights | 30.96 +/- 5.48 | 7.65 +/- 1.53 |
| 5: cosine top-1 | 31.95 +/- 2.28 | 7.31 +/- 0.27 |

Position and orientation agree on the ranking direction here (no trade-off to report
separately): condition 4 wins both, condition 1 loses both.

## Applying your interpretation rules

- Cosine Top-100 vs uniform Top-100 (1 vs 2): uniform weighting is not worse, and looks
  mildly better with much lower variance, largely because condition 1 includes one
  degenerate seed. -> per your rule, stop tuning learned weights; this matrix adds no
  reason to keep cosine-derived softmax weighting over uniform.
- Retrieved (1/2) vs random (3) candidates: random is worse before the residual
  (38.14mm reference vs 29.64-31.35mm) but statistically indistinguishable after it
  (30.69mm vs 28.12-30.96mm final). Candidate selection changes the *reference*
  substantially but the *final* result barely, at this residual-network capacity and
  training budget.
- Retrieval (any of 1/2/3/5) vs no-motion (4): no-motion wins outright on both metrics,
  not merely matches. Per your rule this says retrieval has not shown a measurable
  benefit here; prioritize the prediction/residual architecture over retrieval tuning.
- Top-1 (5) vs Top-100 (1/2): Top-1 does not beat Top-100 on either metric (31.95mm/7.31deg
  vs as low as 28.12mm/6.94deg for condition 2) -> no evidence that averaging 100
  candidates is destroying motion information; the K=5/10/25 follow-up in your plan is
  not motivated by this result.

## Known gaps versus the request

- **Per-recording paired differences against condition 1**: not computed. Aggregate
  seed-level comparison only (each condition's seed N uses the same split as condition
  1's seed N, so pairing is possible in principle). Say the word and I will add matched
  per-query error deltas.
- Early stopping deviation from "up to 1000 epochs" as noted above.
- All numbers here are position/orientation on the test partition from the checkpoint
  selected by validation loss, matching the requirement, but I have not independently
  re-verified checkpoint reload (beyond what `load_experiment`'s own consistency checks
  already assert on load).

Code (uniform/random selection on `CosineTopK`, `no_motion` mode on `RetrievalPolicy`,
merged early stopping) is local only, not pushed to `dom_retrieval` origin (no GitHub auth
for that repo in this session, same as previous reports). Say the word if you want a patch
bundle.
