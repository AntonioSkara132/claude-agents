# Paper Results Tables

**Status:** fact-checking source for the DeformPath paper  
**Repository state used:** `6b31158` on 2026-10-05; raw per-seed and simulator outputs committed under `data/` (regenerate every number with `python data/make_tables.py`; checkpoint hashes in `data/checkpoint_sha256.txt`)  
**Rule:** do not rank differences below approximately 1 mm; the current results are development-test estimates, not an untouched final evaluation.

This file collects the tables currently intended for the paper. Every number must remain traceable to a committed result report. Agents updating this file should preserve the protocol notes and mark any value that is not directly supported by a repository report.

## Experimental protocol

Canonical source: [`RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md`](RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md), especially lines 9--21 and 108--135.

- Seeds: 7, 17, and 27 unless otherwise marked.
- Each seed defines a different episode-grouped split, so variation includes both split composition and training randomness.
- Test sets contain 119--147 segments per seed.
- Point clouds contain 1,024 real points.
- All position errors are lower-is-better.
- Start error is the first-waypoint Euclidean error.
- Shape error is waypoint error after independently removing predicted and true start translation.
- Raw error scores all waypoints without alignment.
- Pose-masked evaluation removes measured pose input.
- Chained fed-own-pose evaluation supplies the preceding model prediction to the next segment.
- Values within approximately 1 mm should be treated as ties.
- The same family of test splits informed model selection; an untouched final test set is still required.

## Table 1: Offline policy comparison

Corrections 2026-10-05 (from the raw per-seed outputs in `data/deployment/`): ACT row now entirely from the min-150-epoch run; the fixed-prior CVAE row from the smoothed-early-stopping run (the earlier 23.8 / 29.7 mixed two runs); the selected CVAE's orientation and motion size filled in; in Table 2 the selected CVAE's chained-measured value was the non-chained raw error (22.2) and is now 21.6; Table 3's hold-still mean was misreported as 3.65 (the 13 values average 3.57, as in Table 4).

Source: [`RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md`](RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md), main comparison at lines 15--21 and selected-configuration/Chamfer follow-up at lines 108--127.

| Method | Start error [mm] ↓ | Shape error [mm] ↓ | Raw error [mm] ↓ | Orientation [deg] ↓ | Motion size [%] | Pose-masked raw [mm] ↓ | Seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dataset-mean stroke | measured* | 27.5 | 27.5 | 8.5† | 73 | 31.4‡ | 3 |
| Chamfer NN, k=5 | measured* | 28.6 | 28.6 | n/a (positions only) | 82 | 31.1‡ | 3 |
| Retrieval + dough-relative anchor | 12.9 | 21.7 | 22.4 | 5.9 | 85 | 31.1 | 3 |
| ACT, anchored, minimum 150 epochs | 21.3 | 26.0 | 28.9 | 7.4 | 85 | 30.9 | 3 |
| Diffusion Policy | 18.2 | 28.6 | 31.8 | 8.5 | 70 | 35.4 | 1 |
| CVAE, fixed prior (smoothed early stopping) | 14.9 | 24.5 | 25.2 | 6.1 | 89 | 30.0 | 3 |
| CVAE, conditional prior | 11.6 | 21.4 | 22.3 | 5.8 | 93 | 30.4 | 3 |
| Selected CVAE, joint start + fixed 5 cm cloud scale | 12.2 | 20.9 | 22.2 | 5.6 | 90 | 28.6 | 3 |

### Interpretation

- Retrieval and the selected CVAE are tied on measured-pose raw error: 22.4 versus 22.2 mm.
- Retrieval reduces raw error by 22.2% relative to ACT: 22.4 versus 28.8 mm.
- Retrieval reduces shape error by 24.1% relative to Chamfer NN: 21.7 versus 28.6 mm.
- The selected CVAE reduces shape error by 26.9% relative to Chamfer NN: 20.9 versus 28.6 mm.
- Diffusion Policy is a one-seed result and should remain visibly marked as such.
- *Both baselines start exactly at the measured pose (start error 0 by construction), so their start column is not comparable to learned start prediction. †Orientation of the dataset-mean stroke = the training-mean quaternion per tool (8.5°); Chamfer NN blends positions only. ‡Pose-masked raw for the baselines places the stroke at the training-mean start offset from the dough centroid (dataset mean) or the neighbours' start offset (Chamfer NN); masked start error 27.5 and 24.5 mm. Source: `data/dataset_mean_stroke.json`, `data/chamfer_nn.json`.

## Table 2: Robustness to missing and recursively predicted pose

Source: [`RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md`](RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md), lines 15--21 and 111--116.

| Method | Pose-masked raw [mm] ↓ | Chained measured [mm] ↓ | Chained masked [mm] ↓ | Chained fed-own-pose [mm] ↓ |
|---|---:|---:|---:|---:|
| Retrieval + dough-relative anchor | 31.1 | 21.8 | 30.0 | 38.3 |
| ACT, anchored, minimum 150 epochs | 30.9 | 28.2 | 28.7 | 28.7 |
| Diffusion Policy (1 seed) | 35.4 | 31.3 | 34.1 | 37.9 |
| CVAE, fixed prior (smoothed early stopping) | 30.0 | 24.1 | 28.5 | 29.2 |
| CVAE, conditional prior | 30.4 | 21.5 | 28.9 | 28.7 |
| Selected CVAE, joint start + fixed 5 cm cloud scale | 28.6 | 21.6 | 27.1 | 28.6 |

### Interpretation

- The selected CVAE reduces fed-own-pose error by 25.3% relative to retrieval: 28.6 versus 38.3 mm.
- ACT is competitive in chained masked and fed-own-pose evaluation despite its worse measured-pose offline accuracy.
- Measured, masked, and fed-own-pose modes answer different deployment questions and must not be collapsed into a single ranking.

## Table 3: Simulator search over 13 calibrated chunks

Canonical source: [`RESULT_SIM_SEARCH_PROGRESS_REWARD_TIMING.md`](RESULT_SIM_SEARCH_PROGRESS_REWARD_TIMING.md), lines 3--40.

All chunks come from episode 18. CMA-ES performs 48 simulator evaluations per test chunk and directly minimizes final-state Chamfer. It is a per-instance search upper bound, not a feed-forward policy with the same test-time budget.

| Condition | Mean final Chamfer [mm] ↓ | Beats hold still | Beats policy |
|---|---:|---:|---:|
| Tools held still | 3.57 | (reference) | 10/13 |
| Policy stroke | 4.27 | 3/13 | (reference) |
| Per-chunk CMA-ES search | 2.91 | 13/13 | 13/13 |

### Interpretation

- The feed-forward imitation policy is worse than holding still on average: 4.27 versus 3.57 mm.
- Per-chunk search reduces Chamfer by 31.9% relative to the policy and by 18.5% relative to holding still.
- Search beats both policy and hold-still on all 13 chunks, showing latent-space headroom.
- Approximate simulator/search noise is 0.4 mm; CMA-ES beats hold still by more than that margin on 8/13 chunks and the policy on 12/13 (`data/sim/cma_sweep_v2`).
- Final-state Chamfer can reward inactivity and is not sufficient by itself to establish task success.

## Table 4: Leave-one-chunk-out transfer of search corrections

Source: [`RESULT_SIM_SEARCH_PROGRESS_REWARD_TIMING.md`](RESULT_SIM_SEARCH_PROGRESS_REWARD_TIMING.md), lines 42--56.

These are separate reruns from Table 3 (the hold-still and policy rollouts were repeated and agree to 0.01 mm; the simulator is nondeterministic only at the ~0.4 mm level under parallel load). Constant-offset and fine-tuned-prior methods receive no per-instance search on the held-out chunk.

| Held-out-chunk method | Mean final Chamfer [mm] ↓ | Beats policy |
|---|---:|---:|
| Hold still | 3.57 | 10/13 |
| Original policy | 4.27 | (reference) |
| Constant mean offset learned from the other chunks | 4.07 | 9/13 |
| Fine-tuned conditional prior learned from the other chunks | 4.25 | 7/13 |
| Per-chunk CMA-ES search upper bound | 2.91 | 13/13 |

### Interpretation

- The large per-chunk search gain does not transfer through either a constant correction or a fine-tuned prior.
- Thirteen chunks from one episode are insufficient evidence of episode-level generalization.
- Table 4 must not be merged with Table 3 without preserving the different rerun protocol.

## Optional Table 5: Recorded-cloud segment dynamics

Source: [`RESULT_REDUCED_MODELS_AND_TRANSLATION_TRACKING.md`](RESULT_REDUCED_MODELS_AND_TRANSLATION_TRACKING.md), lines 28--55.

Grouped split: 451 training, 41 validation, and 119 test strokes. This predicts a coarse height-map/PCA outcome near demonstrated trajectories; it is not a replacement for the full simulator.

| Model | Standardized outcome MSE ↓ | Mean candidate rank among 16 ↓ | Top-1 stroke identification ↑ |
|---|---:|---:|---:|
| Persistence / chance ranking | 1.25 | 8.5 | 6% |
| Mean change | 1.23 | n/a (stroke-independent) | n/a |
| Constant-Jacobian ridge | 0.65 | 4.9 | 23% |
| State-dependent Jacobian MLP | 0.77 | 5.9 | 16% |

### Interpretation

- The linear ridge model reduces standardized outcome MSE by 48% relative to persistence.
- The simpler ridge model also ranks candidate strokes better than the state-dependent neural model.
- Simulator agreement and progress-reward agreement remain untested.

## Conclusion-ready quantitative statements

Use these only with the caveats above:

1. Across three episode-grouped development splits, retrieval reduces raw trajectory error by 22% relative to ACT (22.4 versus 28.8 mm) and shape error by 24% relative to Chamfer nearest-neighbour retrieval (21.7 versus 28.6 mm).
2. The selected CVAE and retrieval are tied at approximately 22 mm measured-pose raw error, but the selected CVAE reduces chained fed-own-pose error by 25% (28.6 versus 38.3 mm).
3. The feed-forward simulator policy is worse than holding still on average (4.27 versus 3.57 mm Chamfer), while 48-rollout per-chunk CMA-ES reaches 2.91 mm and beats both on 13/13 chunks.
4. The per-chunk search improvement does not transfer reliably: leave-one-chunk-out constant-offset and fine-tuned-prior corrections obtain 4.07 and 4.25 mm, respectively, versus 4.27 mm for the original policy.
5. A constant-Jacobian ridge dynamics model reduces standardized outcome MSE from 1.25 to 0.65 and improves 16-way stroke top-1 identification from 6% chance to 23%, but only for a coarse state representation near demonstrations.

## Excluded or historical results

Do not add these to the main paper table without new verification or matched reruns:

- The artifact-only “best of three modes in hindsight” value of 18.4 mm / 85% is not present in repository history.
- Legacy VINN-RBF, direct-GRU, and Motion-BeT rows predate the workspace-translation augmentation fix and are not directly rankable against the final protocol.
- External ACT-BeT uses a different export, horizon, split, and checkpoint history.
- Single-segment simulator comparisons are diagnostic only.
- Reduced-order MPM results use one reconstructed dough and twelve short windows; the committed report explicitly states that the model is not usable.
- Progress-reward and prior-sampling experiments show search headroom and sim-to-real failure, not a successful learned policy improvement.

## Fact-check checklist for future agents

- [ ] Confirm that every table value still matches the cited committed report after future pulls.
- [x] Checkpoint hashes (`data/checkpoint_sha256.txt`), per-seed JSONs (`data/deployment/`) and simulator outputs (`data/sim/`) are committed; `data/make_tables.py` regenerates the numbers.
- [ ] Recompute uncertainty from per-segment paired errors; do not infer significance from sub-millimetre mean differences.
- [ ] Run all main baselines on one locked, untouched episode-level test set.
- [ ] Keep one-seed rows visibly marked.
- [ ] Keep feed-forward policies separate from methods that use simulator search at test time.
- [ ] Keep direct-sweep and leave-one-chunk-out simulator reruns separate.
- [ ] Define whether the paper's primary method is retrieval or the selected CVAE before using “our method” in captions or conclusions.
