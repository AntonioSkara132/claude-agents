# Test dough/tool coordinates against absolute coordinates

Requested: 2026-09-19. Framework revision: `e6a091a` (`dough frame`) or a later revision containing it.

Please run a matched CUDA comparison of `coordinates.mode: absolute` and `coordinates.mode: dough_tool_frame`. The question is whether removing dependence on global translation and heading improves held-out trajectory prediction. No improvement is assumed.

## Calibration comes first

Use `configs/dough_tool_frame.yaml`. Its `coordinates.table_normal` and `coordinates.table_normal_source` are intentionally null: supply a verified table normal in the **policy input point-cloud coordinate frame**, and record its calibration provenance. Do not assume input Z is vertical, reuse simulator Y-up as input coordinates, or apply an episode18-specific normal to every recording without verifying common alignment.

Check the coordinate frames across recordings. If they differ, report the required alignment rather than quietly applying a single incorrect normal. If calibration cannot be established, report the missing information and stop the real-data experiment; a guessed normal is not a valid test.

The implemented frame uses the current cloud centroid, calibrated table normal, and horizontal tool0-to-tool1 direction. One current-time frame transforms the whole input history and output trajectory. Distances stay in metres. Each training-memory demonstration uses its own frame; retrieved local paths are mapped into the query frame for scene-space export. Future targets must never determine the frame.

## Four matched runs

Create two resolved configs from the supplied example, differing only in coordinate mode (and output directory). Run both methods in each:

| Method | Absolute | Dough/tool frame |
|---|---|---|
| Full-history direct | fresh baseline | fresh experiment |
| Full-history Cartesian residual | fresh baseline | fresh experiment |

Keep these settings identical:

- Seed 7, the 611-motion export and the established group-disjoint split: 451 train, 41 validation, 119 test. Verify exact example/group IDs against the prior campaign, not just counts.
- Predownsampled/interpolated 1024-point inputs, 32 output waypoints, six XYZ output channels.
- Eight history frames, 0.7-second window, 0.2-second maximum gap; full history with measured tool positions.
- PointNet embedding 32 / hidden 64; history hidden 64.
- K=100, temperature 0.1, joint encoder training, 20 pretraining epochs, 1000 training epochs, batch 32, learning rate 0.001.
- Loss weights: trajectory 1.0, residual 0.001, acceleration 0.0001, contrastive 0.1.
- Same best-validation-checkpoint selection rule and training budget. Do not introduce early stopping for only one condition.

Retain measured proprioception. Do not add static quaternion inputs, pose-noise ablations, a controller approach phase, or a yaw-minimized trajectory loss to this comparison. Temporal history already provides tool positions. Train-only normalization is fitted separately in each coordinate representation, as implemented; this means the experiment compares the implemented coordinate pipelines, including their normalization, rather than isolating rotation from every loss-weighting effect.

Training retrieval memory must contain only training trajectories. Test and validation trajectories must never enter it. Fit no learned preprocessing statistics on held-out data.

## Preflight and preservation

Use the established isolated checkout workflow; do not modify the protected `/workspace/dom_retrieval` checkout or delete/overwrite anything in `/workspace/checkpoints`. Fetch the implementation into an approved separate checkout if needed. Use fresh output directories and preserve resolved configs, revision IDs, split IDs, dataset snapshots, training logs and selected checkpoints.

Before the long runs:

1. Run the unit suite from the package parent: `python -m unittest discover -s dom_retrieval/tests -q`. The local implementation passed 80 tests.
2. Run a short CUDA smoke experiment for both modes and both methods. Reload each saved checkpoint with its own dataset snapshot and evaluate/export a query. Use separate directories from the full runs.
3. Verify scene-space target and prediction exports, finite values, and train-only retrieval IDs. Use the supported loader without disabling snapshot or split checks.
4. Count `model_coordinates.heading_fallback` per partition and list affected IDs. Below 5 mm horizontal tool separation, the deterministic scene-axis fallback is not yaw-equivariant; report it explicitly.
5. Check translation/yaw consistency using the same sampled cloud points: jointly transform cloud, measured tools, history and targets, rotating about the calibrated table normal. Rebuild the query frame, predict with the fixed checkpoint and fixed training memory, then undo the test transform. Report max/mean prediction discrepancy in millimetres for non-fallback queries. Keep this diagnostic separate from held-out prediction accuracy. Point resampling and changed visibility are not part of this exact-equivariance check.

New snapshots now serialize optional `tool_positions`. This does not repair old snapshots. For any separate static 14D experiment, note that the revision also restricts position augmentation to XYZ columns; old Group A runs perturbed quaternion components too. Those old results are not substitutes for fresh matched baselines.

## Evaluation and report

Please write `RESULT_DOM_RETRIEVAL_DOUGH_FRAME.md` in this coordination repository and push the report, resolved configs and small preview files. Keep large checkpoints/data in their normal remote storage and provide exact paths and hashes.

Report for all four selected checkpoints:

- Full-trajectory mean waypoint error (mm), future-only error excluding the first waypoint, first-waypoint error, final-waypoint error, per-tool errors, and physical XYZ MSE (m²).
- Per-query and per-episode errors, paired local-minus-absolute differences, median and upper-tail errors, selected epoch, validation metric, runtime and hardware/software versions.
- Exact split identity, training-memory count and disjointness checks; heading-fallback counts and calibration source/value.
- Episode61's previously highlighted query (historically test index 118; verify its ID), plus identical representative good/median/bad queries across conditions. Include original-scene preview PNGs and prediction/target arrays or their storage paths. GIFs are sufficient where FFmpeg lacks libx264.
- A short interpretation that distinguishes numerical transformation correctness, held-out accuracy and inference runtime. A single seed/split does not establish a universal architecture advantage.

The latest position-bias correction shows that shifting the whole prediction to match its first waypoint worsens the episode61 outlier: 41.54 to 46.66 mm for the best full-history Cartesian-residual model. Do not motivate this experiment as fixing an established constant starting offset. It tests dependence on global translation/heading; it may or may not improve motion direction, extent or curvature.

The earlier 24.38 mm result is historical context, not the matched control. The ACT-BeT comparison also uses different prediction windows and input information; it is outside this four-run experiment.
