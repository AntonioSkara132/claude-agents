# MPM policy adapter implemented

Published for CUDA follow-up on 2026-09-19.

## Code commits

- `dom_retrieval` commit `1d6b508` (`master`): adds `dom_retrieval/scripts/export_policy_segment.py`.
- `TaichiDough` commit `832c6cb` (`main`): adds `experiments/differentiable_mpm/policy_adapter.py`, `export_policy_controls.py`, and three adapter tests.

The existing simulator calibration and recorded-control code were not modified.

## Data and checkpoint

Use the exact Episode 18 sequence:

```text
/workspace/data/relocated_episode18
```

Its required combined fingerprint is:

```text
b5b2225335813b6c61f1777194c92cfa5de93b082dcf1a4974cd9cf992b825cb
```

Use the existing Group B full-history Cartesian-residual checkpoint:

```text
/workspace/runs/history_deformpath_full_1000_seed7/seed_7/full/cartesian_residual/checkpoint.pt
```

The dataset snapshot is `checkpoint.parent.parent / "dataset.pt"`. Episode 18 is in the policy training partition, so this is an integration/calibration diagnostic, not held-out accuracy.

## Step 1: export one policy segment

Run from the `dom_retrieval` package parent after pulling `1d6b508`:

```bash
python -m dom_retrieval.scripts.export_policy_segment \
  --checkpoint /workspace/runs/history_deformpath_full_1000_seed7/seed_7/full/cartesian_residual/checkpoint.pt \
  --query-id <exact episode18 segment ID> \
  --output /workspace/runs/policy_export_episode18 \
  --device cuda
```

The exporter loads the checkpoint and its saved dataset snapshot, verifies identities through `load_experiment`, uses only the segment-start cloud/history, predicts once, and writes:

```text
prediction_xyz.npy   [T,2,3], metres, UR5e then Gen3
phase.npy             normalized 0..1 phase
manifest.json         query/source/checkpoint/timestamp provenance
```

Do not select a query by annotation ordinal. Use the exact snapshot ID and check `retained_indices`, `start_time_s`, `end_time_s` in the manifest.

## Step 2: build the five control sequences

Use the exact raw recorded poses as `[N,2,7]` XYZW poses and relative recorded times as `[N]`. These must be in the source frame associated with the verified Episode 18 sequence. Supply the calibrated `scene_from_source` and `marker_from_tool_frames` matrices explicitly; do not use a display transform or silently assume an identity marker offset.

From the TaichiDough repository root after pulling `832c6cb`:

```bash
python -m experiments.differentiable_mpm.export_policy_controls \
  --recorded-poses /path/episode18_raw_poses.npy \
  --recorded-times /path/episode18_relative_times.npy \
  --policy-export /workspace/runs/policy_export_episode18 \
  --control-dt 0.0002 \
  --duration <segment duration in seconds> \
  --scene-from-source /path/scene_from_source.json \
  --marker-from-tool-frames /path/marker_from_tool_frames.json \
  --output /workspace/runs/policy_controls_episode18
```

It writes controls for:

1. `recorded_full_pose`
2. `recorded_xyz_fixed_orientation`
3. `predicted_xyz_fixed_orientation`
4. `hold_position`
5. `predicted_xyz_recorded_orientation`

Each `.npz` contains `poses [N,2,7]` and `velocities [N,2,6]`. The adapter interpolates normalized policy phase to the supplied recorded duration, applies the explicit scene and marker transforms, recomputes linear/angular velocities and validates unit quaternions. It does not teleport or start-anchor the predicted path. Raw and any separately corrected path must remain distinguishable in the report.

## Simulator integration

Use the existing `prepare_experiment(...)` to obtain the verified initial particle state, SDF and simulation configuration. For each condition, clone the shared initial `ParticleState`, load it into `ReferenceStepper`, and call `advance(ToolControl(...))` once per control frame. Preserve `{x,v,C,F,Jp}` and use fresh output directories. The generated control files are intentionally separate from `RecordedControls` so the existing baseline remains unchanged.

Record the checkpoint/dataset identity, sequence fingerprint, simulator revision, calibration hashes, transforms, selected segment, condition, start-position error, trajectory error, geometry metrics, finite-state/contact warnings and output paths. Include recorded and hold baselines. Do not claim physical task success from this integration run.

## Verification performed

- `dom_retrieval`: 83 tests passed and exporter compiled.
- `TaichiDough`: 3 policy-adapter tests passed.
- The adapter has not been run on the CUDA host because the checkpoint and verified Episode 18 data are not present in this local environment.
