# Orientation delta output implemented

Implemented in local `dom_retrieval` commit `0ad5d5b` (`Add orientation delta pose heads`). The GitHub push could not complete because the local GitHub token is invalid; the commit is ready to push.

## What changed

- Added XYZW quaternion multiplication and stable rotation-vector exponential mapping in `dom_retrieval/poses.py`.
- Added current measured pose extraction from static 14D tool state or the last valid 14D temporal history row.
- Added world/extrinsic composition `q_delta * q_current`.
- Added opt-in `pose_parameterization: orientation_delta`.
- Direct, Cartesian/nearest/latent retrieval, and transformer policy outputs use an internal 12D per-waypoint correction:
  - tool 0 XYZ + rotation vector
  - tool 1 XYZ + rotation vector
- Public predictions remain absolute `[B,T,14]` poses, so existing position/orientation metrics remain usable.
- Delta output layers are zero-initialized; zero rotation correction therefore holds the measured orientation.
- Raw 12D residuals are kept separately for residual regularization and coordinate conversion.
- Existing XYZ and absolute-pose modes default to unchanged `pose_parameterization: absolute`.
- Added `configs/deformpath_pose_orientation_delta.yaml`.
- Added pose and model tests, including identity/nonzero composition, static/temporal current-state extraction, direct and Cartesian residual output, and checkpoint reload smoke coverage.

## CUDA run

Use the new config after pulling commit `0ad5d5b`:

```text
dom_retrieval/configs/deformpath_pose_orientation_delta.yaml
```

Run the same matched dataset/checkpoint campaign as the previous pose experiments. Do not claim improvement until the full run reports separate position and orientation metrics. The local focused tests pass (`24 tests`). The complete suite still has the two pre-existing Matplotlib `projection="3d"` environment failures in `test_metrics` and `test_open_loop`.
