# Request: matched MPM policy-control ablation on CUDA

Run a small matched ablation for the Episode 18 policy-control path. The purpose is to determine whether each exported control condition changes the simulator result and to verify that the runner uses the selected sequence. These are **plumbing and numerical sensitivity tests only**: the policy archive uses the empirical diagnostic `camera_depth_optical_frame -> mocap` fit, which failed the point-cloud calibration check. Do not present these runs as physical policy-quality or camera-calibration evidence.

## Scientific question

Under identical initial state, dataset, physics parameters, simulator revision, precision, duration, and source episode, do the active controls produce different numerical MPM outcomes from the recorded replay? Separately, does changing only orientation or only XYZ expose a control-path bug?

## Required matched run matrix

Use the same archive and endpoint for every run:

1. `recorded_full_pose` — recorded replay baseline.
2. `recorded_xyz_fixed_orientation` — archive path/control-loader check; should be close to the recorded baseline.
3. `predicted_xyz_fixed_orientation` — predicted XYZ with fixed orientation.
4. `predicted_xyz_recorded_orientation` — predicted XYZ with recorded orientation; primary policy-control case.
5. `hold_position` — stationary-control negative/control-path check.

The prepared prediction segment covers only through recorded frame 43. Therefore use `--end-frame 43` for **all five** runs. Do not extend the policy runs to frame 59 unless a new archive covering that duration is exported.

The archive was re-exported at simulator timestep `0.0002 s`; do not replace it with the earlier 30 Hz archive. The archive directory is expected at:

```text
/workspace/runs/mpm_policy_test/conditions_episode18_diagnostic/
```

If the path differs on the CUDA host, replace only `--controls-archive`.

## Common configuration

Repository and dataset:

```text
TaichiDough: /mnt/Data/studenti/antonio_skara/TaichiDough
dataset: experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json
episode: snimanje_23_10-episode18
path override:
snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla
```

Use these identified physics parameters for every condition:

```text
youngs_modulus = 17072.5
poisson_ratio  = 0.4896
viscosity      = 0.3367
plastic_min    = 0.7278
plastic_max    = 1.0920
backend        = cuda
precision      = f32
end_frame      = 43
```

Use the existing `sim_python_wrapper.sh` / `sim_launcher.py` route or an equivalent command that removes the sysroot `LD_LIBRARY_PATH` before the launcher starts. A leaked `LD_LIBRARY_PATH` previously broke the rendering dependency-check subprocess. Numeric simulation completion is the required result; a host OpenGL failure should be reported separately and must not be treated as a physics failure.

## Commands

First run `--prepare-only` for one archive-backed condition and confirm that the launcher accepts the archive, condition, timing, dataset fingerprint, calibration, and output directory without initializing Taichi. Use a fresh output directory.

Then run each condition in a distinct, non-existing output directory. The recorded baseline may omit `--controls-archive` and `--condition`, but it must use the same endpoint and all other arguments. For the four archive-backed cases use:

```bash
python3 experiments/differentiable_mpm/forward_video_v2/run.py \\
  --dataset experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json \\
  --episode-id snimanje_23_10-episode18 \\
  --youngs-modulus 17072.5 \\
  --poisson-ratio 0.4896 \\
  --viscosity 0.3367 \\
  --plastic-min 0.7278 \\
  --plastic-max 1.0920 \\
  --controls-archive /workspace/runs/mpm_policy_test/conditions_episode18_diagnostic \\
  --condition CONDITION_NAME \\
  --path snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \\
  --simulation-python "$SIM_PYTHON" \\
  --render-python "$RENDER_PYTHON" \\
  --backend cuda \\
  --precision f32 \\
  --end-frame 43
```

For the recorded baseline, use the same command and omit both archive arguments. Set a distinct output location according to the runner's normal output naming or an explicit fresh run directory. Do not overwrite the previously completed frame-59 baseline.

## Checks to report for every run

Report the exact output directory and these values:

- `simulation/simulation_result.json`: status, completed steps, simulated time, last saved source frame, failure;
- `simulation/run_manifest.json`: active control source, selected condition, archive/manifest hashes, recorded baseline provenance;
- `launcher_manifest.json`: dataset/source fingerprint, calibration identifiers, control source and timing;
- whether the selected condition was actually used for `stepper.advance`;
- numeric final particle-state path/hash if the runner records it;
- wall time and approximate warm-up-adjusted steps/s;
- rendering status separately from simulation status.

Also compare the five runs on the same endpoint using whatever existing numeric summaries are already emitted. Do not invent a new loss metric from a different source. If no common particle or tool metric is emitted, report completion and artifact hashes only and say that outcome differences were not quantified.

## Interpretation rules

- A completed run confirms archive loading, control selection, stepping, and provenance recording. It does not validate the diagnostic frame transform.
- The `recorded_xyz_fixed_orientation` case is a control-path check, not an independent learned policy.
- `hold_position` is expected to differ physically from recorded replay; it is useful for checking that active controls affect the simulation.
- A smaller or larger numeric quantity must not be called better policy performance without a calibrated frame and a predefined task metric.
- If any archive-backed run fails, first report whether the failure is archive timing, duration, launcher environment, rendering, or simulation. Do not bypass fingerprint, calibration, source, or overwrite checks.
- Do not overwrite previous outputs. Use one fresh directory per condition.

## Separate retrieval ablation

The retrieval-method ablation will be run in `dom_retrieval`, not in this CUDA task. Its planned controls are direct/no retrieval, zero reference, cosine retrieval, and random eligible training reference, followed by trajectory-supervised retrieval with `K=25,50,100` and an adaptive similarity rule. This CUDA request is only for the matched simulator conditions above.
