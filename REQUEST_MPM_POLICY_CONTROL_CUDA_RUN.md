# Request: run policy-control MPM simulation on CUDA

The TaichiDough runner now accepts exported policy-control archives as an alternate tool-control source.

## Runner changes

Repository: `TaichiDough`

Changed files:

- `experiments/differentiable_mpm/policy_adapter.py`
- `experiments/differentiable_mpm/forward_video_v2/run.py`
- `experiments/differentiable_mpm/forward_video_v2/forward.py`
- `experiments/differentiable_mpm/forward_video_v2/render.py`
- `experiments/differentiable_mpm/forward_video_v2/README.md`
- `experiments/differentiable_mpm/tests/test_policy_controls_archive.py`

The runner validates:

- archive schema `taichidough/mpm-policy-controls/v1`;
- selected condition and archive path containment;
- `poses [N,2,7]` and `velocities [N,2,6]`;
- finite values and unit quaternions;
- regular `control_times_s` and `control_dt_s`;
- enough controls and duration for the requested simulation.

The recorded replay remains unchanged. The selected condition is used only as the active tool-control sequence, and its provenance is written to the launcher, simulation, result, and render manifests.

## CUDA command

Use the existing dataset, episode path override, and identified physics parameters. Run one policy condition first:

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough

SIM_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
RENDER_PYTHON="/absolute/path/to/python-with-pyvista"

python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json \
  --episode-id snimanje_23_10-episode18 \
  --youngs-modulus 17072.5 \
  --poisson-ratio 0.4896 \
  --viscosity 0.3367 \
  --plastic-min 0.7278 \
  --plastic-max 1.0920 \
  --controls-archive /workspace/runs/mpm_policy_test/conditions_episode18_diagnostic \
  --condition predicted_xyz_recorded_orientation \
  --path snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --simulation-python "$SIM_PYTHON" \
  --render-python "$RENDER_PYTHON" \
  --backend cuda \
  --precision f32 \
  --end-frame 59
```

If the archive is available under a different path on the CUDA host, replace only `--controls-archive`.

Run the recorded baseline with the same command but omit both `--controls-archive` and `--condition`. Use identical dataset, path override, parameters, backend, precision, and endpoint.

## Checks before the full run

First validate the launcher without initializing Taichi:

```bash
python3 experiments/differentiable_mpm/forward_video_v2/run.py \
  [the same arguments] \
  --prepare-only
```

Then run the policy condition and record the printed output directory. Compare:

- `launcher_manifest.json`: control source and archive hashes;
- `simulation/simulation_result.json`: status, completed steps, active control provenance;
- `simulation/run_manifest.json`: active control provenance;
- `perspective/render_manifest.json`: rendered control source;
- `perspective/requested_material_perspective.mp4`.

Run the recorded baseline separately with the same physics parameters. Do not overwrite either output directory.

## Interpretation limits

The prepared archive at `/workspace/runs/mpm_policy_test/conditions_episode18_diagnostic/` uses the empirical diagnostic `camera_depth_optical_frame -> mocap` fit. It aligned tool trajectories for plumbing, but point-cloud alignment was poor. Therefore these CUDA runs test archive loading, stepping, rendering, and output provenance only. Do not treat a completed run or lower loss as validation of the physical camera calibration or policy quality.

The exact physical depth-optical-to-mocap static transform is still required before paired policy-vs-recorded results can be interpreted scientifically.
