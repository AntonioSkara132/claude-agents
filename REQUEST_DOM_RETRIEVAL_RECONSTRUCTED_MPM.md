# Request: test the learned trajectory policy in reconstructed MPM

## Goal

Run a paired open-loop simulator experiment with the existing reconstructed scene and identified material parameters: recorded tool actions versus learned tool actions from exactly the same simulator state. Do not retrain the policy or re-identify the material for each controller. First establish that recorded-action replay works, then measure what changes under model-predicted translations.

This request authorizes the small policy-action adapter and forward evaluation needed for this experiment, not a long new optimization campaign. Preserve existing checkpoints, calibration artifacts and outputs. Record both repository revisions and effective configs.

## Policy and experiment status

Use the best reported full-test policy: **Group B full-history Cartesian residual, 24.38 mm**, from `RESULT_DOM_RETRIEVAL_FULL_1000EPOCH_CAMPAIGN.md`. This is a GRU-history encoder with Cartesian Top-K mixture plus residual, not the transformer variant. Its training recipe used 1024 points and K=100.

Resolve its actual checkpoint location from:

```python
import json
from pathlib import Path
manifest = json.loads(Path('/workspace/runs/open_loop_groupB_full_cartesian_residual/manifest.json').read_text())
checkpoint = Path(manifest['checkpoint'])
print(checkpoint)
assert checkpoint.is_file()
assert (checkpoint.parent.parent / 'dataset.pt').is_file()
```

If that manifest is unavailable, locate the checkpoint and verify its saved method/config and evaluation metrics. Do not guess a run directory or substitute the local two-epoch smoke checkpoint as the best model.

`load_experiment` requires the checkpoint and its original dataset snapshot. The known `packed_demo` omission affects static `tool_positions`; the history model uses saved history fields instead. Do not bypass identity/hash checks to get a run started. If the selected artifact does not load, report the exact failure before changing it.

**Episode18 is in the policy training partition.** Any episode18 reconstructed-scene run is an integration/calibration diagnostic, not held-out policy performance. The verified seed7 test episodes are 22,25,32,33,36,39,43,53,61. For a later held-out scene, audit material-identification provenance too; policy separation alone does not ensure simulator calibration separation.

## Simulator: primary identified-parameter run

Repository: `https://github.com/AntonioSkara132/TaichiDough.git`. Inspected local revision: `945281c`. On the CUDA host, record the actual revision and preserve any local changes before updating.

Use the saved corrected-physics selection:

```text
experiments/differentiable_mpm/runs/dataset_fit_20260914T124841_6e3f4630/selected_parameters.json
```

Selected material values:

| Parameter | Value |
|---|---:|
| Young's modulus | 17072.497582525557 Pa |
| Poisson ratio | 0.489644253859494 |
| Viscosity | 0.33672455451768313 Pa s |
| plastic_min | 0.7277641506821126 |
| plastic_max | 1.091951442087265 |

Keep the bundle's other settings: corrected-v1 physics, stretch-clamp plasticity, use_jp=false, floor_retention=0.8, tool_friction_coefficient=0.9, tool_retention=1, tool_stickiness=0, grid48, dt=0.0002 s, 24000 particles, density1200 kg/m³, mass0.1365984 kg, padding dx/8. Use its registered geometry/calibration under `data/episode18_registered_tools_v1/` with the saved hashes. Read the JSON rather than manually replacing the settings with rounded table values.

This fit used episode18 frames1–59 and `ignore_recompute_mismatch=true`; disclose this approximate-gradient calibration provenance. There is no independent held-out calibration result established by this artifact.

### Existing baseline command (recorded actions only)

From the TaichiDough repository root, set paths appropriate to the CUDA host:

```bash
export EPISODE18=/absolute/path/to/episode18_kugla
export SIM_PY=/absolute/path/to/python-with-taichi
export RENDER_PY=/absolute/path/to/python-with-pyvista
export RUN=experiments/differentiable_mpm/runs/dom_policy_identified_$(date -u +%Y%m%dT%H%M%SZ)

"$SIM_PY" experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json \
  --episode-id snimanje_23_10-episode18 \
  --parameters experiments/differentiable_mpm/runs/dataset_fit_20260914T124841_6e3f4630/selected_parameters.json \
  --path "snimanje_23_10-episode18.episode=$EPISODE18" \
  --simulation-python "$SIM_PY" --render-python "$RENDER_PY" \
  --backend cuda --end-frame 59 --prepare-only \
  --output-dir "${RUN}_prepare"
```

After preparation succeeds, repeat without `--prepare-only`, using a different fresh output directory such as `${RUN}_recorded`. This is the existing recorded-action baseline, NOT a policy simulator command. Check `forward_video_v2/README.md` and CLI help on the actual revision before running.

The required `dataset_episode18_only.json` is absent from the inspected local tree; check the CUDA host's original fit artifacts. The launcher checks dataset identity and geometry hashes. If it is missing there too, report the missing artifact and recover the original preparation inputs; do not substitute `dataset.json`, alter fingerprints, or fabricate the missing dataset. Likewise, establish the exact segment whose initial state and duration are supported before feeding a predicted path. `--end-frame 59` is the existing calibration baseline, not permission to truncate a policy motion silently.

### Do not mix in the later chain recipe

The later forward-tested chain used E9780 Pa, nu0.49, viscosity10, plastic_min0.89, plastic_max1.09, floor_retention0.9, stickiness0.1 and padding dx/16, with a single scene offset `[0,-0.002,0]` m for both tools. This is a DIFFERENT recipe, not the saved selection above. It has a runnable reference in `REQUEST_EPISODE18_CHAIN_V3_SIMULATE.md` and results in `RESULT_EPISODE18_CHAIN_V3_SIMULATE.md`. Do not describe these settings as the selected identified parameters or combine their geometry/offsets with the primary fit.

Its successful forward output was `/workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_v3_tooloffset2mm_20260915T160026Z`. It was finite but failed strict repeatability tolerance (about0.56 mm position divergence by chunk4). For the new run, repeat recorded controls at least twice from identical state to quantify numerical variation before interpreting small controller differences. Prior CUDA environment was Python3.11.13/Taichi1.7.4 on RTX4090; if needed, reuse the documented py311/GLIBC loader command in the chain request rather than changing the system environment.

## Adapter to implement on CUDA host

There is currently no existing learned-policy replay CLI. Add a separate small runner or action-export consumer rather than overwrite recorded-control code. Relevant simulator interfaces:

- `state.py`: `ToolControl(poses=[2,7], velocities=[2,6], time)`; quaternion order XYZW, unit norm.
- `data.py`: prepared sequence and geometry become `ToolReplay` / `RecordedControls`.
- `replay.py`: recorded controls are sampled at every simulator dt, with endpoint clamping.
- `episode18_chained_comparison.py`: the `stepper.advance(slot, offset_tool_control(item.controls[step]))` call is an existing action-substitution example, and its state-copy logic includes `x,v,C,F,Jp`.
- `reference/deformpath_dynamics.py`: tool transform composition is `scene_from_source @ source_from_marker @ marker_from_tool_frame`; SDF geometry also uses `marker_from_mesh`.

All paths above are under `experiments/differentiable_mpm/`. Inspect the helpers before using them. Recompute linear/angular velocities consistently from the final transformed poses and timestamps; do not replace XYZ while retaining recorded velocities. Reuse quaternion interpolation/control construction, and validate unit-norm quaternions. For fixed start orientation, angular velocity is zero.

Verify the policy's pointcloud-frame XYZ can be mapped into the simulator's calibrated mocap/source frame. Explicitly account for marker versus tool-frame origins; neither the model's XYZ nor the SDF mesh origin is automatically a tool tip. If this transform cannot be established from calibration metadata, stop and report the missing transform rather than assume identity.

For the first test select one complete policy segment matching this reconstruction's timeline. If a recorded prefix is needed, simulate it once and clone full state for all conditions. Do not default to a full multi-segment chain, simulated-cloud feedback, or DMP conversion for this first integration run.

## Policy export requirements

Create a small explicit adapter; the existing open-loop visualizer does not export a complete simulator command sequence.

- Match the exact source recording and segment by checkpoint snapshot IDs, retained indices and timestamps, not by annotation ordinal as an array index.
- Use the historical input ending at segment start only. If this is a training-group query, still pass its source group to exclude same-recording memory entries.
- Use `load_experiment`, `batch_data`, and `physical_output` with evaluation/inference mode. Predict once. Do not feed later recorded observations into that rollout.
- Convert normalized prediction to metres with stored mean/std; residuals use std only. Input clouds and measured history positions are already in physical units.
- Output order is `[T,6]`, UR5e XYZ then Gen3 XYZ. Reshape to `[T,2,3]` for the action adapter.
- Supply the recorded duration externally: `time = start_time_s + phase * duration_s`. Duration is not predicted. Interpolate positions to simulator control times; do not silently stretch the motion to a different duration.
- Convert pointcloud-frame positions and tool geometry with the simulator's verified coordinate transform and marker-to-tool offset, identically for recorded and predicted actions. Never use display-only flips.
- Export raw predicted positions, recorded positions, timestamps, initial poses/history, retrieval IDs, source identity/hashes, checkpoint identity, frame transforms and all replay corrections.

## Paired conditions

1. **Recorded full-pose replay:** existing simulator calibration/reconstruction baseline.
2. **Recorded XYZ + fixed start orientations:** translation-only control baseline.
3. **Predicted XYZ + the same fixed start orientations:** primary translation-only learned-action test.
4. **Hold-position baseline:** initial tool poses held for the same duration.
5. Optional **predicted XYZ + recorded orientations:** label as oracle-orientation diagnostic, not a complete learned controller; compare with condition 1.

The model does not output quaternion trajectories. In particular, quaternion *input* does not imply quaternion *output*.

Save the raw predicted start-position error. Do not teleport tools to the first predicted waypoint. For a physically continuous execution diagnostic, use the explicitly labeled start-anchored path `p_exec(t)=p_raw(t)-p_raw(0)+p_measured(0)` and report it separately from raw waypoint error. This changes the action and must not be hidden as model output. Do not add an unreported approach motion.

Use a shared full simulator state at the comparison start, including particle velocity, deformation and plastic state where present. For a mid-recording start, warm up with the same recorded prefix and snapshot/restore it if supported. Resetting an already deformed cloud to stress-free material is a different initialization and must be labeled. Prefer the first compatible segment for the first smoke run.

## Outputs and interpretation

Save a compact report with effective parameters/configs and:

- start-position, average/final waypoint errors for raw and executed translations;
- object geometry discrepancy versus recorded clouds at matched timestamps, in physical units, using the existing simulator metric plus a clearly defined symmetric distance if available;
- predicted-action versus recorded-action final geometry difference;
- tool/contact diagnostics already supported, simulation finite-state checks and any tool penetration/discontinuity warnings;
- a hold-position baseline, identical simulation budget and render settings;
- side-by-side recorded-action/predicted-action/hold renders with observed object geometry as a labeled reference.

Recorded future object clouds are evaluation targets only. Distinguish simulator mismatch under recorded actions from additional mismatch under learned actions. Neither successful replay nor low point-cloud distance establishes force control, collision safety or task success.

Commit a concise result report and a few preview PNGs to `claude-agents`; keep large checkpoints and videos on the CUDA host and state their exact paths. Report blockers rather than substituting a different scene or parameter set silently.
