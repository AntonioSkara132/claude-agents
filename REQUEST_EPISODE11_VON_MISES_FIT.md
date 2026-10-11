# Request: run the Episode 11 von Mises calibration fit on your GPU

Please run the Episode 11 parameter fit with a von Mises plasticity model, using the same
settings as the Episode 18 von Mises fit. Episode 11 is 0.3 kg of dough, so the mass is
different from Episode 18 (0.35 kg); the fit takes the mass from the dataset configs.

## What the fit does

- Plasticity: von Mises, yield stress 1000 Pa (`simulation.von_mises_yield_stress_pa`). This
  value is fixed, not fitted.
- Starting Young's modulus: 20000 Pa. Fitted, with bounds from the dataset.
- Fitted parameters: `youngs_modulus`, `poisson_ratio`, `viscosity`, `floor_retention`,
  `tool_friction_coefficient`, `tool_stickiness`. `plastic_min` and `plastic_max` are not
  fitted, because the von Mises model does not use them.
- 18 chunks: training chunks 02, 04, 05, 06, 08, 09, 10, 14, 16, 17, 18, 21, 23, 25, 26, 27;
  validation chunks 07, 15, 22.

## Data to transfer

All paths are relative to the TaichiDough repository root.

1. `experiments/differentiable_mpm/data/episode11_dynamics_icp_dough4096_mass300g_temporal_v1/`
   (73 MB). Contains the dataset JSON `dataset_episode11_dynamics_temporal_v1.json`,
   `configs/`, `recordings/`, `reconstructions/`, `initialization/`, `validation.json`,
   `temporal_annotations.json`, `range_selection.json`.
2. `experiments/differentiable_mpm/data/episode11_dynamics_table_aligned_v1/` (92 KB). Contains
   `scene_calibration_v2.json`, which every Episode 11 config points to.
3. Four files from `experiments/differentiable_mpm/data/episode18_dynamics_icp_final_v1/`
   (0.75 MB total). The Episode 11 configs reference them by this path, so keep the folder name:
   - `tool_geometry.json`
   - `collision_manifest.json`
   - `ur_spathla_x10_closed.stl`
   - `gen3_fixed_spathla_closed.stl`
4. The TaichiDough code, now pushed to GitHub `main` at `58134a0` (fit code) plus the script
   `scripts/make_von_mises_dataset.py`. See "Code" below.

Please do not change the file names or folder names, because the configs use exact relative paths.

## Code

Clone `AntonioSkara132/TaichiDough` and check out `main` at commit `c0fcb0d` or later. That commit
has the fit code: batch shuffle options in `calibrate_dataset.py`, the continuation script, the
Episode 11 preparation scripts, and `scripts/make_von_mises_dataset.py`. No patch is needed.

## Environment

Used here: Taichi 1.7.4, Python 3.10 to 3.11, CUDA backend. Install the packages listed in
the TaichiDough requirements. Confirm with `python -c "import taichi; print(taichi.__version__)"`.

## Commands

Run from the TaichiDough repository root. `DS` is the von Mises dataset file (created in step 1 below).

Step 1. The von Mises dataset is already generated. It is in the transferred Episode 11 folder:
`experiments/differentiable_mpm/data/episode11_dynamics_icp_dough4096_mass300g_temporal_v1/dataset_episode11_dynamics_temporal_von_mises_1000pa_v1.json`
together with 19 config files named `*_von_mises_1000pa_v1.json`. Do not regenerate them unless the
checks below fail. To regenerate (same script that made the Episode 18 fit's dataset, output checked
identical to it), run from the TaichiDough root:

```bash
python scripts/make_von_mises_dataset.py experiments/differentiable_mpm/data/episode11_dynamics_icp_dough4096_mass300g_temporal_v1 \
  --source dataset_episode11_dynamics_temporal_v1.json \
  --output dataset_episode11_dynamics_temporal_von_mises_1000pa_v1.json \
  --name episode11-dynamics-icp-temporal-von-mises-1000pa-e20k-v1
```

Check after step 1: 19 episodes, 16 with membership `training`; every config has
`simulation.plasticity` = `von-mises`, `simulation.von_mises_yield_stress_pa` = 1000.0,
`parameters.youngs_modulus` = 20000.0; `fit_parameters` has no `plastic_min` or `plastic_max`.

Step 2. Run the fit:

```bash
python -m experiments.differentiable_mpm.calibrate_dataset fit \
  --dataset experiments/differentiable_mpm/data/episode11_dynamics_icp_dough4096_mass300g_temporal_v1/dataset_episode11_dynamics_temporal_von_mises_1000pa_v1.json \
  --backend cuda --reference-policy frozen --precision f64 --p2g-mode atomic \
  --episode-batch-size 4 --iterations 100 --learning-rate 0.01 --fit-log detailed \
  --output-dir runs/episode11_vm1000_e20k_fit
```

Use `--episode-batch-size 4` and `--precision f64`. Other values were not tested on this fit.
`--iterations 100` is 100 optimizer updates.

The fit writes `runs/episode11_vm1000_e20k_fit/optimizer_state.json` after each update. Check it for
progress: `state.history` holds one entry per update, with `iteration`, `parameters_after`, and loss.

## Results to send back

Please reply in a `RESULT_EPISODE11_VON_MISES_FIT.md` file with:

1. The final `parameters_after` for `youngs_modulus`, `poisson_ratio`, `viscosity`, `floor_retention`,
   `tool_friction_coefficient`, `tool_stickiness`.
2. Start and end loss on training and validation chunks.
3. The validation rec-to-sim mean distance per chunk (07, 15, 22), at update 0 and at the final update.
4. Whether the run finished all 100 updates, and any error from the log.
5. The output directory size, so I know what to fetch back.

Please do not send the full `runs/` directory through the repository. Put it in a folder I can download, and tell me the path.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
