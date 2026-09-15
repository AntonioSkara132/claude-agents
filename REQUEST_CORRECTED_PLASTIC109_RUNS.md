# Request: corrected Episode18 forwards and chain with plastic_max 1.09

This supersedes the parameter values in `REQUEST_INDEPENDENT_FORWARD_DIAGNOSTIC.md` and `REQUEST_CHAIN_SIMULATION_ONLY.md` wherever they specify `plastic_max=1.00`.

The prior value was a typo. Use:

```text
plastic_min = 0.89
plastic_max = 1.09
```

Preserve every `plastic_max=1.00` run and archive as a diagnostic. Use fresh output directories. Do not launch calibration or edit Python source.

Run from:

```text
/workspace/TaichiDough
```

Use the working Taichi 1.7.4 CUDA invocation established in `RESULT_CHAIN_SIMULATION_TAICHI174.md`, including the conda `py311` sysroot dynamic linker. Verify the process prints Taichi 1.7.4 and `arch=cuda`.

## Shared inputs and parameters

```bash
cd /workspace/TaichiDough

export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export RANGES="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json"
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

test -f "$DATASET" || exit 1
test -f "$RANGES" || exit 1
```

Parameter set for every run:

```text
youngs_modulus = 9780
poisson_ratio = 0.49
viscosity = 10
plastic_min = 0.89
plastic_max = 1.09
floor_retention = 0.9
tool_friction_coefficient = 0.9
tool_stickiness = 0.1
```

## 1. Independent forward diagnostic: C01 and C02

Run `experiments/differentiable_mpm/forward_video_v2/run.py` independently for:

```text
episode18_kugla_chunk01
episode18_kugla_chunk02
```

Use fresh outputs:

```text
experiments/differentiable_mpm/runs/episode18_independent_plastic109_<STAMP>/chunk01
experiments/differentiable_mpm/runs/episode18_independent_plastic109_<STAMP>/chunk02
```

For each invocation pass:

```text
--dataset $DATASET
--backend cuda
--precision f32
--cpu-threads 1
--reference-policy frozen
--youngs-modulus 9780
--poisson-ratio 0.49
--viscosity 10
--plastic-min 0.89
--plastic-max 1.09
--floor-retention 0.9
--tool-friction-coefficient 0.9
--tool-stickiness 0.1
```

Use the appropriate `--episode-id` and output directory for each chunk. Preserve simulation outputs even if optional video rendering fails.

Report every saved frame's `bounds_min` and `bounds_max`. In particular, report C01 frame 0 and frame 18 bounds. The expected physically plausible result should remain near the initial dough region rather than fill `0.04-0.96 m` of the domain.

## 2. Corrected chained run

Run `experiments/differentiable_mpm/episode18_chained_comparison.py` with the same parameters and:

```bash
export CHAIN_OUTPUT="$PWD/experiments/differentiable_mpm/runs/episode18_chained_plastic109_taichi174_${STAMP}"
```

Use placeholder RGB arguments because this computer has no original bag:

```text
--bag-dir /nonexistent/episode18_rgb_bag_not_used_for_simulation
--conversion-metadata /nonexistent/episode18_conversion_not_used_for_simulation.json
```

The expected RGB-stage `FileNotFoundError` occurs after both numerical chains finish. Verify directly that these exist:

```text
chains/from_chunk1/chain_manifest.json
chains/from_chunk2_replayed/chain_manifest.json
```

Verify all seven initial and seven terminal `.npz` files contain finite `x`, `v`, `C`, `F`, and `Jp`, and verify all seven start and seven end particle snapshots. Report the bounds of every snapshot before packaging.

## 3. Package results separately

Create two archives without modifying the run directories:

```text
episode18_independent_plastic109_<STAMP>_states.tar.gz
episode18_chained_plastic109_taichi174_<STAMP>_simulation_states.tar.gz
```

Report for each archive:

- absolute path;
- byte size;
- SHA256;
- Python version;
- Taichi version;
- CUDA device;
- frame counts and particle bounds;
- whether any state is nonfinite.

Do not reuse, overwrite, or delete the earlier `plastic_max=1.00` outputs.
