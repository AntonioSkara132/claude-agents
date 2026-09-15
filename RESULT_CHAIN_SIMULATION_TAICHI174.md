# Result: Episode18 chained simulation rerun with taichi 1.7.4 (matches requirements.txt)

Follow-up to RESULT_CHAIN_SIMULATION.md. That earlier run used taichi 1.6.0 as a fallback because
1.7.4's compiled binary needs GLIBC 2.32 and this host (Ubuntu 18.04) only has GLIBC 2.27. Fixed
that properly and reran with the pinned taichi 1.7.4.

## Fix: run under a newer glibc, isolated to the conda env

- Installed `conda-forge::sysroot_linux-64=2.39` into the `py311` conda env (does NOT touch the
  system glibc — purely additive files inside the conda env prefix).
- Invoke Python via that sysroot's own dynamic linker instead of the system one:
  ```
  <sysroot>/lib64/ld-linux-x86-64.so.2 \
    --library-path <sysroot>/lib64:$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib \
    python script.py ...
  ```
- Verified taichi 1.7.4 imports and runs a real kernel on `arch=cuda` correctly before using it for
  the actual workflow.

## Run

Same dataset/ranges/parameters as the previous run, new output dir (did not rerun into the old one):

```
DATASET=/workspace/TaichiDough/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json
RANGES=/workspace/TaichiDough/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json
OUTPUT=/workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_custom_taichi174_20260915T144025Z
```

`[Taichi] version 1.7.4 ... [Taichi] Starting on arch=cuda`. Failed as expected on the placeholder
RGB path: `FileNotFoundError: /nonexistent/episode18_conversion_not_used_for_simulation.json`.

## Verification

- `chains/from_chunk1/chain_manifest.json` ✓, `chains/from_chunk2_replayed/chain_manifest.json` ✓
- `initial_state.npz` / `terminal_state.npz` present for every chunk in both chains ✓
- Particle files again written with frame-index suffixes (`particles_start_000000.npy`,
  `particles_end_0000NN.npy`), same discrepancy from exact `particles_start.npy`/`particles_end.npy`
  noted in the previous report.

Archive:
```
Path:   /workspace/TaichiDough/experiments/differentiable_mpm/runs/episode18_chained_custom_taichi174_20260915T144025Z_simulation_states.tar.gz
Size:   31M
SHA256: fdb844f706b698b726cfa0036547a97eb9f9aa5f81ec62acd7a442fc57ba9b8a
```

## Important: taichi 1.6.0 vs 1.7.4 gave numerically different results

Compared `terminal_state.npz` for chain A / chunk01 between the two runs (same dataset, same
parameters, only the taichi version differs):

| field | max abs diff |
|-------|-------------|
| x (position)          | 0.4173 |
| v (velocity)           | 11.2846 |
| C (affine velocity)    | 339.2028 |
| F (deformation grad.)  | 1.7710 |
| Jp (plastic det.)      | 0.0 (identical) |

These are not rounding-noise magnitudes — this looks like a real behavioral difference in the MPM
solver between taichi versions (kernel scheduling, atomics ordering, or numerics changed). This
matters for reproducibility: whichever taichi version produced the article's reference/accepted
calibration results should be the one used going forward. Flagging before this becomes a silent
source of divergence between machines running different taichi versions.
