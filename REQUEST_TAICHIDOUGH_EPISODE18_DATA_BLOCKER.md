# Blocker: episode18 data problems in REQUEST_DOM_RETRIEVAL_RECONSTRUCTED_MPM.md

Working through the reconstructed-MPM policy-replay request on this host. Three real problems
found in sequence, first two resolved locally, third is a genuine blocker needing your input.

## 1. Resolved: named calibration run doesn't exist here

`experiments/differentiable_mpm/runs/dataset_fit_20260914T124841_6e3f4630/selected_parameters.json`
does not exist on this host. Only a different, later run exists
(`dataset_fit_20260917T000310_24f0d45c`), with substantially different fitted material values
(E=78906 Pa vs the requested 17072 Pa, Poisson 0.224 vs 0.490, viscosity 0 vs 0.337) — clearly an
independent fit, not the same run under a different name.

**Operator vouched for the requested parameter values directly** (E=17072.497582525557,
ν=0.489644253859494, viscosity=0.33672455451768313, plastic_min=0.7277641506821126,
plastic_max=1.091951442087265). Used them via `forward_video_v2/run.py`'s explicit-CLI parameter
source (`--youngs-modulus`/`--poisson-ratio`/etc.), which is a fully supported alternative to
`--parameters <file>` and bypasses the dataset-fingerprint check a hand-written
`selected_parameters.json` would fail. Confirmed via `forward_config.json`'s resolved
`parameters` dict — exact match, plus `tool_retention: 1.0` came through correctly as the
implicit default.

## 2. Resolved: primary episode18 data copy is corrupted, GLIBC toolchain needed rebuilding

- `/workspace/data/DeformPath3/snimanje_23_10/episode18_kugla/pointclouds.pt` and
  `pointclouds_interpolated.pt` both fail to load (`PytorchStreamReader failed reading zip
  archive: failed finding central directory` — truncated file, not a code issue). Switched to
  `/workspace/data/preprocessed_dataset/episode18_kugla`, which is intact and passed every
  file-hash check during `--prepare-only` (calibration, initial_particles, tool_geometry,
  collision meshes all matched `episode18.json`'s `expected_sha256`).
- The documented GLIBC 2.39 sysroot loader command (from `REQUEST_EPISODE18_CHAIN_V3_SIMULATE.md`)
  does not work on this host — the loader itself fails (`GLIBC_2.35 not found`). Found a working
  alternative: a previously-built patched Python 3.10 interpreter at
  `/root/.claude/jobs/c912ad55/tmp/py310runtime/python3.10` (Taichi 1.7.4, CUDA init confirmed
  working) plus `LD_LIBRARY_PATH="<sysroot>/lib64:/opt/conda/envs/py310/lib"`. That env var leaks
  into and breaks the separate `render_python` (py311) subprocess `run.py` spawns, and popping it
  from `os.environ` inside the running interpreter (safe — dlopen for an already-imported taichi
  extension doesn't need it again) then breaks the *simulation* subprocess `run.py` also spawns
  for `--simulation-python`, since that's a fresh process needing the var at its own startup.
  Fixed with two small wrapper scripts (neither touches TaichiDough source): one that pops
  `LD_LIBRARY_PATH` before executing `run.py` in-process, and one used as `--simulation-python`
  that re-sets it before exec'ing the patched interpreter for the spawned simulation subprocess.

## 3. Blocker: recorded-sequence fingerprint mismatch

With everything above working, the actual simulation now fails:

```
ValueError: Recorded sequence fingerprint does not match the explicit expected value
```

This is a different check than the file-hash checks that passed — it verifies the *recorded
episode sequence itself* (timestamped tool-path/point-cloud timeline) against
`episode18.json`'s `expected_sequence_fingerprint`
(`b5b2225335813b6c61f1777194c92cfa5de93b082dcf1a4974cd9cf992b825cb`). The supporting reference
files (calibration, particles, meshes) all matched by hash; the sequence data itself doesn't.
This means `/workspace/data/preprocessed_dataset/episode18_kugla`'s sequence is either a
reprocessed/different export than whatever the calibration and `episode18.json` config were
originally validated against, even though it shares the same reference files.

Operator declined to bypass this check or guess at another data copy — asking you to supply or
confirm the correct episode18 sequence data (matching the fingerprint above) rather than have me
keep substituting candidate directories.

## Reproduction

```bash
cd /workspace/TaichiDough
export SYSROOT=/opt/conda/pkgs/sysroot_linux-64-2.39-he3f20f0_6/x86_64-conda-linux-gnu/sysroot
export SIM_PY=/root/.claude/jobs/c912ad55/tmp/py310runtime/python3.10  # host-specific; find your own working interpreter
export EPISODE18=/workspace/data/preprocessed_dataset/episode18_kugla
env LD_LIBRARY_PATH="$SYSROOT/lib64:/opt/conda/envs/py310/lib" PYTHONPATH=. "$SIM_PY" \
  experiments/differentiable_mpm/forward_video_v2/run.py \
  --dataset experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/dataset_episode18_only.json \
  --episode-id snimanje_23_10-episode18 \
  --youngs-modulus 17072.497582525557 --poisson-ratio 0.489644253859494 \
  --viscosity 0.33672455451768313 --plastic-min 0.7277641506821126 --plastic-max 1.091951442087265 \
  --floor-retention 0.8 --tool-friction-coefficient 0.9 --tool-stickiness 0.0 \
  --path "snimanje_23_10-episode18.episode=$EPISODE18" \
  --backend cuda --end-frame 59 --output-dir <fresh-dir>
```

Repo revision: `945281c` (matches the original request). Waiting on the correct episode18 data
before continuing to the policy-adapter/paired-conditions work.
