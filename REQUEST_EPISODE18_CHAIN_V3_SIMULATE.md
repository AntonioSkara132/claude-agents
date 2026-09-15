# Request: run corrected Episode18 three-chain simulation with tools lowered 2 mm

Work from:

```text
/workspace/TaichiDough
```

Do not run calibration, read a ROS bag, modify the temporal dataset, overwrite an existing run, delete any prior run/archive, or stop another process.

## Required source

The completed local experimental driver is:

```text
experiments/differentiable_mpm/episode18_chained_comparison.py
```

Expected SHA256:

```text
f8e9e09df8334ed100e170a1ade8f2b7354365a6d952a7f26eb5e151926cafc7
```

Before running, verify the synchronized `/workspace/TaichiDough` copy has that exact hash. If it differs, do not run and report that the corrected v3 source has not reached this checkout.

The script must report schema `taichidough/episode18-chained-comparison/v3` and must contain a `-0.002 m` scene-Y tool offset applied to both tools at every control step. Do not add a second offset elsewhere.

## Numerical experiment

The simulation stage creates:

1. `chain_a`: independent C1 reconstruction, then C1→C2→C3→C4 with C1 numerical context.
2. `independent_from_chunk2`: independent C2 reconstruction, then C2→C3→C4 with C2 numerical context.
3. `from_chain_a_chunk1_end`: Chain A’s complete C1 terminal `x,v,C,F,Jp` state, then replay C2→C3→C4 with C1 numerical context.

Every chain uses:

```text
youngs_modulus = 9780
poisson_ratio = 0.49
viscosity = 10
plastic_min = 0.89
plastic_max = 1.09
floor_retention = 0.9
tool_friction_coefficient = 0.9
tool_stickiness = 0.1
tool_retention = 1.0
tool translation offset = [0.0, -0.002, 0.0] m
```

The script validates constant total mass `0.1365984 kg`, 24,000 particles, grid 48, stretch-clamp plasticity, `use_jp=false`, `jp_hardening=0`, adhesive Coulomb SDF contact, zero contact absorption, and `tool_contact_padding=dx/16`.

## Simulation-only command

Use the established conda `py311` environment and its GLIBC 2.39 sysroot loader:

```bash
cd /workspace/TaichiDough

export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export RANGES="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json"
export SCRIPT="$PWD/experiments/differentiable_mpm/episode18_chained_comparison.py"
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export OUTPUT="$PWD/experiments/differentiable_mpm/runs/episode18_chained_v3_toolminus2mm_taichi174_${STAMP}"
export CONDA_PREFIX="$(conda run -n py311 python -c 'import sys; print(sys.prefix)')"
export LOADER="$(find "$CONDA_PREFIX" -type f -name ld-linux-x86-64.so.2 -path '*sysroot*' -print -quit)"

[ -f "$DATASET" ] || { printf 'missing DATASET: %s\n' "$DATASET"; exit 1; }
[ -f "$RANGES" ] || { printf 'missing RANGES: %s\n' "$RANGES"; exit 1; }
[ -f "$SCRIPT" ] || { printf 'missing SCRIPT: %s\n' "$SCRIPT"; exit 1; }
[ -n "$LOADER" ] && [ -f "$LOADER" ] || { printf 'missing conda sysroot loader\n'; exit 1; }
[ ! -e "$OUTPUT" ] || { printf 'OUTPUT already exists: %s\n' "$OUTPUT"; exit 1; }
printf 'f8e9e09df8334ed100e170a1ade8f2b7354365a6d952a7f26eb5e151926cafc7  %s\n' "$SCRIPT" | sha256sum -c -

export LD_LIBRARY_PATH="$(dirname "$LOADER"):$CONDA_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/targets/x86_64-linux/lib"

"$LOADER" --library-path "$LD_LIBRARY_PATH" "$CONDA_PREFIX/bin/python" -c \
  'import sys, taichi as ti; print(sys.version); print("Taichi", ti.__version__)'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
"$LOADER" --library-path "$LD_LIBRARY_PATH" "$CONDA_PREFIX/bin/python" \
  "$SCRIPT" \
  --stage simulate \
  --dataset "$DATASET" \
  --range-selection "$RANGES" \
  --output-dir "$OUTPUT" \
  --backend cuda \
  --precision f32 \
  --cpu-threads 1 \
  --reference-policy frozen \
  --youngs-modulus 9780 \
  --poisson-ratio 0.49 \
  --viscosity 10 \
  --plastic-min 0.89 \
  --plastic-max 1.09 \
  --floor-retention 0.9 \
  --tool-friction-coefficient 0.9 \
  --tool-stickiness 0.1
```

The process must print Taichi 1.7.4 and `arch=cuda`. `--stage simulate` must finish without bag or conversion-metadata arguments.

## Verify states and replay comparison

```bash
PYTHONDONTWRITEBYTECODE=1 "$LOADER" --library-path "$LD_LIBRARY_PATH" \
  "$CONDA_PREFIX/bin/python" - "$OUTPUT" <<'PY'
from pathlib import Path
import json
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "run_manifest.json").read_text())
assert manifest["schema"] == "taichidough/episode18-chained-comparison/v3/run"
assert manifest["stages"]["simulate"]["status"] == "completed"
assert manifest["calibration_run"] is False
assert manifest["tool_pose_offset_m"] == {
    "axis": "scene_y", "value": -0.002, "applied_to_both_tools": True,
}
expected = {
    "chain_a": [1, 2, 3, 4],
    "independent_from_chunk2": [2, 3, 4],
    "from_chain_a_chunk1_end": [2, 3, 4],
}
fields = {"x", "v", "C", "F", "Jp"}
for name, chunks in expected.items():
    records = manifest["chains"][name]
    assert [row["chunk"] for row in records] == chunks
    chain_manifest = json.loads((root / "chains" / name / "chain_manifest.json").read_text())
    assert chain_manifest["tool_pose_offset_m"]["value"] == -0.002
    for row in records:
        for key in ("initial_state", "terminal_state"):
            path = root / row[key]["path"]
            with np.load(path, allow_pickle=False) as state:
                assert set(state.files) == fields
                assert all(np.isfinite(state[field]).all() for field in fields)
        for frame in row["frames"]:
            points = np.load(root / frame["snapshot"], allow_pickle=False)
            assert points.shape == (24000, 3)
            assert np.isfinite(points).all()
            print(name, row["chunk"], frame["role"],
                  "min", points.min(axis=0).tolist(),
                  "max", points.max(axis=0).tolist())
comparison = json.loads((root / "replay_comparison.json").read_text())
print("replay all_exact:", comparison["all_exact"])
print("replay all_within_tolerance:", comparison["all_within_tolerance"])
for row in comparison["comparisons"]:
    for field, values in row["fields"].items():
        print("replay", row["chunk"], row["role"], field,
              "exact", values["exact_equal"],
              "within_tolerance", values["within_tolerance"],
              "max_abs", values["max_abs_difference"],
              "rmse", values["rmse"])
assert comparison["all_within_tolerance"]
PY
```

CUDA atomic P2G may prevent bitwise equality, so report both `all_exact` and every field’s numerical differences. The tolerance check must pass.

## Package without changing the run

```bash
export ARCHIVE="${OUTPUT}_simulation_states.tar.gz"
tar -C "$(dirname "$OUTPUT")" -czf "$ARCHIVE" "$(basename "$OUTPUT")"
sha256sum "$ARCHIVE"
stat -c 'archive=%n bytes=%s' "$ARCHIVE"
```

Report:

- output directory;
- archive path, bytes, and SHA256;
- Python, Taichi, CUDA device, and `arch=cuda` confirmation;
- all snapshot bounds;
- whether any array is nonfinite;
- replay exact/tolerance results and per-field differences;
- source-identity unchanged status;
- confirmation that prior archives and runs remain unchanged.
