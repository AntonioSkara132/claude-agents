# Request: launch all-13-chunk top-view simulation stage

The user confirms this agent should work from:

```text
root@steffy:/workspace/TaichiDough
```

This computer has the TaichiDough source and Episode18 simulation data but not the original ROS bag. Run only the `simulate` stage of `episode18_temporal_rgb_topview.py`. The local computer will later extract RGB frames and assemble the PDF.

Do not launch calibration, modify `.py` source, regenerate the temporal dataset, or stop another process.

## Locate inputs

```bash
cd /workspace/TaichiDough

export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export RANGES="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json"
export CONVERSION="$(find /workspace -type f -path '*/episode18_kugla/conversion_metadata.json' -print -quit 2>/dev/null)"
export EMPTY_BAG_DIR="/workspace/episode18_empty_bag_placeholder"
export OUTPUT="$PWD/experiments/differentiable_mpm/runs/episode18_rgb_topview_article_best_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$EMPTY_BAG_DIR"

test -f "$DATASET" || { echo "DATASET missing: $DATASET"; exit 1; }
test -f "$RANGES" || { echo "RANGES missing: $RANGES"; exit 1; }
test -f "$CONVERSION" || { echo "CONVERSION missing: $CONVERSION"; exit 1; }
test -d "$EMPTY_BAG_DIR" || exit 1
printf 'DATASET=%s\nRANGES=%s\nCONVERSION=%s\nOUTPUT=%s\n' \
  "$DATASET" "$RANGES" "$CONVERSION" "$OUTPUT"
```

The empty bag directory is accepted only because `--stage simulate` does not read RGB data. Do not run `--stage all`, `extract`, or `figure` with this placeholder.

## Run all 13 independent simulations

Use the exact article best-saved parameter values:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python \
  experiments/differentiable_mpm/episode18_temporal_rgb_topview.py \
  --stage simulate \
  --dataset "${DATASET:?DATASET is unset}" \
  --range-selection "${RANGES:?RANGES is unset}" \
  --bag-dir "${EMPTY_BAG_DIR:?EMPTY_BAG_DIR is unset}" \
  --conversion-metadata "${CONVERSION:?CONVERSION is unset}" \
  --output-dir "${OUTPUT:?OUTPUT is unset}" \
  --backend cuda \
  --precision f32 \
  --cpu-threads 1 \
  --reference-policy frozen \
  --youngs-modulus 8803.00073189629 \
  --poisson-ratio 0.4870814294802972 \
  --viscosity 30.370366046279457 \
  --plastic-min 0.862623700149902 \
  --plastic-max 1.0671377211961366 \
  --floor-retention 0.7 \
  --tool-friction-coefficient 0.9 \
  --tool-stickiness 0.0
```

## Verify and package

```bash
python - "$OUTPUT" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
completed = []
for index in range(1, 14):
    chunk = root / "chunks" / f"chunk{index:02d}"
    result_path = chunk / "simulation" / "simulation_result.json"
    if not result_path.is_file():
        raise SystemExit(f"missing {result_path}")
    result = json.loads(result_path.read_text())
    if result.get("status") != "completed":
        raise SystemExit(f"chunk {index:02d} status={result.get('status')}")
    frames = result.get("frames", [])
    for frame in frames:
        snapshot = chunk / "simulation" / frame["particles"]
        if not snapshot.is_file():
            raise SystemExit(f"missing {snapshot}")
    completed.append((index, len(frames)))
print("completed chunks and frame counts:", completed)
PY

ARCHIVE="${OUTPUT}_states.tar.gz"
tar -C "$(dirname "$OUTPUT")" -czf "$ARCHIVE" "$(basename "$OUTPUT")"
sha256sum "$ARCHIVE"
ls -lh "$ARCHIVE"
```

Report the output directory, all 13 frame counts, archive path, archive size, SHA256, Python version, Taichi version, and CUDA device. Preserve every simulation snapshot. The local computer will use the archive to render the simulator panels and combine them with exact RGB frames from the original bag.
