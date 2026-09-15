# Request: run independent Episode18 forwards for C01 and C02

The chained simulation expands to almost the complete MPM domain during C01. Run ordinary independent forwards for temporal chunks C01 and C02 with the identical parameters. This distinguishes a chained stepping error from instability already present in the ordinary forward path.

Work from:

```text
/workspace/TaichiDough
```

Do not run calibration, edit Python source, overwrite an existing run, or stop another process. Use the same Python/Taichi environment used for the chained run and report its versions. Output stays under `experiments/differentiable_mpm/runs/`.

## Inputs and output

```bash
cd /workspace/TaichiDough

export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export OUTPUT_ROOT="$PWD/experiments/differentiable_mpm/runs/episode18_independent_forward_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"

test -f "$DATASET" || { echo "DATASET missing: $DATASET"; exit 1; }
mkdir -p "$OUTPUT_ROOT"

conda run -n py311 python -c 'import sys, taichi as ti; print(sys.version); print("Taichi", ti.__version__)'
```

## Run C01 and C02 independently

Use the exact parameter set from the invalid chained archive:

```bash
for chunk in 01 02; do
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. conda run -n py311 python \
    experiments/differentiable_mpm/forward_video_v2/run.py \
    --dataset "$DATASET" \
    --episode-id "episode18_kugla_chunk${chunk}" \
    --backend cuda \
    --precision f32 \
    --cpu-threads 1 \
    --reference-policy frozen \
    --youngs-modulus 9780 \
    --poisson-ratio 0.49 \
    --viscosity 10 \
    --plastic-min 0.89 \
    --plastic-max 1.00 \
    --floor-retention 0.9 \
    --tool-friction-coefficient 0.9 \
    --tool-stickiness 0.1 \
    --output-dir "$OUTPUT_ROOT/chunk${chunk}"
done
```

If video rendering fails after simulation, preserve the completed `simulation/` directory and continue with the numerical checks below. Do not rerun into an existing output.

## Report particle extents for every saved frame

```bash
PYTHONDONTWRITEBYTECODE=1 python - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import json
import sys
import numpy as np

root = Path(sys.argv[1])
for chunk in ("01", "02"):
    sim = root / f"chunk{chunk}" / "simulation"
    result_path = sim / "simulation_result.json"
    if not result_path.is_file():
        raise SystemExit(f"missing {result_path}")
    result = json.loads(result_path.read_text())
    print(f"chunk{chunk} status={result.get('status')} frames={len(result.get('frames', []))}")
    for frame in result["frames"]:
        path = sim / frame["particles"]
        points = np.load(path, allow_pickle=False)
        print(
            f"  source_frame={frame['source_frame']} sim_time_s={frame['sim_time_s']} "
            f"min={points.min(axis=0).tolist()} max={points.max(axis=0).tolist()} "
            f"finite={bool(np.isfinite(points).all())}"
        )
PY
```

## Diagnostic interpretation

- If independent C01 also expands from the compact initial state to roughly `0.04-0.96 m`, the instability is in the current configuration/parameters or ordinary stepping path, not in state transfer between chunks.
- If independent C01 stays compact while Chain A C01 expands, the chained runner differs incorrectly from the working forward runner before any chunk boundary.
- Independent C02 is a second check using its own reconstructed initial state.

Package only the two new diagnostic runs and their manifests/snapshots, then report the archive path and SHA256. Do not include or modify unrelated runs.
