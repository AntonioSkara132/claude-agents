# Request: run Episode18 RGB/top-view workflow

Run the existing all-13-chunk Episode18 RGB/top-view workflow on the mounted computer. Do not launch calibration, edit Python source, or overwrite an existing run.

## 1. Enter the TaichiDough repository

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough
```

## 2. Install and verify the standalone ROS bag reader

Use the same active Python environment that will run the simulation:

```bash
python -m pip install rosbags
python -c 'from rosbags.highlevel import AnyReader; print("rosbags OK")'
```

## 3. Set and validate input paths

```bash
export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_finalize_test_20260915d/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export RANGES="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json"
export BAG_DIR="$PWD/../data/deformpath_training/DeformPath2/snimanje_23_10/episode18_kugla"
export CONVERSION="$PWD/../data/deformpath_training/DeformPath3/snimanje_23_10/episode18_kugla/conversion_metadata.json"
export OUTPUT="$PWD/experiments/differentiable_mpm/runs/episode18_rgb_topview_article_best_$(date -u +%Y%m%dT%H%M%SZ)"

printf 'DATASET=%s\nRANGES=%s\nBAG_DIR=%s\nCONVERSION=%s\nOUTPUT=%s\n' \
  "$DATASET" "$RANGES" "$BAG_DIR" "$CONVERSION" "$OUTPUT"

test -f "$DATASET"    || { echo "DATASET missing"; exit 1; }
test -f "$RANGES"     || { echo "RANGES missing"; exit 1; }
test -d "$BAG_DIR"    || { echo "BAG_DIR missing"; exit 1; }
test -f "$CONVERSION" || { echo "CONVERSION missing"; exit 1; }
```

If the mounted data directory has a different location, locate the same named bag and conversion files and update only `BAG_DIR` and `CONVERSION`.

## 4. Run all 13 independent chunks with the article best-saved parameters

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python \
  experiments/differentiable_mpm/episode18_temporal_rgb_topview.py \
  --stage all \
  --dataset "${DATASET:?DATASET is unset}" \
  --range-selection "${RANGES:?RANGES is unset}" \
  --bag-dir "${BAG_DIR:?BAG_DIR is unset}" \
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

These are the exact accepted update-15/evaluation-17 values from:

```text
experiments/differentiable_mpm/runs/dataset_fit_20260913T195731_c5aeaabd/optimizer_state.json
```

The workflow uses the current temporal configuration: stretch-clamp plasticity, adhesive Coulomb SDF contact, constant mass `0.1365984 kg`, and contact padding `dx/16`. Therefore this is an evaluation of the article best-fit values in the new temporal setup, not an exact reproduction of the original article calibration.

## 5. Verify outputs

```bash
test -f "$OUTPUT/figure/episode18_temporal_rgb_topview.png" || exit 1
test -f "$OUTPUT/figure/episode18_temporal_rgb_topview.pdf" || exit 1
test -f "$OUTPUT/figure/episode18_temporal_rgb_topview.json" || exit 1

find "$OUTPUT/chunks" -name simulation_result.json -type f | sort
ls -lh "$OUTPUT/figure/episode18_temporal_rgb_topview."{png,pdf,json}
```

Report the final `OUTPUT` path, whether all 13 chunk simulations completed, the RGB reader used, the number of exact RGB timestamp matches, and the three figure paths. Preserve all saved simulation states.

Writing generated files under `experiments/differentiable_mpm/runs/` does not change calibration source identity. Do not add, delete, move, or edit `.py` files under `experiments/differentiable_mpm/` while calibration is active. If calibration is also using CUDA, do not stop it; report any GPU-memory error instead of changing or killing its process.
