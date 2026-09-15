# Request: launch Episode18 chained simulation states now

The user wants the chained simulation data generated on the mounted RTX 4090 computer now. The recorded RGB rows will be extracted and added later on the local computer, which has the original ROS bag.

Do not launch calibration, edit Python source, regenerate data, or stop an active calibration process.

This request is for the interactive mounted checkout, not an isolated `/workspace/TaichiDough` checkout without the Episode18 data. If your session still cannot access `/mnt/Data/studenti/antonio_skara`, report that and stop.

## Inputs

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough

export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export RANGES="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json"
export OUTPUT="$PWD/experiments/differentiable_mpm/runs/episode18_chained_custom_$(date -u +%Y%m%dT%H%M%SZ)"

test -f "$DATASET" || { echo "DATASET missing: $DATASET"; exit 1; }
test -f "$RANGES" || { echo "RANGES missing: $RANGES"; exit 1; }
printf 'OUTPUT=%s\n' "$OUTPUT"
```

## Run the two numerical chains

The current script performs both complete numerical chains before attempting RGB extraction. Pass intentionally nonexistent RGB inputs. The expected final `FileNotFoundError` occurs only after both chain manifests and all state files have been written.

Use the user's latest explicit parameter set:

```bash
set +e
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python \
  experiments/differentiable_mpm/episode18_chained_comparison.py \
  --dataset "${DATASET:?DATASET is unset}" \
  --range-selection "${RANGES:?RANGES is unset}" \
  --bag-dir /nonexistent/episode18_rgb_bag_not_used_for_simulation \
  --conversion-metadata /nonexistent/episode18_conversion_not_used_for_simulation.json \
  --output-dir "${OUTPUT:?OUTPUT is unset}" \
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
  --tool-stickiness 0.1
RUN_STATUS=$?
set -e
printf 'Expected RGB-stage exit status: %s\n' "$RUN_STATUS"
```

Do not treat the nonzero exit as proof that the numerical chains failed. Verify the required numerical outputs directly.

## Verify complete chain data

```bash
test -f "$OUTPUT/chains/from_chunk1/chain_manifest.json" || { echo 'Chain A manifest missing'; exit 1; }
test -f "$OUTPUT/chains/from_chunk2_replayed/chain_manifest.json" || { echo 'Chain B manifest missing'; exit 1; }

for chunk in 01 02 03 04; do
  test -f "$OUTPUT/chains/from_chunk1/chunk${chunk}/simulation/initial_state.npz" || exit 1
  test -f "$OUTPUT/chains/from_chunk1/chunk${chunk}/simulation/terminal_state.npz" || exit 1
  test -f "$OUTPUT/chains/from_chunk1/chunk${chunk}/simulation/particles_start.npy" || exit 1
  test -f "$OUTPUT/chains/from_chunk1/chunk${chunk}/simulation/particles_end.npy" || exit 1
done

for chunk in 02 03 04; do
  test -f "$OUTPUT/chains/from_chunk2_replayed/chunk${chunk}/simulation/initial_state.npz" || exit 1
  test -f "$OUTPUT/chains/from_chunk2_replayed/chunk${chunk}/simulation/terminal_state.npz" || exit 1
  test -f "$OUTPUT/chains/from_chunk2_replayed/chunk${chunk}/simulation/particles_start.npy" || exit 1
  test -f "$OUTPUT/chains/from_chunk2_replayed/chunk${chunk}/simulation/particles_end.npy" || exit 1
done

echo 'Both chained simulations are complete.'
find "$OUTPUT/chains" -type f | sort
```

## Package the complete partial run

```bash
ARCHIVE="${OUTPUT}_simulation_states.tar.gz"
tar -C "$(dirname "$OUTPUT")" -czf "$ARCHIVE" "$(basename "$OUTPUT")"
sha256sum "$ARCHIVE"
ls -lh "$ARCHIVE"
```

Report:

- final `OUTPUT` path;
- whether both chain manifests passed verification;
- archive path, size, and SHA256;
- any CUDA error exactly as printed.

Preserve all `x`, `v`, `C`, `F`, and `Jp` initial/terminal state files and both start/end particle snapshots for every simulated chunk. Do not rerun into the same output directory.
