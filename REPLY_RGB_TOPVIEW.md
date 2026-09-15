# Reply: Episode18 RGB top-view host and paths

The request is intended for the mounted-data checkout used interactively on host `steffy`, not for an isolated `/workspace/TaichiDough` checkout without the data.

The mounted checkout was observed at:

```text
/home/steffy/Desktop/Data/studenti/antonio_skara/TaichiDough
```

Its resolved storage path may appear as:

```text
/mnt/Data/studenti/antonio_skara/TaichiDough
```

Do not regenerate the dataset from the interpolated `/workspace/data` files. Do not launch calibration or modify Python source.

First test whether your session can access either mounted path:

```bash
for root in \
  /home/steffy/Desktop/Data/studenti/antonio_skara/TaichiDough \
  /mnt/Data/studenti/antonio_skara/TaichiDough
do
  if test -d "$root"; then
    printf 'FOUND_REPO=%s\n' "$root"
  fi
done
```

If neither path is accessible from your session, do not run the workflow. Report that your agent environment cannot access the mounted checkout and leave the request pending for the interactive mounted-machine shell.

If the mount is accessible, use the existing finalized dataset in `episode18_kugla_temporal_v1`:

```bash
cd /home/steffy/Desktop/Data/studenti/antonio_skara/TaichiDough 2>/dev/null || \
cd /mnt/Data/studenti/antonio_skara/TaichiDough

export DATASET="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/dataset_episode18_kugla_temporal_stretch_clamp_v1.json"
export RANGES="$PWD/experiments/differentiable_mpm/data/episode18_kugla_temporal_v1/range_selection.json"
```

The earlier `episode18_kugla_temporal_finalize_test_20260915d` dataset path was incorrect for this mounted checkout. The article optimizer run directory is not required to execute the workflow because the exact parameters are supplied on the command line.

Locate, but do not regenerate or copy, the existing raw bag and conversion metadata:

```bash
export BAG_DIR="$(find /mnt/Data/studenti/antonio_skara /home/steffy/Desktop/Data/studenti/antonio_skara \
  -type d -path '*/DeformPath2/snimanje_23_10/episode18_kugla' \
  -print -quit 2>/dev/null)"

export CONVERSION="$(find /mnt/Data/studenti/antonio_skara /home/steffy/Desktop/Data/studenti/antonio_skara \
  -type f -path '*/DeformPath3/snimanje_23_10/episode18_kugla/conversion_metadata.json' \
  -print -quit 2>/dev/null)"

printf 'DATASET=%s\nRANGES=%s\nBAG_DIR=%s\nCONVERSION=%s\n' \
  "$DATASET" "$RANGES" "$BAG_DIR" "$CONVERSION"

test -f "$DATASET" || { echo 'DATASET missing'; exit 1; }
test -f "$RANGES" || { echo 'RANGES missing'; exit 1; }
test -d "$BAG_DIR" || { echo 'BAG_DIR missing'; exit 1; }
test -f "$CONVERSION" || { echo 'CONVERSION missing'; exit 1; }
```

Only if all four checks pass, follow the run command in `REQUEST_RGB_TOPVIEW.md`, replacing its dataset path with the corrected `episode18_kugla_temporal_v1` path above. Use a fresh output below `experiments/differentiable_mpm/runs/`.

If any check fails, report the exact missing path and stop. Do not substitute the preprocessed `/workspace/data` tensors for the raw RGB bag inputs.
