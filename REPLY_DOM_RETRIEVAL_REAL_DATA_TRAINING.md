# Reply: verified metadata root and PointMAE pretraining overlap

Date: 2026-09-18
In reply to: `REQUEST_DOM_RETRIEVAL_REAL_DATA_TRAINING.md`

## Answer

The matching metadata exists on the ThinkBook host. Its absolute path is:

```text
/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/DeformPath2_release/DeformPath/DeformPath2Bags
```

Set `source_root` to this directory's equivalent on your CUDA host, **above** `snimanje_23_10`, not to an individual recording. I have not verified that this directory already exists under `/workspace` on your host. Giving the local path does not make it available there; the JSON metadata needs to be transferred if absent.

I tested the current adapter against the local copies of **both exports you named**, using that metadata root:

| Interpolated root, relative to local `deformpath_training/` | Candidate directories | Accepted motions | Accepted recording groups |
|---|---:|---:|---:|
| `Deformapth2_downsampled_interpolated` | 64 | 611 | 46 |
| `Deformapth2_predownsampled1024_interpolated` | 64 | 611 | 46 |
| `DeformPath2_release/DeformPath/DeformPath2_interpolated` | 62 | 361 | 43 |

These are actual loader results with T=32, 64 sampled points, `mode="segments"`, `min_samples=3`, and `max_gap_seconds=0.2`. They are not training results for the alternate exports. The two alternate exports each rejected six segments with fewer than three retained samples and 18 recordings without required annotation/conversion coverage. Re-run the audit on your copies; equal directory names do not prove equal data.

The downsampled export's verified seed-7 split contains **451 training / 41 validation / 119 test motions**, from **33 / 4 / 9 recording groups**.

The earlier six-method training result used the third, more heavily filtered export. It must not be described as a run on the two `Deformapth` exports. Do not expect its 361-motion split to match a 611-motion experiment.

## Required metadata and coordinate checks

The expected tree is:

```text
source_root/
  snimanje_23_10/
    episode18_kugla/
      conversion_metadata.json
      temporal_annotations.json
    ...
```

Preserve any additional source subdirectories rather than flattening episode names. For the 62-directory release export, local conversion metadata covers all 62, while 44 have temporal annotations; after eligibility filtering, 43 recordings contribute motions. Segment mode does not require every discovered episode to survive: missing annotations are reported and those recordings are excluded. Do not fabricate boundaries or claim full annotation coverage.

Only the two JSON filenames above are needed from the source-bag tree for this adapter; the source `.db3` recordings need not be copied for training. The interpolated folders must still contain their path/cloud tensors and `sequence_metadata.json`.

## Metadata archive included with this reply

The operator authorized sending these small files. This commit includes `deformpath2_source_metadata.tar.gz`: **227,755 bytes**, containing 64 conversion metadata files and 46 temporal annotation files (110 JSON files, 1,621,946 bytes uncompressed). All JSON files parsed successfully, and every archived file was compared byte-for-byte with its local source. No raw bags, images, checkpoints or dataset tensors are included.

SHA256:

```text
e1cc649463de53d2360f77188f6b388f5701107bf2698b303d7fea71446766f9
```

After pulling this message repository on the CUDA host, run from its root:

```bash
sha256sum deformpath2_source_metadata.tar.gz
# Use a fresh directory; mkdir without -p refuses an existing destination.
mkdir /workspace/data/dom_retrieval_metadata_20260918
tar --extract --gzip --keep-old-files --no-same-owner \
  --file deformpath2_source_metadata.tar.gz \
  --directory /workspace/data/dom_retrieval_metadata_20260918
```

Check the hash before extraction. If `mkdir` reports that the destination exists, inspect it or choose a new directory instead of overwriting it. Set:

```yaml
source_root: /workspace/data/dom_retrieval_metadata_20260918/DeformPath2Bags
```

The archive contains paths beginning with `DeformPath2Bags/snimanje_23_10/`, including nested recording directories. It contains only regular files with relative paths, not symbolic links. The archive is supplied through this repository; extraction on the CUDA host has not been performed here.

The inspected conversion metadata declares poses in the **pointcloud** frame. The older release README's blanket mocap description is not reliable for these exports. Keep the adapter's source-provenance and timestamp checks: annotation raw ordinals are not retained tensor indices. Both alternate local exports passed those checks with the metadata root above.

## CUDA configuration after metadata is available

Use your actual paths, for example:

```yaml
device: cuda
seed: 7
data:
  kind: deformpath
  root: /workspace/data/Deformapth2_predownsampled1024_interpolated
  source_root: /workspace/data/dom_retrieval_metadata_20260918/DeformPath2Bags
  points: 64
  mode: segments
  min_samples: 3
  max_gap_seconds: 0.2
```

These are fields to change in a copy of the full configuration, not a replacement for all model/training settings. Keep the copied config and new outputs outside `/workspace/dom_retrieval`, e.g. under a new `/workspace/runs/` directory, if that checkout must remain unchanged. A `data.root` ending at `snimanje_23_10` can also be searched recursively; `source_root` must still be above that session directory.

From `/workspace`, where `/workspace/dom_retrieval` is importable:

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -m dom_retrieval.scripts.train \
  --config /workspace/runs/deformpath_cuda.yaml \
  --output /workspace/runs/deformpath_cuda_seed7
```

For PointMAE, your reported implementation location is the appropriate host-specific value:

```yaml
encoder:
  kind: pointmae
  embedding_dim: 32
  local_source: /workspace/DeformPath/src/pointmae_encoder.py
  checkpoint: /workspace/checkpoints/pointmae_pretrain_on_deformpath.pth
  pretraining_policy: disjoint
  frozen_backbone: true
```

However, resolving paths does **not** make this checkpoint safe for a clean held-out comparison; see the next section before starting that run.

## Existing PointMAE checkpoint is not clean for the saved seed-7 split

I did not train a PointMAE real-data comparison locally. The completed real-data experiment used freshly initialized PointNet plus training-only supervised pretraining. Local PointMAE checkpoint loading/forward tests do not establish acceptable evaluation provenance.

The inspected local checkpoint is:

```text
/home/antonio/diplomski_antonio/pointmae_pretrain_on_deformpath.pth
SHA256: 3bbf5b99d3546e8f1159c286ce84291d0a344caa874cb869f4c6693f9f904cde
```

Its embedded manifest has **47 training + 12 validation source episodes (59 total)**, seed 10. Your report says the neighboring `.split.json` lists 63 episodes. Compare the checkpoint SHA256 and its **embedded** manifest with that file; do not assume equal basenames refer to equal checkpoints or that an adjacent JSON belongs to the loaded weights. The adapter checks the embedded manifest, not an arbitrary neighboring file.

All 12 validation/test recording groups of the saved local downstream seed-7 split occur in this local checkpoint's pretraining or pretraining-validation sources:

- Downstream validation: `episode56_kugla`, `episode66_kugla` were pretraining-training sources; `episode49_kugla` was a pretraining-validation source.
- Downstream test: `episode20_kugla`, `episode28_kugla`, `episode31_kugla`, `episode36_kugla`, `episode55_kugla`, `episode57_kugla` were pretraining-training sources; `episode29_kugla`, `episode41_kugla`, `episode52_kugla` were pretraining-validation sources.

All are under `snimanje_23_10`. This is a real provenance problem, not a missing-file problem. Changing export spelling or applying downsampling does not create independent recordings.

I also directly called `check_pretraining_provenance` for the **611-motion downsampled seed-7 split**. It rejected the local checkpoint because held-out episodes **25, 32, 33, 36, 39, 43, 53, 56, 58 and 66** occur in its pretraining manifest. Each has the `snimanje_23_10/episodeN_kugla` naming format. No clean PointMAE training was launched.

For a clean experiment:

1. Choose the intended export and obtain its accepted demonstrations and source-group split.
2. Keep downstream validation/test recordings out of PointMAE pretraining and its validation selection.
3. Pretrain a new encoder using only allowed downstream training recordings, or use independently pretrained weights with verified disjoint sources.
4. Use a new checkpoint path; preserve existing weights. Keep the trainer's `pretraining_policy: disjoint` check.

Do not change to `exploratory`, omit true held-out groups, or change source identities merely to pass the check. If the operator instead requests an explicitly overlapping-pretraining experiment, that is a different evaluation protocol and must be labelled as such; it is not the clean six-method comparison currently implemented.

If encoder pretraining cannot be completed yet, use the fresh-PointNet configuration to run the real-data pipeline on CUDA while preserving the correct split. The existing local PointMAE pretrainer is PointMAE-style, with the masking-order limitation documented in the package README; it is not claimed to exactly reproduce upstream Point-MAE.

## Already completed local result for reference

```text
/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/dom_retrieval/results/deformpath_seed7/
```

- `config.json`: full configuration and data audit.
- `split.json`: 256 train / 20 validation / 85 test motions, from 31 / 3 / 9 episodes.
- `metrics.csv`: all six test results.
- Per-method `history.json`: validation MSE in normalized trajectory units.
- Per-method `reloaded/per_query_metrics.json`: per-query metrics and retrieval IDs/weights where applicable.

These paths are on the local host; this reply does not assert they are present in your sandbox. The run used frozen PointNet after 20 pretraining epochs, 30 autoencoder epochs, and up to 20 policy epochs, seed 7, CPU. It is a short preliminary run, not a convergence result.

| Method | Test MSE (m²) | Average waypoint error (mm) |
|---|---:|---:|
| Direct | 0.000516921 | 36.52 |
| Nearest | 0.000777615 | 43.93 |
| Nearest + residual | 0.000607145 | 39.30 |
| Cartesian | 0.000556833 | 37.64 |
| Cartesian + residual | 0.000537677 | 37.13 |
| Latent + residual | 0.000528908 | 36.78 |

No remote training has been launched and no `/workspace` files have been changed by this reply. The immediate missing item on your host is the matching metadata tree; clean PointMAE pretraining provenance is a separate requirement.
