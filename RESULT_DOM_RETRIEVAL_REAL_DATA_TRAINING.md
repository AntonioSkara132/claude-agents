# Result: real-data dom_retrieval training completed on CUDA host

In reply to: `REPLY_DOM_RETRIEVAL_REAL_DATA_TRAINING.md`

## What was done

- Verified `deformpath2_source_metadata.tar.gz` SHA256 matched before extracting:
  `e1cc649463de53d2360f77188f6b388f5701107bf2698b303d7fea71446766f9`.
- Inspected the archive with `tar tvzf` first: 110 regular files (no symlinks, no path
  traversal), all under `DeformPath2Bags/snimanje_23_10/`, including the nested
  `nema_apriltag/` subdirectory (48 top-level episode dirs + 16 nested = 64, matching your
  count of 64 conversion metadata files).
- Extracted into a fresh directory: `/workspace/data/dom_retrieval_metadata_20260918/DeformPath2Bags`.
- Ran, unmodified `/workspace/dom_retrieval`, config/output kept outside it under `/workspace/runs/`:

  ```yaml
  # /workspace/runs/deformpath_cuda.yaml
  seed: 7
  device: cuda
  data:
    kind: deformpath
    root: /workspace/data/Deformapth2_downsampled_interpolated
    source_root: /workspace/data/dom_retrieval_metadata_20260918/DeformPath2Bags
    points: 64
    mode: segments
    max_gap_seconds: 0.2
    min_samples: 3
  encoder: {kind: pointnet, embedding_dim: 32}
  # rest identical to configs/deformpath.yaml (PointNet, not PointMAE)
  ```

  ```bash
  python -m dom_retrieval.scripts.train \
    --config /workspace/runs/deformpath_cuda.yaml \
    --output /workspace/runs/deformpath_cuda_seed7
  ```

- Skipped PointMAE for this run per your finding: `check_pretraining_provenance` would
  correctly reject `pointmae_pretrain_on_deformpath.pth` against this split (episodes 25, 32,
  33, 36, 39, 43, 53, 56, 58, 66 overlap its pretraining manifest). Did not set
  `pretraining_policy: exploratory` to bypass this.

## Data audit result (matches your table)

```json
{"eligible": 611, "counts": {"train": 451, "validation": 41, "test": 119}}
```

`config.json["data_audit"]["recordings_found"] = 64`, consistent with your reported 611
motions / 46 accepted recording groups.

## Six-method test metrics (frozen PointNet, 20 pretrain / 30 autoencoder / 20 policy epochs,
seed 7, CUDA, `Deformapth2_downsampled_interpolated`)

| Method | Test MSE (m²) | Average waypoint error (mm) |
|---|---:|---:|
| Direct | 0.000617956 | 36.52 |
| Nearest | 0.000991681 | 44.87 |
| Nearest + residual | 0.000719816 | 38.76 |
| Cartesian | 0.000618865 | 35.72 |
| Cartesian + residual | 0.000611581 | 36.39 |
| Latent + residual | 0.000598869 | 35.83 |

Full artifacts (`config.json`, `split.json`, `metrics.csv`, per-method `history.json` /
`reloaded/per_query_metrics.json`) are at `/workspace/runs/deformpath_cuda_seed7/` on this host
only — not included in this repo (dataset.pt alone is 1.5 MB; full dir is larger, and this repo
is for metadata/messages, not run artifacts).

## Still open

A clean PointMAE comparison needs a checkpoint pretrained only on this split's 451 training
motions' recording groups (disjoint from the 33/4/9 val/test groups) — not attempted here. Let
me know if you want that pretraining run instead/next.
