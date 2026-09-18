# Request: run dom_retrieval real-data training, or confirm source_root

I was asked to run training for `dom_retrieval` (a demonstration-retrieval package at
`/workspace/dom_retrieval` in this sandbox), using the pretrained encoder at
`/workspace/checkpoints/pointmae_pretrain_on_deformpath.pth` (with
`pointmae_pretrain_on_deformpath.metrics.json` / `.split.json` alongside it).

On this host, `configs/deformpath.yaml` and `configs/pointmae.yaml` cannot run against real
data: they need a `source_root` directory (the `DeformPath2Bags`-equivalent) containing
per-episode `conversion_metadata.json` (+ `temporal_annotations.json`) for every episode in the
interpolated root, and this sandbox only has that metadata for a handful of episodes.

## What I found here

Interpolated root candidates (targets, correct spelling per your note is "Deformapth", not
"DeformPath"):

```text
/workspace/data/Deformapth2_downsampled_interpolated/snimanje_23_10/   (64 episode dirs)
/workspace/data/Deformapth2_predownsampled1024_interpolated/snimanje_23_10/  (64 episode dirs)
```

`pointmae_pretrain_on_deformpath.split.json` in `/workspace/checkpoints` actually references
`/workspace/data/DeformPath2_interpolated/snimanje_23_10/...` (63 episodes, plain "DeformPath"
spelling) as its pretraining sources. A second, different PointMAE pretrain run exists at
`/home/antonio/diplomski_antonio/DeformPath/outputs/pointmae_deformapth2_pretrain.pth`, whose
`.split.json` references `/workspace/data/Deformapth2_downsampled_interpolated/snimanje_23_10/...`
(the "Deformapth" spelling) as sources instead.

Directories that actually contain `conversion_metadata.json` here:

```text
/workspace/data/DeformPath2_no_downsampling/snimanje_23_10/   (10 episodes)
/workspace/data/09_6/                                          (9 episodes)
/workspace/data/DeformPath3/snimanje_23_10/                    (episode18_kugla only)
/workspace/data/preprocessed_dataset/                          (2 episodes)
```

None of these covers all 63-64 episodes in either interpolated root — at most ~10 overlap. Also,
`configs/pointmae.yaml`'s `local_source` points at
`/home/antonio/diplomski_antonio/DeformPath/src/pointmae_encoder.py`, but only
`/home/antonio/diplomski_antonio/DeformPath/outputs/` exists here — no `src/`. The working
`pointmae_encoder.py` only exists in this sandbox at `/workspace/DeformPath/src/pointmae_encoder.py`.

## What I need from you

Whichever is easier on your side:

1. **Run it yourself**, if your environment has the full `DeformPath2Bags`-equivalent
   `source_root` (with conversion metadata for all episodes) and the real
   `DeformPath/src/pointmae_encoder.py`. Something like:

   ```bash
   cd /path/to/dom_retrieval/..
   python -m dom_retrieval.scripts.train \
     --config dom_retrieval/configs/pointmae.yaml \
     --output dom_retrieval/results/pointmae_real_run
   ```

   (fixing `configs/pointmae.yaml`'s `checkpoint`/`local_source` paths and `data.root`/
   `data.source_root` to your real paths first, using seed 7 as configured). Report back the
   resulting `config.json` (esp. `data_audit`), per-method validation/test metrics, and whether
   `check_pretraining_provenance` accepted the checkpoint cleanly (no overlap error).

2. **Or just confirm the correct `source_root` path** (full conversion-metadata directory
   matching `Deformapth2_downsampled_interpolated/snimanje_23_10` or
   `DeformPath2_interpolated/snimanje_23_10`) if it exists somewhere I haven't checked, so I can
   wire the config up and run it here myself.

Until then, I'm running `configs/synthetic.yaml` here as the guaranteed-working baseline, since it
needs no real data.

Do not modify `/workspace/dom_retrieval` or delete/overwrite anything in `/workspace/checkpoints`.
