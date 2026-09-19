# Visual attachment: episode61_kugla outlier previews for all 16 models

Referenced in `RESULT_DOM_RETRIEVAL_FULL_1000EPOCH_CAMPAIGN.md`. That report only gave file paths
on this host; here are the actual rendered comparisons for the one query that most clearly
separates "fixed" from "still broken" — `episode61_kugla`, the mid-episode/near-stationary
segment (test index 118).

`episode61_outlier_previews.tar.gz`: 16 PNGs (static final-frame preview of each open-loop
render: 3D predicted-vs-recorded trajectory + per-tool error-over-phase chart), one per model,
~3.0 MB total. No GIFs/raw arrays included — those are larger (84 MB across all 16 models) and
stay on this host at `/workspace/runs/open_loop_group{A,B_full}_<method>/` if needed later.

SHA256: `a6237e017651d1293cd9c63ac58a9be23994e08a570e1693772b768a20d11759`

Filenames map directly to the earlier results table, e.g.
`open_loop_groupA_latent_residual_query0118.png` = Group A (proprioception+quaternions+noise),
`latent_residual` method, this query.

Worth looking at side by side:

- **Fixed**: `open_loop_groupA_latent_residual_query0118.png` (16.9mm), `open_loop_groupA_cartesian_residual_query0118.png` (23.1mm), `open_loop_groupB_full_cartesian_query0118.png` (24.9mm)
- **Still broken**: `open_loop_groupA_nearest_query0118.png` (83.8mm), `open_loop_groupA_transformer_residual_query0118.png` (81.0mm), `open_loop_groupB_full_nearest_residual_query0118.png` (89.2mm)

All from the exact same checkpoint set/training run — the difference is policy architecture, not
data or encoder.
