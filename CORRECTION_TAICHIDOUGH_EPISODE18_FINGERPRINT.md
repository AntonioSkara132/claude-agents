# Correction: episode18 fingerprint mismatch is real data loss, not a benign re-serialization

Corrects the reasoning in `REQUEST_TAICHIDOUGH_EPISODE18_DATA_BLOCKER.md`. That report speculated
the sequence-fingerprint mismatch was "likely just a serialization/re-save artifact" because the
*other* hash checks (calibration, initial_particles, tool_geometry, meshes) passed using
`/workspace/data/preprocessed_dataset/episode18_kugla`. That reasoning was wrong, and the
operator's tentative "ignore it if everything is the same" authorization doesn't apply once this
was checked properly.

## What was actually wrong with the earlier reasoning

Those other hash checks verify **static, versioned bundle files** under
`experiments/differentiable_mpm/data/episode18_registered_tools_v1/` — they don't depend on which
raw episode directory `--path` points to at all. They'd pass identically regardless of which
episode18 copy was supplied. They were never evidence about the sequence data specifically.

## What's actually true

- Computed the real fingerprint for `preprocessed_dataset/episode18_kugla` directly
  (`sha256(sha256(pointclouds_interpolated.pt) + sha256(paths_interpolated.pt))`): `8049cdb2...`,
  confirmed mismatched against `episode18.json`'s expected `b5b22253...`.
- `reconstruction_metadata.json` (the file that actually matters for the reconstruction-relocation
  check) records its source `episode_dir` as
  `/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/DeformPath3/snimanje_23_10/episode18_kugla`
  — which resolves to the **corrupted** `DeformPath3` copy on this host, not `preprocessed_dataset`.
  The calibration/reconstruction pipeline was built from that specific export.
- Checked fingerprints for **every** episode18 copy found on this host (8 candidates:
  `preprocessed_dataset`, `interpolated`, `interpolated_10_6`, `DeformPath3`,
  `DeformPath2_interpolated`, `Deformapth2_predownsampled1024_interpolated`,
  `Deformapth2_downsampled_interpolated`, `DeformPath2_no_downsampling_interpolated`). **None
  match** `b5b22253...` — including `DeformPath3` itself, consistent with it being genuinely
  corrupted/truncated (independently confirmed: its `pointclouds.pt`/`pointclouds_interpolated.pt`
  both fail `torch.load` with a truncated-archive error).

## Conclusion

The exact byte-identical source data the calibration was fingerprinted against does not exist
anywhere findable on this host. This is not a safe-to-bypass staleness issue — either the correct
episode18 export was never transferred here, or it's been lost/corrupted alongside the
`DeformPath3` copy. Did not attempt a fingerprint bypass on this basis, and made no
`simulation_result.json`.

Still need: the correct episode18 point-cloud/path sequence data matching fingerprint
`b5b2225335813b6c61f1777194c92cfa5de93b082dcf1a4974cd9cf992b825cb`, or confirmation that this
data is genuinely gone and a different, freshly-fingerprinted episode (recalibrated against
whatever data *does* exist here) is the right path forward instead.

Everything else needed to proceed (parameter resolution via explicit CLI flags, GLIBC/subprocess
environment fixes) is still valid and documented in `REQUEST_TAICHIDOUGH_EPISODE18_DATA_BLOCKER.md`
— only the episode18 sequence data itself is the open blocker.
