# Blocker: dom_retrieval's episode18 training data and the verified MPM source disagree by ~13cm

Ran `dom_retrieval/scripts/export_policy_segment.py` (from `1d6b508`) successfully — no errors,
produced `prediction_xyz.npy`/`phase.npy`/`manifest.json` for query
`snimanje_23_10/episode18_kugla/segment-2d750affdcc9d983-pc000000-pc000046` (checkpoint: Group B
full-history cartesian_residual). Stopped before building the control sequences because the
exported segment's source data doesn't match the verified MPM episode18 source.

## What was found

The manifest's `source_directory` is
`/workspace/data/Deformapth2_predownsampled1024_interpolated/snimanje_23_10/episode18_kugla` —
this is what `dom_retrieval` actually trained on for this checkpoint. Its `paths_interpolated.pt`
sha256 (`787dabdf...`) does **not** match `/workspace/data/relocated_episode18`'s
(`a8600ddf...`) — the exact, fingerprint-verified source the MPM calibration and baseline
simulation use (`RESULT_TAICHIDOUGH_BASELINE_SIMULATION.md`).

Compared the raw `path` arrays directly rather than assume the hash mismatch is cosmetic:

- Same shape `(388, 2, 14)`, same time range (`0.066701904` to `13.00913`), **timestamps line up
  exactly frame-for-frame**.
- **Tool 0 XYZ at the identical timestamp (t=0.0667s) differs by ~13cm in X**:
  `Deformapth2_predownsampled1024_interpolated` = `[0.3655, 0.0058, 0.4965]` vs.
  `relocated_episode18` = `[0.2328, 0.0475, 0.4653]`.

So these are two different exports of the same recording session (same timing structure) with
different *absolute tool-position values* — most likely different resolved camera-to-mocap
calibrations. This thread found exactly this pattern earlier in a different context
(`RESULT_DOM_RETRIEVAL_DOUGH_FRAME`-adjacent investigation): some recordings resolved
`camera_to_mocap` as `static_fallback`, others as `derived_offline_from_apriltag3`, with different
translation/quaternion values. `relocated_episode18` (via its `DeformPath3` source) is confirmed
`derived_offline_from_apriltag3`; haven't confirmed which `Deformapth2_predownsampled1024_interpolated`
resolved to (no `conversion_metadata.json` present in that interpolated-only directory to check
directly).

## Why this blocks the adapter work

If the model's raw prediction (in whatever frame `Deformapth2_predownsampled1024_interpolated`
actually is) were fed directly into the simulator — which is calibrated against
`relocated_episode18`'s frame — it would inject a ~13cm systematic offset having nothing to do
with model accuracy, on a task whose real motions are ~30mm. That would silently corrupt every
paired-condition result. Stopped before building the 5 control sequences rather than guess which
export is authoritative or apply an uncharacterized correction.

## Open question for discussion

Which of these is actually correct, and is there a real, derivable transform between them (e.g.
if `Deformapath2_predownsampled1024_interpolated` used a stale/fallback camera calibration that
was later corrected for the `DeformPath3`/`relocated_episode18` export), or are they simply
incompatible and the checkpoint would need to be retrained on data traceable to the same
calibration as the MPM reconstruction before this integration test means anything?

Exported segment artifacts at `/workspace/runs/policy_export_episode18/` on this host.
