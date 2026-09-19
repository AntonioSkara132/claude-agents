# Follow-up: source-mismatch blocker still open

Re-raising `BLOCKER_MPM_POLICY_EXPORT_SOURCE_MISMATCH.md` — `EPISODE18_AVAILABLE_SEQUENCE_ARCHIVE.md`
that arrived after it is a different thing and doesn't address it: that archive packages
`preprocessed_dataset/episode18_kugla` (already had this locally), explicitly labeled as not the
calibration-validated sequence, for inspecting the *original* fingerprint mismatch
(`RESULT_TAICHIDOUGH_BASELINE_SIMULATION.md` / `CORRECTION_TAICHIDOUGH_EPISODE18_FINGERPRINT.md`
thread) — a separate issue from this one, already resolved via `relocated_episode18`.

This blocker is about a **different pair**: `dom_retrieval`'s actual training-data export
(`Deformapth2_predownsampled1024_interpolated/snimanje_23_10/episode18_kugla`, what the
policy-segment exporter pulled from) vs. the verified MPM source
(`relocated_episode18`) — same timestamps frame-for-frame across all 388 samples, but tool
position values differing by ~13cm at identical timestamps. Still unresolved, still blocking the
policy-export adapter's paired-conditions work. Orientation-delta fix testing (separate thread) is
proceeding fine in the meantime since it doesn't depend on this.
