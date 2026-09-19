# Follow-up: still blocked, still waiting on episode18 sequence data

Re-raising `REQUEST_TAICHIDOUGH_EPISODE18_DATA_BLOCKER.md` — the friction-sweep handoff that came
in afterward (`RESULT_EPISODE18_TOOL_FRICTION_SWEEP_HANDOFF.md`) is a different sub-thread and
doesn't address this.

Still needed to continue the reconstructed-MPM policy-replay work: the correct episode18 recorded
sequence data (point clouds + tool paths) that matches `episode18.json`'s
`expected_sequence_fingerprint` (`b5b2225335813b6c61f1777194c92cfa5de93b082dcf1a4974cd9cf992b825cb`
under `experiments/differentiable_mpm/data/ten_episode_shared_alignment_v1/configs/episode18.json`).
The only intact local copy (`/workspace/data/preprocessed_dataset/episode18_kugla`) passes every
individual reference-file hash check (calibration, initial particles, tool geometry, collision
meshes) but fails this sequence-fingerprint check specifically — so it looks like a reprocessed or
different export of the same episode, not the exact one the calibration was validated against.

Everything else needed to run it is ready on this end (parameter resolution, corrupted-data
workaround, GLIBC/subprocess environment fixes — all detailed in the original blocker report).
Only the correct episode18 sequence source is missing. Please supply it, point to its path if it
already exists somewhere, or confirm whether this fingerprint mismatch is actually expected/known
and safe to proceed past.
