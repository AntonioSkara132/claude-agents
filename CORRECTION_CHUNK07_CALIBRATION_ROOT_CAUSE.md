# Real root cause of the chunk07 tool/dough misalignment (correction #2)

Follow-up to `CORRECTION_CHUNK07_CALIBRATION_MISMATCH.md`, which "fixed" the visible
misalignment by swapping in `episode18_registered_tools_v1`'s real `scene_from_source`
matrix — but that was still wrong, just wrong in a way that happened to run without
erroring. User caught it: tools and dough weren't in the same frame at all (rendered
frame 0 showed the dough blob off to one side, tools elsewhere, not touching).

## Actual bug

`episode18_registered_tools_v1`'s `scene_from_source` matrix is calibrated to pair with
**its own** `calibration.json` — the one used for dough reconstruction in every
full-episode run on the hosted page (`rbf_best_predicted`, `obsdecoupled_best_predicted`,
etc. — checked their `forward_config.json`, all use
`episode18_registered_tools_v1/scene_calibration_v2.json` for `calibration` too, not
just for the control-pose transform).

chunk07 instead uses **its own, separate** `calibration.json` for dough reconstruction
(each chunk has its own `scene_calibration_v2.json`, and chunk07's happens to be
identity). Borrowing `episode18_registered_tools_v1`'s `scene_from_source` for the
control-pose transform while the simulator loads dough particles through chunk07's own
(different) calibration puts tools and dough in two different coordinate conventions —
exactly the symptom reported.

## Fix

Use each chunk's **own** `scene_from_source` (matching its own `calibration.json`) with
identity `marker_from_tool_frames`, not a borrowed matrix from a different config. For
both chunk07 (`episode18_kugla/chunks`) and chunk07 relocated
(`episode18_kugla_relocated/chunks`), that own `scene_from_source` happens to be
identity.

Verified against the point cloud, not just eyeballed: dom_retrieval's raw observation
input for the matching query, transformed by the validated camera->mocap matrix, now
lands within **0.007mm median / 1.1mm p90** of the simulator's own chunk07
reconstruction (nearest-neighbor residual, 1024 vs 3991 points). Rendered frame 0 now
shows both tools directly on/around the dough blob, matching every full-episode
reference render.

Reran both chunk07 sims with the fix: both completed cleanly (2669/2669 steps, no
failure). Updated on the hosted page with the point-cloud verification figure:
https://claude.ai/artifact/MmxhTp5Gxscb4MGajC2sk6

## Takeaway for future chunk runs

`scene_from_source` must always be paired with the *same* `calibration.json` it was
computed against — never mix a `scene_from_source` from one config with a
`calibration.json` from another, even if both nominally describe "the same physical
scene." For any future chunk, pull `scene_from_source` from that chunk's own
`scene_calibration_v2.json`, not from `episode18_registered_tools_v1` or any other
chunk's.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
