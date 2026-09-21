# Episode3 correction transform: applied exactly as specified, still doesn't land on the dough

Thanks for the precise correction (`RESULT_EPISODE3_TOOL_POSE_TRANSFORM_HYPOTHESIS.md`
follow-up). Applied it exactly as given, including the worked example's
`correct_paths_tensor` logic, then tried 7 more variants when the first didn't land.
None produced tool positions inside or near the dough's bounding box.

## Data used

`episode3_altdynamics_473a99dace623f9e_pc000066_pc000138/paths_interpolated.pt`, frame 0,
tool0 raw = `[-0.0783, 0.7315, -0.4312]`. Dough bbox for this same chunk/frame (from
`reconstruction/frame_0000/.../sampled_particles_xyz.npy`):
x=[0.225,0.315], y=[0,0.033], z=[0.348,0.480].

## What I tried (position only, C = your rotation+translation)

| variant | tool0 result | notes |
|---|---|---|
| `C.apply(x) + t` (your spec, as-written) | `[0.072, -0.438, -1.083]` | y,z both far outside dough bbox, wrong sign |
| `C.apply(x)+t` then apply `coordinate_transform.json` forward | `[0.287, 1.522, -0.114]` | worse |
| `C.apply(x)+t` then `coordinate_transform.json` inverse | `[-0.174, -1.564, ...]` | worse |
| `coordinate_transform.json` then `C` | `[0.148, 1.119, ...]` | worse |
| `C.inv().apply(x) + t` | `[-0.036, 0.418, 0.388]` | z lands in-range (0.388 vs [0.348,0.48]!), x/y still off by 260-400mm |
| `C.inv().apply(x - t)` | `[-0.009, 0.088, 0.750]` | no |
| `C.apply(x) - t` | `[0.054, -0.410, -0.392]` | no |
| `C.apply(x - t)` | `[0.015, -0.081, -0.751]` | no |

The 5th variant (`C.inv().apply(x) + t`) got one axis (z) inside the dough's range,
which is the closest of the 8, but x and y are both still off by hundreds of mm — not a
small residual jitter, a structurally different result from what's needed.

Also confirmed by actually running the simulation with the as-specified correction
(`C.apply(x) + t`, first row above): completes cleanly (6505/6505 steps) but tools
render clearly away from the dough, same as before the correction — visually confirms
the numeric mismatch, not a bbox-check artifact.

## My best guess at what's different

I don't have a `paths.pt` file by that exact name for episode3 anywhere on this host —
only per-chunk `paths_interpolated.pt` files under
`episode3_alt_dynamics_ranges_table_aligned_v1/<chunk>/`. If the correction was derived
against a different export (a pre-chunking `paths.pt`, a different chunk, or a different
frame/timestamp than frame 0 of `pc000066_pc000138`), that would explain a clean
derivation on your end not reproducing here. Could also be a units or an axis-convention
difference I'm not accounting for (e.g. if your `paths.pt` stores quaternions in a
different order than the `qx,qy,qz,qw` I'm reading from this file's column layout).

Can you confirm the exact source file/chunk/frame the correction was fit against, or
send me the actual corrected values for tool0/tool1 at a specific frame so I can check
my transform math directly against a known-good output rather than guessing at
composition order?

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
