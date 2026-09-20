# Blocked: episode3_alt_dynamics_ranges_table_aligned_v1 needs episode3_manual_tools_manifest_v1

Tried to run a simulation from the new
`experiments/differentiable_mpm/data/episode3_alt_dynamics_ranges_table_aligned_v1`
dataset (episode3, 39 chunks, table-aligned, identity calibration; nice touch that the
README explicitly states point clouds and tool poses share one rigid transform — saved me
from re-deriving that the hard way like I had to for the ACT/BeT pose frame earlier
today).

## What's missing

Every chunk config (e.g.
`configs/episode3_altdynamics_473a99dace623f9e_pc000066_pc000138.json`) points its
tool-geometry inputs at:

```
experiments/differentiable_mpm/data/episode3_manual_tools_manifest_v1/
  tool_geometry.json
  collision_manifest.json
  ur_spathla_collision_solid.stl
  gen3_spathla_collision_solid.stl
```

That directory doesn't exist on this host. The only locally-present manifest is
`episode3_manual_tools_manifest_frame0431_v1` (different name), and I checked its file
hashes against what the config's `expected_sha256` declares before assuming it was just a
rename — they don't match:

- `tool_geometry.json`: expected `0bea9b56db47...`, local frame0431 copy is `dad93453eaf4...`
- `collision_manifest.json`: expected `70afda754b56...`, local frame0431 copy is `d44a2820258...`

So it's genuinely different content, not a naming drift I can path-override around.
Everything else checks out: `episode`, `calibration`, `initial_particles` all present
with matching hashes; `reconstruction_metadata` hash differs but that's the expected
relocation-rewrite case (handled by the `expected_sequence_fingerprint` path, same as
episode18's chunks).

Didn't want to substitute the mismatched manifest and produce a simulation with silently
wrong tool collision geometry, so stopped here rather than guess. This also matches the
dataset's own `readiness.json` (`status: ready_for_input_validation`, blockers list
"forward and backward qualification have not run") — sounds like this may not be
surprising on your end.

## What I need

`episode3_manual_tools_manifest_v1` transferred to this host (same
`/workspace/TaichiDough/experiments/differentiable_mpm/data/` layout as everything else),
or confirmation of which manifest this dataset is actually meant to use if
`episode3_manual_tools_manifest_v1` was renamed/superseded and the chunk configs just
weren't updated to match.

Ready to run the moment that's available — everything else about this dataset looks
straightforward to work with (repo-relative paths throughout, no `/home/antonio/...`
absolute-path overrides needed this time, unlike episode18's chunks).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
