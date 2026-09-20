# Episode3 alt-dynamics input validation: 3 fixed locally, 1 real blocker

Ran the "input-only validation" that `episode3_alt_dynamics_ranges_table_aligned_v1`'s
own `readiness.json` listed as the next step (`next_step: input-only validation`,
blockers: "input-only validation has not been run", "forward and backward qualification
have not run"). Tool manifest is now present and hashes clean (thanks for the transfer).
Tried to actually run chunk `episode3_altdynamics_473a99dace623f9e_pc000066_pc000138`
through `forward_video_v2`. Found 4 issues, fixed 3 in a local config copy, blocked on the
4th.

## Fixed locally (all in a scratch copy under `/workspace/runs/episode3_test/`, nothing
touched under `/workspace/TaichiDough/` itself)

1. **Stray top-level keys**: the per-chunk config
   (`configs/episode3_altdynamics_..._pc000066_pc000138.json`) has `material_model` and
   `status` at the top level; `ExperimentConfig`'s loader rejects unknown fields
   (`"Unknown experiment settings: material_model, status"`). Removed both for local use
   — they read as dataset-manifest-level provenance fields that leaked into the
   per-episode config.
2. **`tool_sdf_resolution`/`tool_mesh_scale`/`backend` misplaced under `simulation`**:
   these belong at the top level of `ExperimentConfig` (that's where episode18's
   registered-tools config has them) but this config nests them inside `simulation`
   instead, which raised `"Unknown/fitted mass settings in simulation: backend,
   tool_mesh_scale, tool_sdf_resolution"`. Moved `tool_sdf_resolution`/`tool_mesh_scale`
   to the top level (values unchanged: 64 / 0.001, matching episode18's registered-tools
   config) and dropped the duplicate `simulation.backend`.
3. **Training/validation window overlap**: `training: {end_frame: 42}` and
   `validation: {start_frame: 42, end_frame: 42}` overlap at frame 42, which
   `ExperimentConfig.validate()` rejects (`"Training and held-out windows must be ordered
   and disjoint"`, requires `training.end_frame < validation.start_frame`). Set
   `training.end_frame` to 41 locally. This is probably true of all 39 chunks generated
   the same way, not just this one.
4. **`reconstruction_metadata.json` hash drift**: `expected_sha256.reconstruction_metadata`
   in the config doesn't match the actual local file (expected `2be7730b7e8b...`, actual
   `5b504fe75cb4...`). Checked the actual file's content before overriding anything: it's
   well-formed, self-consistent (correct relative `episode_dir`, sensible bounds for this
   chunk, frame 0). Read as the same benign "file regenerated after the config's hash was
   captured" pattern seen for every chunk-based dataset this session (episode18's chunks
   and relocated chunks both had this same field drift). Used the actual local hash in my
   scratch config copy rather than the stale expected one.

## Blocked: `source_frame` must be `"mocap"`, this calibration's is `"episode3-table-aligned"`

`table_aligned_identity_calibration.json` has `source_frame:
"episode3-table-aligned"`, `scene_frame: "episode3-table-aligned"` — matching your
README's note that this dataset's point clouds and poses are pre-transformed to
table-aligned before storage (no calibration-time transform needed, hence "identity").
But `prepare_experiment` (`data.py`) hard-checks:

```python
if calibration.source_frame != "mocap" or calibration.scene_frame not in {
    "mocap", "table-aligned", "episode3-table-aligned"
}:
    raise ValueError("Recorded dough replay requires mocap source and mocap or table-aligned scene coordinates")
```

`scene_frame` is allowed to be `episode3-table-aligned` explicitly, but `source_frame`
must literally be `"mocap"` — no exception for a pipeline that pre-transforms before
storage. I don't know if that's a real physical requirement (something about how
`load_observation_sequence` or downstream code interprets `source_frame`) or just a
schema string that should include `episode3-table-aligned` alongside `scene_frame`'s
allowed set. Didn't want to relabel it myself and risk masking a real assumption
mismatch (e.g. if `source_frame` also gates some other transform or distortion handling
downstream I haven't traced).

**Ask**: is `source_frame: "episode3-table-aligned"` intentional and this check needs
widening (mirroring the `scene_frame` allowance), or should this dataset's calibration
actually declare `source_frame: "mocap"` since the underlying data originated from mocap
even though it's stored pre-transformed?

Once that's resolved I expect this dataset runs cleanly — everything else (paths, hashes
once corrected, particle reconstruction, tool geometry) checked out fine, and I've
already got the reusable local config-fixing script for the other 38 chunks if useful.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
