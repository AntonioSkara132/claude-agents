# Episode20 (temporal_v2): clean run, first try, no patching needed

Contrast note to the episode3 thread. Tried `episode20_kugla_temporal_v2` (10 chunks,
`calibration_inputs_v2/dataset_episode20_kugla_temporal_stretch_clamp_v2.json`) as a
second new-episode test.

Everything checked out immediately: `source_frame: mocap` (correct, unlike episode3),
`table_alignment_reused_from_episode18: true` in its own `validation.json`, every file
hash in the chunk config matched on the first check, `expected_sequence_fingerprint`
already populated, training/validation windows already disjoint. Point-cloud check
against the validated `T_M_DO_new` transform: 3.4mm median / 6.0mm p90 residual —
holds up on a third episode now (episode18, and this one).

dom_retrieval has real training data for this episode too (segment boundaries match the
chunk boundaries almost exactly), so ran an actual model prediction: `vinn_rbf` seed27
(the zero-residual RBF checkpoint) on `episode20_kugla_chunk01`'s matching segment.
Simulation completed cleanly on the first attempt: 2836/2836 steps, no failure, tools
correctly on the dough at frame 0, visible manipulation by the end. 16.6mm avg / 4.4deg
for this specific query. Published on the report:
https://claude.ai/artifact/9vf5L8KX8hCxBDmk59Mdae

Take: the episode3 issues (missing tool manifest, config schema mismatches, the
still-open point-cloud/tool-pose transform question) were real, specific
data-preparation gaps for that dataset, not something wrong with the simulator, the
calibration chain, or the export pipeline in general — episode20 shows the whole chain
works cleanly when the input data is actually complete and consistent.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
