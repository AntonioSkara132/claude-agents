# Correction: episode20 "clean first-try" confirmation was run on training data, not held-out data

`RESULT_EPISODE20_CLEAN_FIRST_TRY.md` reported a successful MPM simulation on
`episode20_kugla_temporal_v2` using the `vinn_rbf` seed-27 checkpoint
(`/workspace/runs/vinn_rbf/cond_vinn_rbf_seed27_out/vinn_rbf/checkpoint.pt`), framed
as a clean confirmation on a new/novel episode.

That framing is wrong. Checked directly: `episode20_kugla` is in that checkpoint's
own **train** partition, not test or validation --
`cond_vinn_rbf_seed27_out/split.json` puts it under `train`. The checkpoint's config
uses `seed: 27` on `data.root: Deformapth2_camera1024_interpolated_nodbscan` --
the same seed and data root used throughout the rest of this session's investigation
(bandwidth sweep, dropout sweep, chained-anchor test, points sweep, branch-removal
fix), all of which share the identical deterministic split from `grouped_split`
(train=33 episodes including 20, validation=4: 18/27/37/41, test=9:
26/30/34/35/39/51/53/55/57). The model had already seen episode20 during training,
so the "16.6mm/4.4deg open-loop, clean first try" result reported there is evidence
the model fits its own training data reasonably, not evidence of generalization to
unseen episodes.

## What this affects

- `RESULT_EPISODE20_CLEAN_FIRST_TRY.md` and the report artifact's Episode20 section:
  reframe as a training-data sanity check, not a held-out confirmation.
- Every *other* generalization claim in this session used episode18 (validation
  partition, not train -- confirmed separately, see `RESULT_CHAINED_ANCHOR_BRITTLENESS.md`
  and `BLOCKER_EPISODE26_TOOL_POSE_CALIBRATION_MISMATCH.md`) or episode26 (test
  partition, confirmed held-out). Those are unaffected by this correction.
- The only two test-partition episodes with any TaichiDough simulation data
  prepared are episode26 (tool-pose calibration broken, see the blocker above) and
  episode53 (already flagged "not ready" in its own prep manifest for the same
  category of tool-registration problem, before even attempting it). So there is
  currently **no genuinely held-out episode with working MPM simulation data** --
  episode20's result doesn't count as one, episode26's doesn't count as a validated
  physics run, and episode53 is pre-flagged broken.

## Recommendation

Don't cite the episode20 MPM result as evidence of generalization going forward.
If a genuinely held-out MPM confirmation is wanted, it requires either fixing
episode26's tool-pose transform (now well-characterized: point clouds exact,
poses off by 30-270mm on different axes per tool -- same shape of problem
`T_M_DO_new` fixed for ACT/BeT) or building fresh TaichiDough data for one of the
other 7 test episodes with none prepared, which is an open-ended effort similar to
the unresolved episode3_alt_dynamics saga.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
