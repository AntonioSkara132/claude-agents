# Relative-history representation: recovers most of the lost accuracy AND keeps the chained-anchor fix's robustness

Tests the "unexplored" recommendation from `RESULT_CHAINED_ANCHOR_BRITTLENESS.md`: reintroduce
tool-motion history into `HistoryEncoder`, but as relative deltas instead of raw absolute
position, to recover the accuracy lost by removing the tool-pose branch entirely without
reintroducing the retrieval-discontinuity failure mode.

## Implementation

`models/history.py`: `HistoryEncoder` gets a new `pose_representation` constructor arg with
four modes: `none` (current default, no tool pose fed to the encoder at all -- the existing
branch-removal fix), `absolute` (the original, brittle behavior), `relative_to_first` (each
history frame's pose relative to the window's own first frame), `frame_to_frame` (delta
between consecutive frames, first frame's delta is zero). Config: `history.pose_representation`.

`poses.py`: added `to_frame_deltas` (consecutive-frame version, alongside the existing
`to_relative_delta` from the motion-BeT work, which already did the relative-to-first-frame
conversion and is reused directly here). Verified `to_frame_deltas` round-trips exactly through
sequential `compose_orientation_delta` calls (max error 3e-7).

One real correctness issue caught and fixed: padded (invalid) history slots are zero-filled,
including the quaternion sub-block, which crashes `normalize_quaternions` ("must be finite and
nonzero") when fed through the relative/delta math unconditionally. Fixed by substituting frame
0's own (always-valid, since every sequence has at least one valid frame) pose into invalid
slots before computing, then masking as before -- invalid slots' results are discarded either
way, this just avoids the crash. Verified directly with variable-length histories including the
count=1 edge case, and confirmed gradients flow correctly for both new representations. Full
existing 104-test suite passes unmodified.

## Real results (production dataset, seed 27, bw=0.3, decomposed loss -- same base as the rest of this investigation)

| representation | normal raw | normal orient | epochs (best) |
|---|---|---|---|
| absolute (original, brittle) | 20.56mm | 5.13deg | -- (prior run) |
| none (branch removed) | 44.12mm | 5.78deg | -- (prior run) |
| **relative_to_first** | **23.94mm** | **5.47deg** | 173 (best=143) |
| **frame_to_frame** | **25.99mm** | **5.54deg** | 197 (best=167) |

Both properly converged (long training, well past early-stopping patience) -- not undertrained
artifacts. Both recover most of the accuracy gap: `none`'s 44.12mm drops to 23.94mm
(`relative_to_first`) or 25.99mm (`frame_to_frame`), close to the original `absolute`
architecture's 20.56mm.

## Chained-anchor robustness (multi-episode, same 3 held-out episodes as `RESULT_MULTI_EPISODE_CHAINED_ANCHOR.md`)

| representation | episode26 ratio | episode30 ratio | episode34 ratio |
|---|---|---|---|
| absolute (original) | 3.6x | 4.0x | 5.1x |
| none (branch removed) | 1.0x | 1.0x | 1.0x |
| **relative_to_first** | **0.97x** | **1.05x** | **1.20x** |
| **frame_to_frame** | **1.03x** | **1.07x** | **1.29x** |

Both new representations stay close to 1.0x on every episode -- nowhere near the original
architecture's 3.6-5.1x catastrophic blowup. `relative_to_first` is the tighter of the two
(max 1.20x vs 1.29x) and also the more accurate one under normal conditions.

## Conclusion

**`relative_to_first` looks like a genuinely good fix, not just a mitigation.** It gets within
~3.4mm of the original architecture's accuracy (23.94mm vs 20.56mm) while being just as robust
to anchor perturbation as fully removing the branch (0.97-1.20x vs 1.0x exactly) -- a much
better accuracy/robustness tradeoff than either prior option (`reference_dropout`: ~39-40mm
normal / +44-47% degradation, or full branch removal: 44.12mm normal / +0% degradation).
`frame_to_frame` is a close second, slightly worse on both axes.

Mechanistically this makes sense: the discontinuity came from feeding raw absolute position
(with no built-in invariance) into the one part of the pipeline that decides retrieval. A
window-relative or frame-to-frame representation is naturally bounded and centered near zero
regardless of where in the workspace the anchor sits, so a small perturbation to the anchor
produces a proportionally small, bounded change in the encoded feature -- instead of an
arbitrary absolute-coordinate jump that can land the retrieval query in a completely different
region of embedding space.

## Recommendation

Switch the standing default from `pose_representation: none` (full removal) to
`relative_to_first` for any checkpoint intended for closed-loop/chained deployment. Not yet
done: multi-seed confirmation (seed 27 only, matching the rest of this investigation's current
state), and a direct single-pair bit-exact mechanism check (confirming retrieval indices don't
flip under anchor perturbation, the way it was checked for the `none` fix) -- the aggregate
ratios strongly suggest the mechanism is fixed, but this hasn't been verified at that level of
directness yet for these two representations specifically.

Configs and checkpoints: `/workspace/runs/relative_history/cond_relative_to_first_seed27{.yaml,.log,_out/}`,
`/workspace/runs/relative_history/cond_frame_to_frame_seed27{.yaml,.log,_out/}`. Chained-anchor
raw data: `/workspace/runs/relative_history/multi_chain_relative_to_first.json`,
`multi_chain_frame_to_frame.json`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
