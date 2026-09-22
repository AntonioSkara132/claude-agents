# relative_to_start trajectory representation: large start-error win, modest cost elsewhere, real but variable chained-anchor improvement

Follow-up to `RESULT_STOP_GRAD_REFERENCE_EXPOSES_START_POINT_SHORTCUT.md`. That
result showed retrieval's main measurable contribution was a start-point shortcut,
not genuine shape-relevant selection, and that motions in this model are absolute
(not relative-to-their-own-start) at the data level, which is what made that
shortcut structurally possible in the first place. The `relative to first pose`
commit (`749bc72`) fixes this directly and more surgically than stop-grad did:
retrieved candidate trajectories are stored/matched in a representation relative to
their own recording's start, then recomposed against the *query's actual current
pose* (`current_tool_pose(observations)`) via `compose_orientation_delta` before
being used as the reference. This makes start-point matching deterministic by
construction rather than something retrieval has to solve, while leaving retrieval
free to be selected on shape alone.

Trained `cartesian_residual` + RBF (bw=0.3, decomposed start/shape loss), 3 seeds
(7/17/27), config: `trajectory_representation: relative_to_start`.

## Accuracy result (full test set)

| | start | shape | raw | orientation |
|---|---|---|---|---|
| baseline (absolute), seed 27 only | 10.88mm | 19.63mm | 19.95mm | 5.46deg |
| **relative_to_start, 3-seed mean** | **2.36 +/- 0.28mm** | **22.10 +/- 1.14mm** | **22.05 +/- 0.97mm** | **5.80 +/- 0.12deg** |

**Start error drops ~5x (10.88mm -> 2.36mm) and is tight across seeds.** This is
the clean, robust part of the result -- consistent with the mechanism, since
start-point matching is no longer something retrieval has to learn at all. Shape,
raw, and orientation are each modestly *worse* than the single-seed absolute
baseline (shape +13%, raw +11%, orientation +6%) -- a real but small cost. Note the
baseline column is single-seed; a 3-seed absolute baseline would sharpen this
comparison, not currently available.

## Chained-anchor result: real improvement, but noisy across seeds -- don't
## over-read the best seed

Ran the standard single-pair chained-anchor test
(`/workspace/runs/reference_dropout/test_chained_anchor.py`, episode18 validation
pair, same as used throughout this investigation) on all 3 seeds:

| seed | normal raw | chained raw | raw degradation |
|---|---|---|---|
| 7 | 14.68mm | 22.17mm | +51% |
| 17 | 15.48mm | 23.78mm | +54% |
| 27 | 23.87mm | 25.47mm | +7% |
| **mean** | **18.01 +/- 4.16mm** | **23.81 +/- 1.35mm** | **~+32%** |

Seed 27 alone looked like a near-total fix (+7%) and was initially reported that
way before the other two seeds came in -- that was a favorable outlier, not
representative. Averaged properly across 3 seeds, degradation is ~+32%, which is
still a real, meaningful improvement over the original absolute-representation
architecture's +342-359% and is roughly comparable to (maybe modestly better than)
`reference_dropout`'s +44-47% (`RESULT_REFERENCE_DROPOUT_TRAINING.md`) -- but it is
not the clean, near-complete fix the single-seed result implied, and there's
substantial seed-to-seed variance (7% to 54%) that a single run would have missed
entirely. Lesson reinforced: don't trust a single-seed chained-anchor result,
especially not a favorable one, without at least 2-3 seeds -- this is now the
second time this session a single-seed read turned out to be misleading (the first
being the flawed distance-to-static-reconstruction proxy that gave a false
"episode20 is broken" read).

## Comparison across all mitigations tried this session

| mitigation | normal raw | chained-anchor raw degradation | mechanism |
|---|---|---|---|
| none (absolute representation) | ~20mm | +342-359% | -- |
| `reference_dropout=0.25-0.75` | ~29-40mm | +44-47% | blunt: makes reference unreliable during training so residual learns not to over-depend on it |
| full branch removal (`pose_representation="none"`) | ~44mm | 0% (exact) | removes the raw anchor input from the retrieval-query embedding entirely |
| **`relative_to_start`** | **~22mm (3-seed)** | **~+32% (3-seed, noisy)** | **removes retrieval's need to solve start-matching; anchor-sensitivity is reduced but not eliminated since `current_tool_pose` still directly composes the reference position** |

`relative_to_start` currently looks like the best normal-condition-accuracy option
among the robustness-improving mitigations (~22mm vs 29-44mm for the others), with
robustness improvement in between `reference_dropout` and full branch removal. It
doesn't eliminate the chained-anchor effect the way branch removal does (expected --
`current_tool_pose` still directly composes the final/reference position here, so a
perturbed anchor still shifts the output somewhat, just without the
retrieval-selection-flip mechanism that caused catastrophic blowups under the
absolute representation).

## Recommendation

`relative_to_start` is a strong candidate for the new default given the start-error
win is large, consistent, and mechanistically well-understood, and the
chained-anchor robustness, while noisy, is directionally real and never approaches
the catastrophic blowups seen under the absolute representation. Worth running a
3-seed absolute-representation baseline (not just the single seed 27 currently on
hand) before making a final call, since the "modest cost elsewhere" (shape/raw/
orientation) comparison right now rests on one baseline seed. Also worth testing
whether `relative_to_start` and `reference_dropout` compose (apply both together)
for a possible further robustness gain, since they target different parts of the
same underlying problem -- not attempted yet.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
