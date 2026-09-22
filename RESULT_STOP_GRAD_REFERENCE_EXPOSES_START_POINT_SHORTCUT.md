# Stop-grad at reference: degradation concentrates in start error, not shape

Follow-up to the `stop grad at reference trajectory` commit (`aba41b1`). Pulled it,
trained `cartesian_residual` on it, and compared against the existing bw=0.3
decomposed-loss baseline (`RESULT_DECOMPOSED_START_SHAPE_LOSS.md` /
`RESULT_CHAINED_ANCHOR_BRITTLENESS.md` lineage).

## Bug fixed before training

`models/retrieval_policy.py` had `reference = reference.detached()` -- not a real
`Tensor` method, this crashes immediately (`AttributeError`). Fixed to
`reference.detach()`.

## A real gap in the change, not hit by this experiment

The detach is unconditional, placed before the `pose_parameterization ==
"orientation_delta"` branch. For methods *with* a residual network
(`cartesian_residual`, what this experiment tested), this correctly blocks all
reference-to-retrieval gradient, since `reference` also directly composes the final
prediction there -- confirmed this is consistent with the intent. But for
residual-*less* methods (`vinn_rbf`, `nearest`, `cosine` without residual),
`prediction = reference` directly, so detaching leaves **zero gradient path at
all** -- training becomes a no-op for those methods. Confirmed via the existing test
suite: `test_all_policy_modes_and_batch_sizes` and
`test_fresh_memory_keys_and_joint_gradients` now fail with `RuntimeError: element 0
of tensors does not require grad and does not have a grad_fn`. Not fixed (out of
scope for what was asked -- flagging for whoever touches this next). If
residual-less methods need to keep training under this change, the fix is to only
detach the copy of `reference` passed into `self.residual(...)`, not the copy used
to compose the final prediction directly.

## Result (bw=0.3, decomposed start/shape loss, seed 27, full test set)

Reconstructed the baseline checkpoint under the *current* code (its saved config
predates the new `history.pose_representation` field, which now defaults to
`"none"` and silently changes architecture on load) by explicitly setting
`history.pose_representation: "absolute"` before rebuilding -- this reproduces the
original `pose_dim+2` tool-features layout exactly, confirmed by a clean
`load_state_dict(strict=True)`. Without this, `load_experiment` reconstructs the
wrong architecture for any checkpoint saved before that field existed.

| | start | shape | raw | orientation |
|---|---|---|---|---|
| baseline (no stop-grad) | 10.88mm | 19.63mm | 19.95mm | 5.46deg |
| stop-grad | 27.53mm | 25.49mm | 31.27mm | 6.32deg |
| **relative degradation** | **+153%** | **+30%** | **+57%** | **+16%** |

Training dynamics: baseline ran 153 epochs before early stopping; stop-grad
plateaued at 37. Retrieval entropy (normalized) went from 0.993 (baseline) to 0.998
(stop-grad) -- slightly more uniform/diffuse, consistent with retrieval no longer
being actively shaped toward anything task-specific.

## Interpretation

The degradation is **not evenly spread** -- start error more than doubles (+153%)
while shape only degrades mildly (+30%). This is a real, specific finding, not just
"cutting this gradient hurts": the gradient that normally flows from the prediction
loss through `reference` back into retrieval appears to be doing most of its work on
the **start-point** loss term specifically, not on general trajectory-shape
selection. With `loss.start` and `loss.shape` weighted equally (1.0 each) in this
config, retrieval evidently found it easier/cheaper to reduce loss by learning to
pick trajectories whose *starting point* suits the residual's job, rather than
learning genuinely shape-relevant retrieval -- and shape accuracy holds up
reasonably on its own regardless of which trajectory gets retrieved, likely because
the residual network compensates for shape more readily than for start-point offset.

This reframes the earlier retrieval-mechanism findings from this session
(near-uniform/low-information retrieval weights, residual doing most of the real
work -- see the Summary section of the consolidated report): retrieval's one
concrete, measurable contribution looks like it's mostly a start-point shortcut,
not the general "pick a good template motion" role it's nominally supposed to play.

## Recommendation

Don't adopt stop-grad-at-reference as-is -- it's a net accuracy regression on every
axis measured, and the start-error concentration suggests it's removing a
shortcut the model was legitimately relying on, not fixing a pathology. If the goal
was specifically to test/prevent retrieval from doing start-point shortcut-fitting
(as opposed to blocking all reference-to-retrieval gradient), a more targeted
follow-up would keep gradient flowing for the shape-relevant part of the loss while
blocking it for the start-specific part -- not attempted here. Also worth fixing the
residual-less-methods gradient gap before this is used more broadly, since it
silently disables their training entirely.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
