# Proposal: split the position loss into a start term (residual) and a shape term (retrieval)

Prompted by the zero-reference results (`RESULT_ZERO_REFERENCE_INTERVENTION.md`,
`RESULT_ZERO_REFERENCE_RBF_RESIDUAL_CONFIRMED.md`): orientation barely degrades when the
reference is zeroed (+15-28%), but position roughly doubles (25-30mm -> 44-48mm) for
every residual-network checkpoint tested, cosine or RBF. That's the concrete evidence
behind this proposal, not just intuition.

## Current loss

`training/losses.py::trajectory_loss` computes one `position_loss` via
`pose_discrepancy(prediction, target)` — a single raw (unaligned, full-trajectory)
position term. It backpropagates through the retrieval reference and the residual
correction jointly. Nothing in the objective distinguishes "the model should nail t=0"
from "the model should nail the relative motion shape."

## Proposal

Split position supervision into the same two terms this investigation already uses for
reporting (`start` and `shape`, see the Method section of the report), and weight them
so each pathway gets a signal suited to what it can actually fix:

```python
start_loss = (pred_xyz[:, 0] - true_xyz[:, 0]).square().mean()   # t=0 only
pred_shape = pred_xyz - pred_xyz[:, :1]
true_shape = true_xyz - true_xyz[:, :1]
shape_loss = (pred_shape - true_shape).square().mean()           # start-aligned
```

Rationale: the residual network has direct access to the current tool pose (it's not
blind to it structurally, even though the zero-reference intervention showed it isn't
using retrieval for this) — hitting the exact start point is something a
proprioception-aware correction should be able to do close to exactly, independent of
retrieval quality. Shape (the relative pattern of motion over the horizon) is what
retrieval should be supplying — a demonstration whose *motion*, not just its absolute
position, resembles the query. Right now both pathways are jointly optimized against a
single raw-position target that conflates the two, which the zero-reference asymmetry
(orientation fine, position not) suggests isn't cleanly separating what each part of the
architecture is actually good at.

## What I'd want to see

Not proposing exact loss weights (that's a hyperparameter search question, not something
I can usefully guess at from here) — just flagging that this decomposition is now
directly motivated by measured behavior, not just the pre-existing reporting
convenience. Happy to help evaluate a trained variant against the same zero-reference
protocol (does splitting the loss reduce the position gap under zero-reference, or make
retrieval's shape contribution measurably stronger) if you build one, using the same
3-seed RBF bandwidth=0.15 setup as the comparison baseline.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
