# Suggested next change: predict orientation as a delta from current measured quaternion

Follow-up to `RESULT_DOM_RETRIEVAL_POSE_FIX_1000EPOCH.md`. The input fix (`1834ca3`) didn't move
`cartesian_residual`'s orientation error at full 1000-epoch budget (9.35°→9.28°, within noise,
still above the 8.81° do-nothing baseline). Input-completeness alone isn't the fix. Here's my top
recommendation for what actually would help, plus two supporting findings from digging further.

## Top recommendation: reparametrize the orientation output as a delta from current orientation

Right now the model predicts an *absolute* target quaternion per waypoint, learned from scratch,
even though the current measured orientation is available as input (after `1834ca3`). Given real
rotation is tiny (mean total rotation only 7.74° per the earlier report) and mostly noise around
"no change," that's a much harder learning target than it needs to be.

Suggested change: reparametrize so the orientation head predicts a **small rotation delta from the
measured current quaternion** (quaternion multiply to compose, not raw additive-then-normalize —
`transform_tool_state` in `data/coordinates.py` already has the quaternion-composition math this
could reuse), and **zero-initialize the delta head** so the default prediction is "no rotation" —
matching the ~90%-of-the-time reality of this dataset. Training then only has to learn the
*correction* on top of a sensible prior, not the whole absolute mapping. This directly targets
what was actually measured (predicted magnitude is already plausible; predicted *direction* is
essentially uncorrelated with truth) rather than another input-side change.

## Two supporting findings while investigating this

1. **The history input is completely unnormalized.** `models/history.py::HistoryEncoder.forward`:
   `auxiliary = torch.cat((tools[mask], times[mask][:, None], elapsed[mask][:, None]), dim=-1)` →
   straight into `self.tool_features` (a single `Linear`), no normalization step anywhere. Not
   catastrophic for this dataset specifically (position happens to be sub-meter scale, quaternion
   is inherently bounded [-1,1], so raw scales aren't wildly mismatched by coincidence), but not
   principled either — worth adding a position-normalization step (reuse the training-set
   mean/std the same way `TrajectoryNormalizer` already does for the target trajectory) while
   touching this code.
2. **`tool_features` feeds raw absolute quaternions per history frame, not relative rotation.**
   A single linear layer treats quaternion components as generic Euclidean features, ignoring the
   manifold/double-cover structure. Feeding the *relative* rotation between consecutive history
   frames (closer to an explicit angular-velocity feature) instead of absolute quaternions per
   frame would likely be a much more directly learnable "is this segment rotating or not" signal,
   given what we now know about the true dynamics.

## Suggested priority

Try the delta-head reparametrization first — it's the one untested idea that directly explains
the measured failure mode (right magnitude, wrong direction) rather than a general architecture
improvement. The two supporting findings are cheaper, lower-priority follow-ons if the delta-head
change alone doesn't fully close the gap to the 8.81° baseline.

Happy to test whatever gets committed, same as the last two rounds.
