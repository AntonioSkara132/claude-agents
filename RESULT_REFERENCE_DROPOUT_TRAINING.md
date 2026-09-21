# Reference dropout: the real fix for zero-reference robustness (loss decomposition wasn't)

Follow-up to `RESULT_DECOMPOSED_START_SHAPE_LOSS.md`. That result showed the decomposed
loss doesn't change zero-reference robustness — expected in hindsight, since
`intervention` is hard-blocked from running during training
(`RetrievalPolicy.forward`: `if intervention != "none" and self.training: raise`), so no
model had ever actually been trained under a broken reference. Implemented a real fix:
training-time reference dropout, distinct from the eval-only intervention system.

## What changed

- `models/retrieval_policy.py`: new `reference_dropout: float = 0.0` constructor arg
  (validated to `[0, 1]`). In `forward()`, only when `self.training`, each batch sample
  independently has its `reference` zeroed with this probability, right after
  `reference = project_pose_trajectory(reference)` and before it reaches the residual
  network. Eval-time behavior is completely unchanged (dropout only fires under
  `self.training`, same guard style as the existing intervention check).
- `training/trainer.py`: reads `reference_dropout` from config (default 0.0, so every
  existing config is unaffected), passes it through.
- `tests/test_models.py`: 3 new tests (dropout=1.0 always zeros in train mode and never
  in eval mode; dropout=0.0 is a no-op; invalid probability rejected). Full suite:
  101 -> 104 tests, all passing.

Trained two variants on top of the decomposed start/shape loss baseline
(`RESULT_DECOMPOSED_START_SHAPE_LOSS.md`), 3 seeds each: `reference_dropout: 1.0`
("always") and `reference_dropout: 0.5` ("sometimes").

## Result

| condition | normal: raw | normal: orientation | zero-ref: raw | zero-ref: orientation |
|---|---|---|---|---|
| baseline (decomposed loss, no dropout) | 21.07+/-1.67mm | 5.35+/-0.21deg | 44.41+/-1.47mm | 7.15+/-0.16deg |
| **reference_dropout=1.0 ("always")** | 31.84+/-2.84mm | 8.31+/-0.15deg | 44.24+/-1.38mm | 8.31+/-0.15deg |
| **reference_dropout=0.5 ("sometimes")** | 29.38+/-3.57mm | 6.19+/-0.69deg | **34.68+/-6.22mm** | 6.69+/-1.02deg |

Normal/zero-reference gap (raw position, relative): baseline +110.7%, "sometimes" only
**+18.0%**.

## Two distinct findings

**"always" (dropout=1.0) makes the residual network reference-blind, not
reference-robust.** Checked directly, not just inferred from the metrics: for this
checkpoint, `raw_residual` (the network's own delta output) is **bit-exact identical**
whether fed the real retrieved reference or an explicitly zeroed one, even though the
reference itself clearly differs (checked `.abs().max()` on both). The network learned
to ignore the reference input entirely, since it never varied during training. Normal
performance is worse than baseline (31.84 vs 21.07mm) because real retrieval information
is available but structurally unused; zero-reference performance is about the same as
baseline's zero-reference number (44.24 vs 44.41mm) because both are now equally
"blind." Orientation is identical between normal and zero-ref for the same reason
(8.31deg both times) — not evidence of robustness, evidence of indifference.

**"sometimes" (dropout=0.5) is the real result**: a genuine, substantial robustness
improvement (over 100% degradation down to 18%), at a real cost to normal-condition
accuracy (21.07 -> 29.38mm, about 40% worse). This is the expected dropout-style
accuracy/robustness tradeoff, and it's the first intervention in this whole
investigation that actually moves the zero-reference number rather than just measuring
it.

## Recommendation

If retrieval reliability in deployment is a real concern, `reference_dropout` around 0.5
(untuned — haven't swept this) is a legitimate, tested way to buy robustness, at a
meaningful accuracy cost under normal (working-retrieval) conditions. If retrieval is
expected to always work, stick with the decomposed-loss baseline (best normal-condition
numbers, no dropout). Not recommending `reference_dropout=1.0` for anything — it
doesn't produce a "direct" style model as well as the actual `direct` method does
(31.84mm vs 25.87mm, `RESULT_DIRECT_GRU_BASELINE.md`), it just breaks retrieval's
usefulness while paying its full computational/architectural cost.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
