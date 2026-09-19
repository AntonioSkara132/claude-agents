# Deepened HistoryEncoder projection: no meaningful win either

Follow-up to `RESULT_DOM_RETRIEVAL_LEARNED_RERANKER.md`. Different architecture change,
same question: does more capacity right before the retrieval embedding help.

Pipeline (history enabled, which all our real configs use):
`PointNetEncoder (its own Linear->ReLU->Linear) -> tool_features fusion -> GRU ->
HistoryEncoder.projection -> embedding used for cosine similarity / score`.

`HistoryEncoder.projection` was a single `Linear(hidden_dim, embedding_dim)`
(`models/history.py:28`) — confirmed with the user this is the layer meant by
"the linear layer between the encoder and score" from the earlier clarifying exchange
(PointNetEncoder's own projection sits earlier, before the GRU, and was left alone).
Added an opt-in `deep_projection: bool` (`history.deep_projection` in config) that swaps
it for `Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, embedding_dim)`.
Zero-init was not used here (unlike the reranker) since this sits inside the encoder
itself, not as an additive correction on top of an existing signal — trained from
scratch. 1 new test (`test_deep_projection_adds_a_layer_and_stays_functional`), all
existing tests still pass.

Ran `cartesian_residual` + `orientation_delta`, 1000 epochs, fresh init (no warm start,
since the architecture changed shape-compatibility doesn't matter here but there's no
reason to reuse weights for a layer that no longer exists in the same form), against the
same 1000-epoch no-reranker/no-deep-projection checkpoint used as baseline throughout:

| variant | position | orientation |
|---|---|---|
| baseline (single linear projection) | 25.25mm | 6.51° |
| deep projection (Linear-ReLU-Linear) | 26.68mm | 6.39° |

Best validation checkpoint was epoch 264 (not epoch 0 — it did genuinely train, unlike
the warm-started reranker run that reverted to init). Net effect: +1.4mm position,
-0.12° orientation — same noise band as every other architecture variant tried today.

## Read

Two independent capacity increases right before the retrieval score (reranker MLP on
the score, deeper projection on the embedding itself) both landed inside noise. I think
this points away from "the last stage before scoring is underpowered" as the bottleneck,
and I'd stop investigating that specific direction unless you have a different signal
for why it should matter. The retrieval_entropy_normalized staying pinned near 0.99-1.0
across every variant that performs well (this run: 0.9972) keeps pointing at the same
thing — the `ResidualNetwork` correction head is doing essentially all the useful work
regardless of how retrieval is weighted or how the embedding is computed upstream of it.

Code changes are local only, same as the reranker ones (no GitHub auth for this repo in
this session) — say the word if you want a patch bundle.
