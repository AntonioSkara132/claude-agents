# RBF kernel retrieval: consistent orientation win, all 3 seeds

Follow-up to `RESULT_DOM_RETRIEVAL_ARCHITECTURE_SWEEP_20260920.md`, which flagged one
seed as preliminary. All 3 seeds are in now and confirm it.

`cartesian_residual`, orientation_delta, `retrieval_mode: rbf`, `retrieval_bandwidth: 0.5`,
Top-100, everything else matched to the rest of today's sweep (new dataset, early
stopping patience 30). Per-recording start/shape/raw position + orientation:

| seed | start | shape | raw | orientation |
|---|---|---|---|---|
| 17 | 19.28mm | 22.32mm | 23.12mm | 5.72deg |
| 27 | 17.35mm | 20.78mm | 21.24mm | 5.68deg |
| 7 | 17.09mm | 22.09mm | 22.37mm | 5.33deg |
| **mean +/- std** | 17.91+/-1.24mm | 21.73+/-0.83mm | **22.24+/-0.94mm** | **5.58+/-0.20deg** |

This is the best orientation number across every condition tried today or in the
original 5-condition ablation matrix (previous best: 6.02deg single-seed baseline;
6.94-7.90deg across every constant-residual/observation-decoupled variant). It also has
the tightest orientation variance of any condition (std 0.20deg vs 0.15-1.53deg
elsewhere). Position (22.24mm raw) is mid-pack, not a standout, but not a tradeoff either
— it's competitive with everything except the best observation-decoupled runs.

Only change from the matched cosine-Top-100 baseline: cosine similarity + softmax
replaced by an RBF/Gaussian kernel over the same normalized embeddings
(`weights = softmax(-||q-k||^2 / (2*bandwidth^2))` over the Top-100 by that same kernel
distance). No changes to the residual network, encoder, or anything else.

## Not yet tried

- RBF combined with observation-decoupled residual (`residual_input: observation`) —
  the best non-RBF family so far; haven't stacked the two.
- Bandwidth sweep — 0.5 was a single guess, not tuned.
- `retrieval_all: true` with RBF (weight every eligible motion, no Top-100 cutoff) — the
  option exists (your `kernel machines` commit) but this run used Top-100.

Say the word on which of these to prioritize, or if there's a specific mechanism you want
checked for why RBF's kernel shape helps orientation specifically (e.g. whether it's
softer/harder than cosine's softmax at the same nominal temperature/bandwidth, which
would show up in retrieval_entropy_normalized — haven't pulled that number for these
three runs yet).
