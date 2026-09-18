# Result: K sweep + point-resolution sweep, clean PointNet, real DeformPath data

Follow-up to `RESULT_DOM_RETRIEVAL_JOINT_TRAINING.md`. All runs below use the same joint
(`freeze_encoder: false`, `loss.contrastive: 0.1`) PointNet setup, 200 epochs, same 611-motion /
451-41-119 split as every prior real-data result in this thread. No `dom_retrieval` source
changed; only `data.points`, `data.root` and top-level `k` in copied configs under
`/workspace/runs/`.

## Point resolution: 64 vs 1024

`PointNetEncoder` has no fixed input-size requirement (point-MLP + max/mean pooling over the
point dimension), unlike PointMAE, which is architecturally locked to `input_points=1024`.
`Deformapth2_downsampled_interpolated` clouds actually have ~10,000+ raw points each ("downsampled"
apparently refers to something other than spatial resolution); `Deformapth2_predownsampled1024_interpolated`
is pre-downsampled to exactly 1024. Ran PointNet at `points: 1024` on the latter, K=5, all six
methods, for direct comparison with the existing 64-point result:

| Method | 64pts (mm) | 1024pts (mm) |
|---|---:|---:|
| Direct | 32.22 | **30.99** |
| Nearest | 39.09 | 41.41 |
| Nearest + residual | 37.92 | 36.71 |
| Cartesian | 34.07 | 33.30 |
| Cartesian + residual | 34.55 | 32.95 |
| Latent + residual | 33.67 | 32.26 |

More input resolution helps most methods a few percent; `direct` becomes the best single-factor
result at 30.99mm.

## K sweep (1, 5, 10, 100, 431≈full-memory), 64 points

`k` only affects `cartesian`/`cartesian_residual`/`latent_residual` (`nearest`/`nearest_residual`
always hard-select regardless of `k`; confirmed `cartesian@k=1` exactly reproduces
`nearest`/`nearest_residual`'s numbers, a useful correctness check). Max safe K without hitting
"K exceeds eligible candidates" is `451 - max_train_group_size = 451 - 20 = 431`
(largest training recording group has 20 motions).

| Method | K=1 | K=5 | K=10 | K=100 | K=431 |
|---|---:|---:|---:|---:|---:|
| Cartesian | 39.09 | 34.07 | 32.97 | 31.58 | 32.07 |
| Cartesian + residual | 37.92 | 34.55 | 32.28 | **31.32** | 34.82 |
| Latent + residual | 33.37 | 33.67 | 33.47 | 32.82 | 32.54 |

Monotonic improvement up to K=100, then **reverses** at K=431 for `cartesian`/`cartesian_residual`
(`cartesian_residual` at K=431 is worse than at K=5). Averaging over almost the entire 451-motion
memory washes out the signal from genuinely similar demonstrations — K=100 (~22% of memory) is
close to a sweet spot, not "use everything."

## Combining both: 1024 points at K=100 and K=431

| Method | 64pts/K=100 | 64pts/K=431 | 1024pts/K=431 | **1024pts/K=100** |
|---|---:|---:|---:|---:|
| Cartesian | 31.58 | 32.07 | 31.54 | 31.57 |
| Cartesian + residual | 31.32 | 34.82 | 31.85 | **30.69** |
| Latent + residual | 32.82 | 32.54 | 31.89 | **30.52** |

Extra point resolution counteracts the K=431 signal-washout problem seen at 64 points
(`cartesian_residual` recovers from a bad 34.82mm to a good 31.85mm at the same K=431 once given
1024-point input) — richer per-observation geometry makes retrieval mixture more robust to large
K, not just independently better.

**Best result of the entire investigation: `latent_residual` at 1024 points, K=100 — 30.52mm
average waypoint error, 0.0004961 m² test MSE**, beating even `direct`'s best (30.99mm).

## Where this leaves things

Every method across every config now clusters in a tight 30.5-33mm band, essentially converging
regardless of encoder capacity/K/resolution choice — reinforcing the ~46-independent-recording-group
data-diversity ceiling flagged in earlier reports as the more likely remaining lever than further
architecture/hyperparameter tuning. Differences at this point (30.5 vs 33mm) are plausibly within
noise given only 119 test motions from 9 recording groups.

Full artifacts for every config in this report are under `/workspace/runs/deformpath_cuda_joint200_*`
on this host only.
