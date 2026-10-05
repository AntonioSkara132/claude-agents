# Reduced dough models (MPM-based and cloud-based) and translation tracking (2026-10-01 to 10-05)

Follow-up to `RESULT_SIM_SEARCH_PROGRESS_REWARD_TIMING.md`. Goal: a fast surrogate of dough dynamics for
ranking candidate strokes, and a check that the policies track a moved dough.

## 1. TaichiDough's learned reduced-order MPM model on our calibrated dough

TaichiDough `96c2696` added a POD + residual-MLP model of the full MPM state (x, v, C, F, Jp; rank 8 per
field -> 40 latents) that predicts one physics step given tool controls. Its 19 tests pass; the synthetic
27-particle demo reproduces the documented numbers (rollout position RMS 2.3-4.1 mm vs 4.0-5.9 mm for
persistence).

On our dough (chunk06 reconstruction, 24,000 particles, 3 strokes x 4 windows of 300 physics steps = 60 ms,
8/2/2 whole-trajectory split, `dom_retrieval/rl/collect_reduced_order.py`):

| held-out window | learned rollout (mm, per component) | persistence | constant velocity |
|---|---|---|---|
| stroke onset (step 0) | 0.55 | **0.15** | 0.15 |
| mid-stroke (step 1400) | 1.17 | 2.94 | **1.14** |

Positions compress fine (0.07-0.57 mm) but the affine-velocity field C compresses no better than "unchanged";
validation diverged after epoch ~25 (epoch 6 selected). Latent rollout 10 ms vs ~0.27 s of simulator for the
same 300 steps (~25x), but at this accuracy there is nothing worth running fast. Not usable as is. What would
change it, in order: a coarse latent step (tens of physics steps per step, so whole strokes fit: today a
stroke is 13 GB of stored states), per-field rank (higher for v/C/F), curriculum + noise injection for the
recursive training, then whole-stroke data (collector ready), and a ranking-agreement evaluation.

## 2. A dough model from recorded point clouds only (`dom_retrieval/dynamics/`)

**Frame level does not work.** Per-frame changes of a reduced state (PCA of a 12^3 occupancy grid, or a
height map, or summary features) are measurement noise: lag-1 autocorrelation of frame deltas -0.45, net
change over a stroke 0.2-0.45x the random-walk sum of frame deltas. The existing constant-Jacobian ridge
baseline barely beats persistence in latent space (one-step 1.10 vs 1.32 standardized MSE on test); a
state-dependent Jacobian MLP (`dynamics/jacobian.py`) overfits the noise and diverges in rollouts (8-step
error 5.8x persistence on validation after 150 epochs).

**Segment level works** (`dynamics/segment_jacobian.py`): denoised start/end states (24x24 height map
averaged over 3 frames -> PCA-16 + centroid + log radius) and the stroke's 31 waypoint increments predict
the stroke's outcome: delta_s = drift(s0) + sum_k J(s0, p_k, phase_k) u_k. Grouped split by episode
(451 / 41 / 119 strokes).

| held-out episodes (119 strokes), standardized outcome MSE | |
|---|---|
| persistence (dough unchanged) | 1.25 |
| mean change | 1.23 |
| **constant-Jacobian ridge** | **0.65** |
| state-dependent Jacobian MLP | 0.77 (fits training to 0; overfits) |

Stroke identification (given start and end dough, pick the true stroke among 16 candidates transplanted
from other test segments; chance = mean rank 8.5, top-1 6%): ridge mean rank 4.9, **top-1 23%**; MLP 5.9 /
16%. Conclusions: (a) a learned dough model from recordings alone exists only at the stroke level; (b) with
451 strokes the simplest model wins; (c) it has no sim-to-real gap, costs microseconds with closed-form
stroke gradients, and ranks strokes well above chance, so it can serve as a first filter before simulator
rollouts; (d) it is coarse and only trustworthy near demonstrated strokes. Untested yet: agreement of its
ranking with the simulator and the progress model on the 13 chunks' sampled strokes.

## 3. Translation tracking (`scripts/analysis/translation_tracking.py`)

40 test scenes, cloud + measured pose shifted by +-0.05 / +-0.10 m in x and y; predicted paths compared
with the unshifted prediction.

| checkpoint | start moves by, per 100 mm shift | off-axis drift | stroke-shape change |
|---|---|---|---|
| retrieval + dough-relative anchor (`runs/rel_anchor/cond_rel_t01_seed27_out`), measured or masked | **100.0 mm** | 0.0 | 0.0 |
| CVAE best (`configs/deformpath_cvae_best.yaml`), measured or masked | **100.0 mm** | 0.0 | 0.0 |
| retrieval, aug fix, no anchor (`runs/augfix`) | 99.2 / 101.3 | 0.5-0.8 | 0.0 |
| retrieval before the aug fix (`runs/traj_head`) | **52 mm** | 0.2-0.8 | 0.0 |

The current models are exactly translation-equivariant by construction (no part of the network sees an
absolute position); the pre-fix model followed the dough only halfway, which is the augmentation bug's
signature. Not covered: rotation, dough moving within a stroke (open loop), the simulator.

## Status of pre-fix results

All headline numbers are post-fix (`3f0b988`). Still carrying pre-fix numbers in `docs/RUNNING.md` section
7: progress input, fixed steps, previous-stroke input, dough-size features, per-waypoint residual. Worth
re-running on the current config: progress input and previous-stroke input (both involve the start).

Code: dom_retrieval `3337138` (dynamics), `504dfee` (reduced-order collector), analysis script in
`scripts/analysis/`. Runs: `/workspace/runs/dynamics/`, `TaichiDough/experiments/differentiable_mpm/runs/reduced_order_real/`.
