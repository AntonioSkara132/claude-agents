# Simulator-in-the-loop search, learned progress reward, timing (2026-10-01)

Follow-up to `RESULT_CONDITIONAL_PRIOR_CVAE_NEW_BEST.md`. Checkpoint throughout: the current best CVAE
(`configs/deformpath_cvae_best.yaml`: conditional prior, joint start, fixed-scale clouds; seed 27). All
simulation uses the 13 calibrated TaichiDough chunks of episode 18 (~0.5 s windows), identical physics
(`rl/stroke_env.py` reproduces `run.py` to 0.01-0.04 mm chamfer).

## 1. Episodic Gym env + CMA-ES over the CVAE's latent (`rl/stroke_env.py`, `rl/cma_search.py`)

One `step` = one stroke simulated to completion (~2.4 s after a 35 s setup); action = offset from the
conditional prior's mean, decoded by the frozen CVAE (a = 0 is the policy's own stroke). Reward = -chamfer
(mm) between the final dough and the recorded run's final dough. Fits a black-box/episodic search, not
dense-step RL (no per-step reward hook; no tool-pose adjoint).

**Correction.** An earlier version fed the action as the raw latent, so "z = 0" was NOT the policy's
prediction (the prior mean has norm 0.7-2.8). The chunk06 "squashed edge" defect reported earlier was a
property of that off-prior stroke. Numbers below use the real prediction.

**13-chunk sweep** (`rl/multi_chunk_diagnostic.py`; CMA-ES popsize 12 x 4 generations = 48 rollouts,
started from the policy's stroke; ~0.4 mm search noise, the simulator is not bit-reproducible under load):

| chunk | hold still | policy | CMA-ES | lift of moved dough, policy / human |
|---|---|---|---|---|
| 01 | 2.68 | 3.81 | 2.33 | +30 / +12 mm |
| 02 | 3.40 | 3.76 | 2.69 | +22 / +10 |
| 03 | 3.17 | 7.54 | 3.13 | +27 / +5 |
| 04 | 4.00 | 2.94 | 2.48 | +23 / +16 |
| 05 | 5.79 | 4.47 | 3.93 | +28 / +18 |
| 06 | 3.21 | 4.53 | 2.75 | +22 / +11 |
| 07 | 3.12 | 5.24 | 2.98 | +21 / +6 |
| 08 | 3.06 | 3.88 | 2.95 | - |
| 09 | 6.43 | 6.76 | 5.46 | +26 / +8 |
| 10 | 2.16 | 2.77 | 2.05 | - |
| 11 | 3.14 | 2.48 | 2.34 | +6 / +3 |
| 12 | 3.02 | 3.91 | 2.45 | +17 / +10 |
| 13 | 3.21 | 3.35 | 2.31 | +18 / +10 |
| **mean** | **3.57** | **4.27** | **2.91** | |

- The policy's own stroke beats holding still on 3/13 chunks: by simulated outcome the imitation policy is
  on average worse than doing nothing (4.27 vs 3.57 mm). CMA-ES beats both on 13/13.
- Consistent signature: the policy lifts the dough it moves about twice as much as the human does
  (dough piles up instead of spreading). Part of this is physics: even the recorded stroke ends ~7 mm
  taller in simulation than in reality (section 3).

**Leave-one-chunk-out distillation** (`rl/finetune_prior.py`): fine-tune only the prior p(z | c) on 12
chunks' CMA-ES latents (Gaussian NLL + KL to the original prior), predict the 13th with no search.

| | hold still | policy | constant mean offset | tuned prior | CMA-ES on that chunk |
|---|---|---|---|---|---|
| mean chamfer, 13 folds | 3.57 | 4.27 | 4.07 | 4.25 | 2.91 mm |
| beats the policy | - | - | 9/13 | 7/13 | 13/13 |

The searched strokes do not transfer: with this reward the optimum is "reproduce what the human did in
this chunk", which is not a function of the dough state (two similar states can need different strokes),
so it cannot be learned from the state. Twelve states from one dough is also far too little. A state-based
reward is required for distillation to make sense.

## 2. Learned progress reward (`rl/progress_reward.py`)

Small PointNet (fixed-scale robust normalization, 5 mm voxel canonicalization) -> sigmoid -> p in [0, 1],
trained on 13,303 camera clouds of 43 episodes with soft-label cross-entropy against normalized episode time
plus a pairwise ranking term; 7 held-out episodes (episode 18 among them).

| held-out episodes | dough-size baseline | progress model |
|---|---|---|
| pairs ordered correctly, >=5% / >=20% / one stroke apart | 74 / 79 / 77 % | 85 / 89 / 87 % |
| Spearman with time, mean / worst episode | 0.60 / -0.28 | 0.78 / 0.24 |
| MAE vs normalized time | - | 0.14 |

Calibration: tracks time one-to-one up to ~50% of an episode, then plateaus (true 55-85% -> predicted
~0.57; finished dough 0.71): the visible shape changes little late in an episode, and "time" is a weak
label for "done". A shape-based target (e.g. similarity to the final cloud) is the fix if 1.0 must mean
"ball".

**In the simulator it does not work.** Particles rendered as the depth camera sees them
(`rl/sim_cloud.py`: calibration + intrinsics, z-buffer, table cutoff; 4-6 mm centred chamfer to the real
cloud of the same frame) and scored (`rl/progress_agreement.py`): the simulated human stroke raises p on
only 7/13 chunks; progress and chamfer pick the same winner among hold-still / policy / CMA on 4/13;
Spearman(Δp, -chamfer) = 0.11. Cause: after a stroke the simulated dough is ~7 mm taller than the real one
(98% height 43 vs 36 mm; the physics piles dough up), and the model reads taller as less progressed.
Blocked by simulator fidelity (height), not by the model.

## 3. Timing

Segment waypoints are evenly spaced in time (`preprocessing.py`: linspace), so only the total duration is
missing. `cvae.duration_head` (log duration from z and condition; 3 seeds): 9-10% median error, no better
than the training-median duration (7%): stroke speed is not predictable from dough + pose. At matched clock
times (seed 27, 125 segments): fixed time-step CVAE 23.8 mm raw, segment + median duration 23.2,
segment + duration head 23.3, true duration 22.3 (bound). Keep the segment model, play it back over the
median (or predicted) duration; fixed-step paths cost capacity (motion size 82% vs 87%).

Also: Gaussian-NLL reconstruction with a learned per-waypoint variance (`reconstruction: gaussian_nll`,
`beta_nll`) ties MSE on accuracy and is worse when the model is fed its own pose (31.9 / 30.9 vs 28.6 mm);
sigma rises ~7x along the stroke. Kept off.

## 4. Prior-sampling data collection round (`rl/collect_prior_samples.py`, `rl/surrogate_q.py`, `rl/awr_prior.py`)

Reward: a footprint-only progress model (vertical axis dropped; held-out pair accuracy 82/87/84%). In the
simulator the human's stroke raises it on 9/13 chunks, the same rate as on real clouds, and it does not
favour piled-up dough: over 845 sampled strokes, Spearman(Δp, final height) = -0.17, Spearman(Δp, footprint
radius) = -0.78, the five best strokes per dough end lower than the policy's on 8/10 chunks. Its correlation
with chamfer-to-the-human is ~0 (a goal-directed reward, not an imitation one).

Collection: 13 doughs x (policy + 32 prior samples at T=1 + 32 at T=2) = 845 simulated strokes (~20 min).
The policy's stroke scores positive progress on all 13 (hold-still ~0); best-of-64 beats the policy on
progress 13/13 and on chamfer 10/13; per dough 9-83% of samples beat the policy. Headroom exists.

Leave-one-dough-out tests of whether that headroom is predictable from the dough:

| held-out dough, no search | policy | Q-surrogate pick | AWR-tuned prior | constant top-8 offset | best-of-64 (simulator) |
|---|---|---|---|---|---|
| mean Δp | +0.177 | +0.19 | +0.178 | +0.159 | +0.287 |
| beats the policy on Δp | - | 7/13 | 8/13 | 3/13 | 13/13 |

- Q(c, z) ranks a held-out dough's samples with Spearman 0.34; Q(z) without the dough does the same
  (0.36): what is learnable from 13 states is a dough-independent direction, not a state-dependent choice.
- AWR fine-tuning of the prior (advantage-weighted NLL + KL leash; with KL weight 1.0 the prior barely moves,
  with 0.1 it moves ~1 latent unit) ties the policy either way.

Conclusion of the round: every piece works in isolation (search finds better strokes; the reward is
state-based and not hacked; samples contain the headroom), but a state -> stroke improvement does not
transfer from 13 states of one dough. The missing ingredient is states, not samples per state: hundreds of
dough states (roll the simulator forward with chosen strokes) before fine-tuning can be expected to
generalize. Secondary: Δp over a 0.5 s window is small (±0.05-0.3) against ~±0.4 mm simulator noise;
longer strokes or multi-stroke returns would give a cleaner signal.

## What to do next

1. Many more states: roll the simulator forward with sampled/chosen strokes (3-5 strokes per chain from
   each chunk's dough) to generate hundreds of dough states, re-collect, and repeat the Q / AWR leave-out
   tests. This is the one lever not yet pulled, and the one the round points at.
2. Simulator height fidelity (the 7 mm pile-up): needed before a height-aware reward or long chains.
3. A geometric ball-ness term (compactness incl. height) alongside the footprint progress reward, so
   multi-stroke optimization cannot drift toward heaps the 2D model cannot see.

Code: dom_retrieval `4229784` .. `1a9557b` and later (rl/, duration head). Runs: `/workspace/runs/cma_multichunk_v2/`,
`/workspace/runs/progress_reward/`, `/workspace/runs/cvae_duration/`.
