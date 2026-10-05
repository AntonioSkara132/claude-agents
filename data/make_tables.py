"""Regenerate the numbers in PAPER_RESULTS_TABLES.md from the raw outputs committed under data/.

  python data/make_tables.py

Per-seed offline results: data/deployment/<run>_seed{7,17,27}.json (dom_retrieval's evaluate_deployment --json).
Simulator sweep: data/sim/cma_sweep_v2/chunkNN.json (rl/multi_chunk_diagnostic.py), leave-one-chunk-out:
data/sim/cma_loo_v2 (rl/finetune_prior.py), prior sampling: data/sim/prior_samples + data/sim/awr_loo.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "Retrieval + dough-relative anchor": "rel_anchor_cond_rel_t01",
    "ACT, anchored, minimum 150 epochs": "act_min150_act",
    "ACT, anchored, patience 50": "gen_anchor_act",
    "CVAE, fixed prior (anchor)": "gen_anchor_cvae",
    "CVAE, fixed prior (smoothed early stop)": "cvae_smooth_cvae",
    "CVAE, conditional prior": "cvae_cprior_cvae",
    "CVAE, conditional prior + joint start": "cvae_start_joint",
    "Selected CVAE, joint start + fixed 5 cm scale": "cvae_size_joint_fixedscale",
    "Selected CVAE + duration head": "cvae_duration_joint_dur",
    "Selected CVAE, Gaussian NLL": "cvae_nll_joint_nll",
    "Selected CVAE, beta-NLL 1.0": "cvae_betanll_joint_betanll",
}


def offline():
    print("## Offline, 3-seed means (start / shape / raw mm, orientation deg, motion size, pose-masked raw, "
          "chained measured / masked / fed-own-pose)")
    for label, run in RUNS.items():
        rows = [json.load(open(p)) for p in sorted(glob.glob(f"{HERE}/deployment/{run}_seed*.json"))]
        if not rows:
            continue
        m = lambda f: np.mean([f(r) for r in rows])
        print(f"{label:48s} {m(lambda r: r['pose_measured']['start_mm']):5.1f} / {m(lambda r: r['pose_measured']['shape_mm']):5.1f} / "
              f"{m(lambda r: r['pose_measured']['raw_mm']):5.1f} | {m(lambda r: r['pose_measured']['orientation_deg']):4.1f} deg | "
              f"{m(lambda r: r['pose_measured']['motion_size']):4.0%} | masked {m(lambda r: r['pose_masked']['raw_mm']):5.1f} | chained "
              f"{m(lambda r: r['chained']['raw_mm_measured']):5.1f} / {m(lambda r: r['chained']['raw_mm_masked']):5.1f} / "
              f"{m(lambda r: r['chained']['raw_mm_fed_back']):5.1f}   (n={len(rows)})")


def simulator():
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{HERE}/sim/cma_sweep_v2/chunk*.json"))]
    c = lambda k: np.mean([r["chamfer_mm"][k] for r in rows])
    print(f"\n## Simulator sweep, {len(rows)} chunks: hold {c('hold_still'):.2f} | policy {c('policy_z0'):.2f} | CMA-ES {c('cma_best'):.2f} mm; "
          f"policy beats hold on {sum(r['chamfer_mm']['policy_z0'] < r['chamfer_mm']['hold_still'] for r in rows)}, CMA beats hold on "
          f"{sum(r['chamfer_mm']['cma_best'] < r['chamfer_mm']['hold_still'] for r in rows)} (by > 0.4 mm on "
          f"{sum(r['chamfer_mm']['cma_best'] < r['chamfer_mm']['hold_still'] - 0.4 for r in rows)}), CMA beats policy on "
          f"{sum(r['chamfer_mm']['cma_best'] < r['chamfer_mm']['policy_z0'] for r in rows)}")
    loo = [json.load(open(p)) for p in sorted(glob.glob(f"{HERE}/sim/cma_loo_v2/chunk*.json"))]
    k = lambda key: np.mean([r["chamfer_mm"][key] for r in loo])
    print(f"## Leave-one-chunk-out, {len(loo)} folds: hold {k('hold_still'):.2f} | policy {k('original_prior'):.2f} | constant offset "
          f"{k('constant_offset'):.2f} (beats policy {sum(r['chamfer_mm']['constant_offset'] < r['chamfer_mm']['original_prior'] for r in loo)}) | "
          f"tuned prior {k('tuned_prior'):.2f} (beats policy {sum(r['chamfer_mm']['tuned_prior'] < r['chamfer_mm']['original_prior'] for r in loo)}) | "
          f"CMA on that chunk {np.mean([r['cma_best_chamfer_mm'] for r in loo]):.2f}")
    awr = [json.load(open(p)) for p in sorted(glob.glob(f"{HERE}/sim/awr_loo/chunk*.json"))]
    if awr:
        print(f"## AWR leave-one-out, {len(awr)} folds: progress reward policy {np.mean([r['policy']['delta_p'] for r in awr]):+.3f} | AWR "
              f"{np.mean([r['awr']['delta_p'] for r in awr]):+.3f} (beats policy {sum(r['awr']['delta_p'] > r['policy']['delta_p'] for r in awr)}) | "
              f"best-of-64 {np.mean([r['best_of_k_delta_p'] for r in awr]):+.3f}")


def dynamics():
    r = json.load(open(f"{HERE}/segment_jacobian_results.json"))["test"]
    e, i = r["latent_mse"], r["identification"]
    print(f"\n## Segment Jacobian, test ({r['segments']} strokes): persistence {e['persistence']:.2f} | mean delta {e['mean_delta']:.2f} | ridge "
          f"{e['ridge_constant_jacobian']:.2f} | MLP {e['jacobian']:.2f}; identification ridge rank {i['ridge_mean_rank']:.1f} top-1 "
          f"{i['ridge_top1']:.0%}, MLP rank {i['jacobian_mean_rank']:.1f} top-1 {i['jacobian_top1']:.0%} (chance {i['chance_mean_rank']:.1f}, {i['chance_top1']:.0%})")


if __name__ == "__main__":
    offline(); simulator(); dynamics()
