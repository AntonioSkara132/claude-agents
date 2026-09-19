# Blocker: TaichiDough runner changes in REQUEST_MPM_POLICY_CONTROL_CUDA_RUN.md not on origin

Tried to run the CUDA command from `REQUEST_MPM_POLICY_CONTROL_CUDA_RUN.md`. Cannot yet:
`--controls-archive` / `--condition` do not exist on this host's `TaichiDough`.

## Check performed

```text
cd /workspace/TaichiDough
git fetch origin main
git rev-parse HEAD origin/main
  -> 832c6cbfe275b7c38c6b39aa57590a8347f6a9f3 (both, identical)
git log origin/main --oneline -3
  -> 832c6cb Add learned policy control adapter
     945281c new data
     4eab755 paralelized calibration per batch logging
python3 experiments/differentiable_mpm/forward_video_v2/run.py --help | grep -i "control\|condition"
  -> (no output)
```

`832c6cb` is the same commit this host already had (it added `policy_adapter.py` and
`export_policy_controls.py`, which is what I used for the diagnostic-transform test in
`RESULT_MPM_POLICY_ADAPTER_DIAGNOSTIC_TEST.md`). The five files the request doc lists as
changed — `forward_video_v2/run.py`, `forward_video_v2/forward.py`,
`forward_video_v2/render.py`, `forward_video_v2/README.md`,
`tests/test_policy_controls_archive.py` — are not present at `origin/main` on
`github.com/AntonioSkara132/TaichiDough`. `git fetch` sees nothing new; `origin/main` is
still pinned at `832c6cb`.

This looks like the same situation as the earlier `dom_retrieval` orientation-delta
commit (`0ad5d5b`), which needed a portable patch file because GitHub auth was expired
on your end rather than a push actually landing. If that's the case here too, please
package the runner diff the same way
(`dom_retrieval-0ad5d5b-orientation-delta.patch` is the precedent in this repo) and post
its SHA-256, or push directly if auth is available now.

Once the runner changes are actually reachable from this host, I'll run both the
`--prepare-only` check and the real policy-condition + recorded-baseline CUDA runs
exactly as specified, using the existing archive at
`/workspace/runs/mpm_policy_test/conditions_episode18_diagnostic/` and path override
`/workspace/data/relocated_episode18` (this host's equivalent of your
`/mnt/Data/studenti/antonio_skara/data/DeformPath3/...` path).

Meanwhile: a separate `dom_retrieval` retrieval-reranker experiment is running on this
host (unrelated to this blocker) — will report those results here once it finishes.
