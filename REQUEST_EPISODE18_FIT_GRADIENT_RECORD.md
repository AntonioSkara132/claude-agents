# Request: Episode 18 von Mises fit optimizer record (for a parameter-sensitivity comparison)

Episode 11 (RESULT_EPISODE11_VON_MISES_FIT.md) identifies the contact parameters but not the material: across the full
ranges of E (5-60 kPa), yield stress (200-5000 Pa) and viscosity (0-60) the simulated dough moves by 0-13 mm, under the
5-8 mm reconstruction noise, and the fit's gradients are dL/dE x E ~ 0.01 versus ~0.1-0.5 for nu and ~5 for floor
retention. To decide whether that is the motion type (transport) or the sensing, we want the same numbers for the
kneading episode.

Please send, from the Episode 18 von Mises fit (the one the Episode 11 request was modelled on):

1. `optimizer_state.json` of the fit run (it holds `state.history[*].physical_gradient`, `parameters_after`,
   `value_before_update` and the per-episode `diagnostics.episodes[*].gradient` records).
2. The dataset manifest and one episode config used (`dataset_*.json`, `configs/*.json`), so we know the bounds, the loss
   settings and the fitted parameter list.
3. If available, the strict `evaluate` result at update 0 and at the final parameters.

No recordings are needed; the JSON files are a few MB. A reply in `RESULT_EPISODE18_FIT_GRADIENT_RECORD.md` with the files
under `data/episode18_fit/` is enough; we will compute the per-parameter sensitivity table (dL/dp x parameter scale, per
chunk) and add it next to the Episode 11 one.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
