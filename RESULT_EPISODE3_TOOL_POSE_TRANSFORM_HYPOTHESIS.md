# Episode3: found a plausible tool-pose transform, but it breaks point-cloud loading

Follow-up to `RESULT_EPISODE3_RUNS_BUT_TOOLS_MISPLACED.md`. Tried fitting a transform
from a few points rather than guessing blind.

## The fit

Raw tool positions for this chunk cluster tightly (both tools barely move within the
40-frame window: tool0 y in [0.645, 0.735], tool1 y in [0.988, 1.109], both z around
-0.41 to -0.47). Neither identity nor `coordinate_transform.json`'s matrix reconciled
these with the dough. Tried a 90-degree-about-x rotation instead
(`scene = (x, -z, y) + t`, i.e. raw Y and Z swap) with translation solved by least
squares against the dough centroid:

```
T = [[1, 0,  0,  0.2789],
     [0, 0, -1, -0.4219],
     [0, 1,  0, -0.4561],
     [0, 0,  0,  1]]
```

Result for frame 0: tool0 -> (0.201, 0.009, 0.275), tool1 -> (0.348, 0.021, 0.553).
Dough bbox is x=[0.225,0.315], y=[0,0.033], z=[0.348,0.480]. Both tools now land **inside
the dough's height range** (0.009 and 0.021, both within [0, 0.033]) and just outside its
horizontal footprint on opposite sides (tool0 short on x/z, tool1 over on both) — exactly
the shape you'd expect for two tools approaching a small object from either side. That's
a much better fit than random chance for 2 tools x 3 axes.

## But: this can't be the calibration's `scene_from_source`, because point clouds already need identity

Checked before assuming: the raw, untransformed point cloud for this same chunk/frame
already matches the dough particle bbox almost exactly (`[0.229,0.323]` vs dough's
`[0.225,0.315]` in x, etc.) — confirming identity really is correct for point clouds,
consistent with the README. Substituting the fitted matrix above as
`scene_from_source` (the same field the pipeline uses for both point clouds and tool
poses, per the README's design) breaks point-cloud loading:
`"Observation 0 has only 0 valid pixels after filtering"` — the point cloud gets
double-transformed and lands completely outside the expected observation region.

So point clouds and tool poses need genuinely **different** transforms for this
dataset, which doesn't fit the current single-shared-calibration architecture (and
contradicts the README's "point clouds and tool poses use the same rigid transform"
note — that may be true of the *camera_depth_optical_frame -> table-aligned* step
during data preparation, but if so, the raw `paths_interpolated.pt` values I'm reading
aren't post that transform, unlike the point clouds).

## What I need

Does episode3 have (or need) a separate mocap-tag-to-scene calibration for tool poses
specifically, analogous to `episode18_registered_tools_v1/scene_calibration_v2.json`'s
role for episode18 — used only by `ToolReplay`, not by the point-cloud/observation
loading path? If so, where should I look for it or how should it be derived? My fitted
matrix above is a decent starting guess (geometrically plausible, not validated) but I
don't want to publish results built on a self-fit calibration without your confirmation
that a two-calibration architecture (one for points, one for tools) is actually how this
dataset is meant to work.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
