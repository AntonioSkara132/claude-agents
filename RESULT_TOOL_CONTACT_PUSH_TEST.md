# Result: tool-contact review and push-and-return test (2026-10-13)

Page with animations and tables: https://claude.ai/artifact/Sn7B3APjJty9dURYi4JK3k (private; scripts and numbers in `data/push_test/`).

## Contact model (`coulomb-adhesive-v1`, solver.py 395-445, 468-560)
Grid-node (and particle) response within `tool_contact_padding` = dx/16 = 1.3 mm of the tool SDF. Approaching nodes lose their
normal relative velocity; tangential velocity is scaled by max(0, 1 - mu*|v_n|/|v_t|) (Stomakhin 2013 = DPSI's law, static regime
included). Adhesion: `result -= s*(result - v_tool)` every substep for every node in the band, in all directions, approaching or
separating, no pull-off limit. Floor: tangential velocity x floor_retention*(1 - floor_stickiness) per 0.2 ms substep (a damping
rate, not friction; dt-dependent). Both DPSI and EMPM use Coulomb table friction with a fitted coefficient and no adhesion term
(DPSI floured the tools; EMPM imposes the gripper velocity).

## Push-and-return test (Episode 14 v3 chunk 11 dough, initial material parameters, tool 0 only)
Dough-centroid shift along the push, mm (after push / after return):

| geometry | stickiness 0 | 0.1 | 0.3 |
|---|---|---|---|
| recorded pressing orientation, edge-on, recorded height | 7 / -0.5 | - | 91 / -22 |
| same, blade 10 mm lower | 23 / 17 | 36 / 10 | 48 / -21 |
| same, blade 20 mm lower | 42 / 29 | 51 / 25 | 73 / 18 |
| blade face-on to the push, on the table | 29 / 25 | - | 31 / 23 |

Edge-on, the plate slices through the dough; whatever moves with it is carried by the adhesion term and comes back with the
withdrawing blade (up to 113 mm). Face-on, Coulomb contact pushes the dough on its own and the dough stays pushed; stickiness
changes nothing. Recorded-chunk forward sweep (v3 tool paths, last-frame chamfer mm, stickiness 0 vs 0.3): chunk 11 5.2 vs 19.9,
chunk 08 17.2 vs 70.6, chunk 06 13.8 vs 58.9.

**Conclusion:** the simulator pushes and presses dough correctly; the adhesion term is unphysical on withdrawal and along the
tool face and should be fixed at 0 (friction-only, as DPSI) or replaced by a normal-only adhesion with a pull-off limit. The
floor model should become Coulomb friction with a fitted coefficient. The Episode 14 v3 fit runs with stickiness 0.

## Side finding: the v3 tool paths also fix most adjoint failures
Gradient preflight of the nine Episode 14 training chunks: v2 paths 1 of 9 finite adjoints; v3 (ICP-refined, manual mount) 7 of 9
(07 fails at step 128, 09 at 3520). Most of the "segment-boundary" failures came from tool poses inside the dough.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
