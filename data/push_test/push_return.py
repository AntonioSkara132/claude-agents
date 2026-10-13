"""Synthetic test: one tool pushes the dough 80 mm and comes back; does the dough stay pushed or follow the tool?"""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/TaichiDough")
from pathlib import Path
from experiments.differentiable_mpm.config import load_config
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.policy_adapter import build_condition_poses, ConditionControls, _velocities
from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
base = Path("/workspace/TaichiDough")
cfg_path = base / "experiments/differentiable_mpm/data/episode14_dynamics_density1200_roi_temporal_v3/configs/episode14_dynamics_chunk11_von_mises_1000pa_endframes_framedepth_v1.json"
with reference_policy("frozen"):
    config = load_config(cfg_path); init_runtime("cuda", "f32", cpu_threads=1, seed=0)
    prepared = prepare_experiment(config, split="training", build_sdf=True); modules = get_reference_modules(physics_version="corrected-v1")
dt = prepared.simulation_config.dt
sequence = modules.dynamics.load_observation_sequence(base / config.paths["episode"])
calibration = modules.topview.load_calibration(base / config.paths["calibration"])
geometry = modules.dynamics.load_tool_geometry(base / config.paths["tool_geometry"], sequence.names)
rec_times = sequence.times - sequence.times[0]; rec_poses = sequence.poses[:, :, :7].astype(np.float64)
scene = np.asarray(calibration.scene_from_source, dtype=np.float64); marker = np.asarray(geometry.marker_from_mesh, dtype=np.float64)
rec_ctrl_times = np.arange(int(rec_times[-1] / dt) + 1) * dt
rec_sim_poses, _ = build_condition_poses("recorded_full_pose", rec_ctrl_times, rec_times, rec_poses, None, None, scene, marker)
x0 = np.asarray(prepared.initial_state.x, dtype=np.float64); centroid = x0.mean(0); top = x0[:, 1].max()
# contact pose of the pressing tool: the recorded tool-0 pose at the frame where its centre is closest to the dough centroid
d0 = np.linalg.norm(rec_sim_poses[:, 0, :3] - centroid, axis=1); k = int(d0.argmin()); contact = rec_sim_poses[k, 0].copy()
print(f"dough centroid {centroid.round(3)}, top {top:.3f} m; tool-0 closest recorded pose at step {k}: {contact[:3].round(3)}, {d0[k]*1000:.0f} mm from centroid")
# synthetic path for tool 0: approach from -x, push through the dough by 80 mm, hold, return, hold. Tool 1 parked 0.3 m above its start.
# push direction: horizontally from the recorded contact pose toward the dough centroid (the blade sits ~10 cm from the
# tool-frame origin, so translating the recorded contact pose along this line sweeps the blade through the dough)
direction = centroid - contact[:3]; direction[1] = 0.0; direction /= np.linalg.norm(direction)
back, forward = 0.06, 0.10   # start 60 mm behind the contact pose, push to 100 mm beyond it (through the dough), and return
segs = [(0.3, 0.0, 0.0), (1.0, 0.0, 1.0), (0.3, 1.0, 1.0), (1.0, 1.0, 0.0), (0.3, 0.0, 0.0)]
times, alphas = [0.0], [0.0]; t = 0.0
for dur, a0, b0 in segs:
    n = int(round(dur / dt)); times += list(t + dt * np.arange(1, n + 1)); alphas += list(a0 + (b0 - a0) * np.arange(1, n + 1) / n); t += dur
times, alphas = np.array(times), np.array(alphas); steps = len(times)
poses = np.zeros((steps, 2, 7)); poses[:, 1] = rec_sim_poses[0, 1]; poses[:, 1, 1] += 0.30
HOFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
poses[:, 0, :3] = contact[None, :3] + direction[None] * (-back + (back + forward) * alphas[:, None]); poses[:, 0, 1] += HOFF
poses[:, 0, 3:] = contact[3:]
print(f"push direction {direction.round(2)}, blade path from {poses[0,0,:3].round(3)} to {poses[int(1.3/dt),0,:3].round(3)}")
vel = _velocities(poses, times)
controls = ConditionControls(poses, vel, float(dt))
end_push = int(round(1.3 / dt)); end_all = steps - 1
results = {}
for stick in ([float(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else (0.0, 0.3)):
    params = dict(prepared.parameters); params["tool_stickiness"] = stick
    stepper = Stepper(prepared.simulation_config, params, capacity=66, sdf=prepared.sdf)
    stepper.load_state(0, prepared.initial_state); slot = 0; snaps = {}
    for i in range(steps):
        stepper.advance(slot, controls[i]); slot += 1
        if i in (end_push, end_all): snaps[i] = np.asarray(stepper.state(slot).x, dtype=np.float64).copy()
        if slot == stepper.capacity - 1 and i + 1 < steps: stepper.load_state(0, stepper.state(slot)); slot = 0
    xp, xr = snaps[end_push], snaps[end_all]
    tool_end = poses[end_all, 0, :3]
    near_tool = np.mean(np.linalg.norm(xr - tool_end, axis=1) < 0.05)
    results[stick] = dict(push_shift_mm=float((xp.mean(0) - centroid) @ direction * 1000), return_shift_mm=float((xr.mean(0) - centroid) @ direction * 1000),
                          push_back_mm=float((xp.mean(0) - xr.mean(0)) @ direction * 1000), frac_within_5cm_of_tool_at_end=float(near_tool),
                          height_max_push_mm=float(xp[:, 1].max() * 1000), height_max_return_mm=float(xr[:, 1].max() * 1000))
    np.savez(f"/workspace/runs/episode14/push_test/stick{stick}_h{int(HOFF*1000)}.npz", x0=x0, push=xp, ret=xr, poses=poses[[0, end_push, end_all]])
    print(f"stickiness {stick}: dough centroid shift along push after push {results[stick]['push_shift_mm']:.1f} mm, after return {results[stick]['return_shift_mm']:.1f} mm (pulled back {results[stick]['push_back_mm']:.1f} mm); {near_tool*100:.0f}% of particles within 5 cm of the tool at the end", flush=True)
json.dump(results, open(f"/workspace/runs/episode14/push_test/results_h{int(HOFF*1000)}.json", "w"), indent=1)
