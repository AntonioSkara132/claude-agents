"""Animate the push-and-return test in 3-D with the tool surface drawn (from its SDF zero level)."""
import sys, numpy as np
sys.path.insert(0, "/workspace/TaichiDough")
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image
from experiments.differentiable_mpm.config import load_config
from experiments.differentiable_mpm.data import prepare_experiment
from experiments.differentiable_mpm.policy_adapter import build_condition_poses, ConditionControls, _velocities
from experiments.differentiable_mpm.reference_adapter import get_reference_modules, reference_policy
from experiments.differentiable_mpm.runtime import init_runtime
from experiments.differentiable_mpm.solver import Stepper
HOFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
base = Path("/workspace/TaichiDough")
cfg_path = base / "experiments/differentiable_mpm/data/episode14_dynamics_density1200_roi_temporal_v3/configs/episode14_dynamics_chunk11_von_mises_1000pa_endframes_framedepth_v1.json"
with reference_policy("frozen"):
    config = load_config(cfg_path); init_runtime("cuda", "f32", cpu_threads=1, seed=0)
    prepared = prepare_experiment(config, split="training", build_sdf=True); modules = get_reference_modules(physics_version="corrected-v1")
dt = prepared.simulation_config.dt
sequence = modules.dynamics.load_observation_sequence(base / config.paths["episode"]); calibration = modules.topview.load_calibration(base / config.paths["calibration"])
geometry = modules.dynamics.load_tool_geometry(base / config.paths["tool_geometry"], sequence.names)
rec_times = sequence.times - sequence.times[0]; rec_poses = sequence.poses[:, :, :7].astype(np.float64)
scene = np.asarray(calibration.scene_from_source, dtype=np.float64); marker = np.asarray(geometry.marker_from_mesh, dtype=np.float64)
rec_ctrl_times = np.arange(int(rec_times[-1] / dt) + 1) * dt
rec_sim_poses, _ = build_condition_poses("recorded_full_pose", rec_ctrl_times, rec_times, rec_poses, None, None, scene, marker)
x0 = np.asarray(prepared.initial_state.x, dtype=np.float64); centroid = x0.mean(0)
d0 = np.linalg.norm(rec_sim_poses[:, 0, :3] - centroid, axis=1); k = int(d0.argmin()); contact = rec_sim_poses[k, 0].copy()
direction = centroid - contact[:3]; direction[1] = 0.0; direction /= np.linalg.norm(direction)
back, forward = 0.06, 0.10
segs = [(0.3, 0.0, 0.0), (1.0, 0.0, 1.0), (0.3, 1.0, 1.0), (1.0, 1.0, 0.0), (0.3, 0.0, 0.0)]
times, alphas = [0.0], [0.0]; t = 0.0
for dur, a0, b0 in segs:
    n = int(round(dur / dt)); times += list(t + dt * np.arange(1, n + 1)); alphas += list(a0 + (b0 - a0) * np.arange(1, n + 1) / n); t += dur
times, alphas = np.array(times), np.array(alphas); steps = len(times)
poses = np.zeros((steps, 2, 7)); poses[:, 1] = rec_sim_poses[0, 1]; poses[:, 1, 1] += 0.30
poses[:, 0, :3] = contact[None, :3] + direction[None] * (-back + (back + forward) * alphas[:, None]); poses[:, 0, 1] += HOFF; poses[:, 0, 3:] = contact[3:]
FACE_ON = len(sys.argv) > 2 and sys.argv[2] == "face"
if FACE_ON:
    sdf0 = prepared.sdf; dist0 = np.asarray(sdf0.distances[0]); sp0 = np.asarray(sdf0.spacings[0]).reshape(-1); mn0 = np.asarray(sdf0.minimums[0]).reshape(-1)
    surf = mn0[None, :] + np.argwhere(np.abs(dist0) < 0.8 * float(np.max(sp0))) * sp0[None, :]
    # blade = the thin part: keep surface points within the half of the tool farthest from the handle along the longest axis
    ctr = surf.mean(0); w, v = np.linalg.eigh(np.cov((surf - ctr).T)); long_axis = v[:, 2]; proj = (surf - ctr) @ long_axis
    blade = surf[proj < np.percentile(proj, 50)]; wb, vb = np.linalg.eigh(np.cov((blade - blade.mean(0)).T)); n_local = vb[:, 0]
    def rot_q(q):
        x, y, z, w_ = q; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w_), 2*(x*z+y*w_)], [2*(x*y+z*w_), 1-2*(x*x+z*z), 2*(y*z-x*w_)], [2*(x*z-y*w_), 2*(y*z+x*w_), 1-2*(x*x+y*y)]])
    Rc = rot_q(contact[3:]); n_world = Rc @ n_local
    if n_world @ direction < 0: n_world = -n_world
    axis = np.cross(n_world, direction); s_ = np.linalg.norm(axis); c_ = float(n_world @ direction)
    if s_ > 1e-8:
        axis /= s_; K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]); Ralign = np.eye(3) + s_ * K + (1 - c_) * K @ K
    else: Ralign = np.eye(3)
    Rn = Ralign @ Rc
    # quaternion (x, y, z, w) from Rn
    tr = np.trace(Rn); w_ = np.sqrt(max(0, 1 + tr)) / 2; qx = (Rn[2, 1] - Rn[1, 2]) / (4 * w_); qy = (Rn[0, 2] - Rn[2, 0]) / (4 * w_); qz = (Rn[1, 0] - Rn[0, 1]) / (4 * w_)
    q = np.array([qx, qy, qz, w_]); q /= np.linalg.norm(q)
    world_blade = (Rn @ blade.T).T; drop = -(world_blade[:, 1].min()) + 0.003   # blade's lowest point 3 mm above the table
    # re-centre: the blade centre (not the tool origin) must travel along the line through the dough centroid
    blade_offset = world_blade.mean(0); blade_offset[1] = 0.0
    lateral = np.array([direction[2], 0.0, -direction[0]])        # horizontal unit vector perpendicular to the push
    origin_path = centroid[None] - blade_offset[None] + direction[None] * (-0.11 + 0.18 * alphas[:, None])
    origin_path[:, 1] = drop
    poses[:, 0, :3] = origin_path; poses[:, 0, 3:] = q
    print(f"face-on: blade centre starts at {(origin_path[0] + blade_offset).round(3)} (dough centroid {centroid.round(3)}), lateral offset {float((origin_path[0] + blade_offset - centroid) @ lateral)*1000:.1f} mm", flush=True)
    print(f"face-on: blade normal in world {np.round(Rn @ n_local, 2)} (push {np.round(direction, 2)}); tool origin height {drop*1000:.0f} mm; blade bottom 3 mm above the table", flush=True)
controls = ConditionControls(poses, _velocities(poses, times), float(dt))
# tool-0 surface points from the SDF zero level, in the tool frame
sdf = prepared.sdf; dist = np.asarray(sdf.distances[0]); r = dist.shape[0]; sp = np.asarray(sdf.spacings[0]).reshape(-1); mn = np.asarray(sdf.minimums[0]).reshape(-1)
idx = np.argwhere(np.abs(dist) < 0.8 * float(np.max(sp))); local = mn[None, :] + idx * sp[None, :]
if len(local) > 2500: local = local[np.random.default_rng(0).choice(len(local), 2500, replace=False)]
def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
every = 100; runs = {}
for stick in (0.0, 0.3):
    params = dict(prepared.parameters); params["tool_stickiness"] = stick
    stepper = Stepper(prepared.simulation_config, params, capacity=66, sdf=prepared.sdf); stepper.load_state(0, prepared.initial_state); slot = 0; frames = []
    for i in range(steps):
        stepper.advance(slot, controls[i]); slot += 1
        if i % every == 0 or i == steps - 1: frames.append((i, np.asarray(stepper.state(slot).x, dtype=np.float32)[::3].copy()))
        if slot == stepper.capacity - 1 and i + 1 < steps: stepper.load_state(0, stepper.state(slot)); slot = 0
    runs[stick] = frames; xp = frames[min(range(len(frames)), key=lambda j: abs(times[frames[j][0]] - 1.3))][1]; xr = frames[-1][1]
    print(f"stickiness {stick}: shift after push {(xp.mean(0) - centroid) @ direction * 1000:.1f} mm, after return {(xr.mean(0) - centroid) @ direction * 1000:.1f} mm", flush=True)
# render
out = []; c = centroid; az = np.degrees(np.arctan2(direction[0], direction[2]))
for f in range(len(runs[0.0])):
    fig = plt.figure(figsize=(14, 6))
    for col, stick in enumerate((0.0, 0.3)):
        i, x = runs[stick][f]; pose = poses[i, 0]; R = rot(pose[3:]); tool = pose[:3][None] + local @ R.T
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter((x[:, 0] - c[0]) * 1000, (x[:, 2] - c[2]) * 1000, x[:, 1] * 1000, s=1, c=x[:, 1] * 1000, cmap="viridis", vmin=0, vmax=50)
        ax.scatter((tool[:, 0] - c[0]) * 1000, (tool[:, 2] - c[2]) * 1000, tool[:, 1] * 1000, s=1, c="0.35", alpha=0.5)
        ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_zlim(0, 200); ax.set_box_aspect((1, 1, 0.5)); ax.view_init(elev=28, azim=az + 55)
        ax.set_title(f"stickiness {stick}   t = {times[i]:.2f} s   ({'push' if times[i] < 1.3 else 'hold' if times[i] < 1.6 else 'return' if times[i] < 2.6 else 'hold'})")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]"); ax.set_zlabel("height [mm]")
    fig.suptitle(f"Push-and-return on the Episode 14 chunk 11 dough, {'blade face-on to the push, resting on the table' if FACE_ON else f'blade height offset {HOFF*1000:+.0f} mm'} (grey = tool 0 surface from its SDF)")
    fig.canvas.draw(); out.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("P", palette=Image.ADAPTIVE)); plt.close(fig)
path = f"/workspace/runs/episode14/push_test/push_return_3d_{'face' if FACE_ON else 'h'+str(int(HOFF*1000))}.gif"
out[0].save(path, save_all=True, append_images=out[1:], duration=80, loop=0); out[len(out)//2].convert("RGB").save(path.replace(".gif", "_mid.png")); print("saved", path, len(out), "frames")
