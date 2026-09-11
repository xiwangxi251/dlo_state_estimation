"""Offline diagnostics for self-supervised current-frame path ranking."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import cv2
import mujoco
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="id_shape_nominal_current")
    ap.add_argument("--camera", choices=["opst", "wrist"], default="opst")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, default=Path(os.environ.get("DLO_RUN_ROOT", repo_root / "data" / "recorded_run")))
    ap.add_argument("--project-src", type=Path, default=Path(os.environ.get("DLO_PROJECT_ROOT", repo_root.parent)))
    args = ap.parse_args()
    sys.path.insert(0, str(args.project_src / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(repo_root / "trackdlo_standalone" / "src"))
    from dlo_position.geometry import transform_points
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario
    from trackdlo_standalone.geometry import depth_to_meters, resample_polyline
    from trackdlo_standalone.initialization import depth_skeleton_paths, depth_skeleton_paths_global, ordered_skeleton_pixels, segment_hsv
    from trackdlo_standalone.metrics import frame_metrics
    episode = args.run_root / "episodes" / "expert" / args.scenario / "seed_20280804"
    with (episode / "episode.json").open("r", encoding="utf-8") as f: meta = json.load(f)
    cfg = env_config_for_scenario(get_scenario(args.scenario), seed=int(meta["result"]["requested_seed"]), episode_seconds=15.0)
    cfg.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(cfg)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    camera_id = int(env.dynamicvla_opst_camera_id if args.camera == "opst" else env.dynamicvla_wrist_camera_id)
    camera_name = env.config.dynamicvla_opst_camera_name if args.camera == "opst" else env.config.dynamicvla_wrist_camera_name
    K = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
    traj = np.load(episode / "trajectory.npz", allow_pickle=False)
    spec = mujoco.mjtState(int(traj["state_spec"]))
    cap = cv2.VideoCapture(str(episode / ("global.mp4" if args.camera == "opst" else "wrist.mp4")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok, bgr = cap.read()
        if not ok: break
        if frame < 20: continue
        si = int(traj["frame_state_indices"][frame]); mujoco.mj_setState(env.model, env.data, traj["states"][si], spec); mujoco.mj_forward(env.model, env.data)
        renderer.enable_depth_rendering(); renderer.update_scene(env.data, camera=camera_name); depth = depth_to_meters(renderer.render().copy())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = segment_hsv(rgb, (112,180,80), (130,255,255))
        truth = transform_points(np.linalg.inv(world_from_camera_optical(env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id])), env.data.xpos[env.cable_ids].copy())
        candidates=[]
        for name, fn in (("ordinary", lambda: [ordered_skeleton_pixels(mask)]), ("depth", lambda: depth_skeleton_paths(mask,depth,K)), ("global", lambda: depth_skeleton_paths_global(mask,depth,K))):
            try:
                raw=fn()
                if isinstance(raw,np.ndarray): raw=[raw]
                for j,pix in enumerate(raw):
                    pts=[]
                    for row,col in np.asarray(pix,np.int32):
                        if not (0<=row<depth.shape[0] and 0<=col<depth.shape[1]): continue
                        d=float(depth[row,col])
                        if not np.isfinite(d) or d<=0:
                            q=depth[max(0,row-2):row+3,max(0,col-2):col+3]; q=q[np.isfinite(q)&(q>0)]
                            if not len(q): continue
                            d=float(np.median(q))
                        pts.append(((col-K[0,2])*d/K[0,0],(row-K[1,2])*d/K[1,1],d))
                    if len(pts)<4: continue
                    p=np.asarray(pts,float); p=p+0.014*p/np.maximum(np.linalg.norm(p,axis=1,keepdims=True),1e-9)
                    steps=np.linalg.norm(np.diff(p,axis=0),axis=1)
                    # Features available without ground truth.  Curvature is
                    # computed after a robust step clip so one depth hole does
                    # not dominate the score.
                    tangent=np.diff(p,axis=0); tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
                    turn=np.mean(1.0-np.clip(np.sum(tangent[:-1]*tangent[1:],axis=1),-1,1)) if len(tangent)>1 else 1.0
                    depth_slope=np.abs(np.diff(p[:,2]))
                    metric_a = frame_metrics(resample_polyline(p, len(truth)), truth)
                    metric_b = frame_metrics(resample_polyline(p, len(truth))[::-1], truth)
                    rows.append(dict(frame=frame, name=f"{name}_{j}", length=float(steps.sum()),
                                     step_med=float(np.median(steps)),
                                     step_p95=float(np.percentile(steps, 95)),
                                     step_max=float(np.max(steps)), turn=float(turn),
                                     depth_jump=float(np.percentile(depth_slope, 95)),
                                     nearest_cm=float(np.min(np.linalg.norm(p[:,None,:]-truth[None,:,:],axis=2),axis=1).mean()*100),
                                     ordered_cm=float(min(metric_a["ordered_error_m"], metric_b["ordered_error_m"])*100)))
            except Exception:
                pass
    import csv
    with args.out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    cap.release(); renderer.close(); env.close(); traj.close()

if __name__ == "__main__": main()
