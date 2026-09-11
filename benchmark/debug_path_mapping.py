"""Print current-path interval mapping for one rendered episode frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


def main() -> None:
    root = Path(r"C:\Users\27642\Desktop\dynamic_cable")
    run_root = root / "linux_log" / "expert_grasp_fix_4x50" / "run_20260824_113325"
    scenario = "id_shape_nominal_current"
    episode = run_root / "episodes" / "expert" / scenario / "seed_20280804"
    sys.path.insert(0, str(root / "panda_cable_grasp" / "src"))
    sys.path.insert(0, str(root / "trackdlo_standalone" / "src"))
    from dlo_position.geometry import transform_points
    from dlo_position.recorded_benchmark import camera_matrix, world_from_camera_optical
    from panda_cable_grasp.env.environment import CableGraspEnv
    from panda_cable_grasp.evaluation.motion_diagnostics import env_config_for_scenario
    from panda_cable_grasp.scenarios.registry import get_scenario
    from trackdlo_standalone import TrackDLOConfig, TrackDLOTracker
    from trackdlo_standalone.geometry import backproject_mask, depth_to_meters, voxel_downsample
    from trackdlo_standalone.initialization import segment_hsv
    from trackdlo_standalone.initialization import ordered_skeleton_pixels
    import trackdlo_standalone.initialization as init_module
    from scipy.interpolate import splprep, splev
    from run_trackdlo_current import _current_skeleton_centerline_paths

    with (episode / "episode.json").open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    config = env_config_for_scenario(get_scenario(scenario), seed=int(metadata["result"]["requested_seed"]), episode_seconds=15.0)
    config.dynamicvla_cameras_enabled = True
    env = CableGraspEnv(config)
    renderer = mujoco.Renderer(env.model, height=360, width=480)
    camera_id = int(env.dynamicvla_opst_camera_id)
    camera_name = env.config.dynamicvla_opst_camera_name
    wrist_id = int(env.dynamicvla_wrist_camera_id)
    wrist_name = env.config.dynamicvla_wrist_camera_name
    K = camera_matrix(480, 360, float(env.model.cam_fovy[camera_id]))
    Kw = camera_matrix(480, 360, float(env.model.cam_fovy[wrist_id]))
    trajectory = np.load(episode / "trajectory.npz", allow_pickle=False)
    spec = mujoco.mjtState(int(trajectory["state_spec"]))
    global_cap = cv2.VideoCapture(str(episode / "global.mp4"))
    wrist_cap = cv2.VideoCapture(str(episode / "wrist.mp4"))
    frames = {}
    for index in range(81):
        ok, bgr = global_cap.read()
        ok_w, bgr_w = wrist_cap.read()
        if not ok or not ok_w:
            break
        if index in (20, 80):
            state_index = int(trajectory["frame_state_indices"][index])
            mujoco.mj_setState(env.model, env.data, trajectory["states"][state_index], spec)
            mujoco.mj_forward(env.model, env.data)
            renderer.enable_depth_rendering(); renderer.update_scene(env.data, camera=camera_name)
            depth = renderer.render().copy()
            renderer.enable_depth_rendering(); renderer.update_scene(env.data, camera=wrist_name)
            depth_w = renderer.render().copy()
            frames[index] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, cv2.cvtColor(bgr_w, cv2.COLOR_BGR2RGB), depth_w)
    rgb20, depth20, _, _ = frames[20]
    rgb80, depth80, rgbw80, depthw80 = frames[80]
    print("init_module", init_module.__file__)
    tracker = TrackDLOTracker(K, TrackDLOConfig(hsv_lower=(112,180,80), hsv_upper=(130,255,255), visibility_mode="neighborhood", min_visible_nodes=3, accept_nonconverged=True, max_iter=2))
    from trackdlo_standalone.initialization import initialize_nodes
    direct_nodes, _ = initialize_nodes(rgb20, depth20, K, 45, (112,180,80), (130,255,255))
    print("direct_nodes", np.linalg.norm(np.diff(direct_nodes,axis=0),axis=1).sum(), "max", np.linalg.norm(np.diff(direct_nodes,axis=0),axis=1).max())
    tracker.initialize(rgb20, depth20)
    mask20 = segment_hsv(rgb20, (112, 180, 80), (130, 255, 255))
    px20 = ordered_skeleton_pixels(mask20)
    dm20 = depth_to_meters(depth20)
    raw = []
    for row, col in px20:
        v = float(dm20[row, col])
        raw.append(((col-K[0,2])*v/K[0,0], (row-K[1,2])*v/K[1,1], v))
    raw = np.asarray(raw); raw_len=np.linalg.norm(np.diff(raw,axis=0),axis=1).sum()
    sp,_=splprep(raw.T,s=0.0005,k=min(3,len(raw)-1)); sd=np.column_stack(splev(np.linspace(0,1,max(300,len(raw))),sp)); spl_len=np.linalg.norm(np.diff(sd,axis=0),axis=1).sum()
    print("raw skeleton",len(raw),"len",raw_len,"spline_len",spl_len,"raw_max",np.linalg.norm(np.diff(raw,axis=0),axis=1).max(),"spline_max",np.linalg.norm(np.diff(sd,axis=0),axis=1).max())
    print("reference_length", tracker._reference_length, "init_node_length", np.linalg.norm(np.diff(tracker.nodes, axis=0),axis=1).sum(), "max_step", np.linalg.norm(np.diff(tracker.nodes, axis=0),axis=1).max())
    p_paths = _current_skeleton_centerline_paths(rgb80, depth80, K)
    w_paths = _current_skeleton_centerline_paths(rgbw80, depthw80, Kw)
    p20_paths = _current_skeleton_centerline_paths(rgb20, depth20, K)
    # Render the wrist depth at initialization frame for orientation memory.
    rgbw20 = frames[20][2]
    depthw20 = frames[20][3]
    w20_paths = _current_skeleton_centerline_paths(rgbw20, depthw20, Kw)
    world_from_camera = world_from_camera_optical(env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id])
    world_from_wrist = world_from_camera_optical(env.data.cam_xpos[wrist_id], env.data.cam_xmat[wrist_id])
    T = np.linalg.inv(world_from_camera) @ world_from_wrist
    init_paths = list(p20_paths) + [transform_points(T, path) for path in w20_paths]
    init_paths.sort(key=lambda value: float(np.linalg.norm(np.diff(value,axis=0),axis=1).sum()), reverse=True)
    if init_paths:
        tracker._current_path_endpoints = (init_paths[0][0].copy(), init_paths[0][-1].copy())
    print("init path lengths", [float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) for p in init_paths], "memory", tracker._current_path_endpoints)
    paths = list(p_paths) + [transform_points(T, path) for path in w_paths]
    print("primary path lengths", [float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) for p in p_paths])
    print("wrist path lengths transformed", [float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) for p in [transform_points(T, path) for path in w_paths]])
    paths.sort(key=lambda value: float(np.linalg.norm(np.diff(value,axis=0),axis=1).sum()), reverse=True)
    truth80 = transform_points(np.linalg.inv(world_from_camera), env.data.xpos[env.cable_ids].copy())
    hand_world = np.asarray(env.hand_position, dtype=np.float64)
    hand_camera = transform_points(np.linalg.inv(world_from_camera), hand_world.reshape(1,3))[0]
    print("hand_camera",hand_camera,"endpoint_dist",np.linalg.norm(truth80[[0,-1]]-hand_camera,axis=1))
    finger_world = 0.5 * (env.data.xpos[env.left_finger_id] + env.data.xpos[env.right_finger_id])
    finger_camera = transform_points(np.linalg.inv(world_from_camera), finger_world.reshape(1,3))[0]
    print("finger_camera",finger_camera,"endpoint_dist",np.linalg.norm(truth80[[0,-1]]-finger_camera,axis=1))
    from trackdlo_standalone.metrics import frame_metrics
    for index, path in enumerate(paths):
        length = float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        oriented = path
        if tracker._current_path_endpoints is not None:
            ms, me = tracker._current_path_endpoints
            if np.linalg.norm(path[-1]-ms)+np.linalg.norm(path[0]-me) < np.linalg.norm(path[0]-ms)+np.linalg.norm(path[-1]-me):
                oriented = path[::-1]
        obs = tracker._current_path_observations(oriented, tracker.nodes, list(range(45)), orientation_hint=1)
        trial = tracker.nodes.copy()
        for node_index, observation in obs.items():
            trial[node_index] = observation
        metric = frame_metrics(trial, truth80)
        visible_err = float(np.mean(np.linalg.norm(trial[list(obs)] - truth80[list(obs)], axis=1))) if obs else float("nan")
        path_steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        direct_ep = np.linalg.norm(path[0]-tracker.nodes[0]) + np.linalg.norm(path[-1]-tracker.nodes[-1])
        reverse_ep = np.linalg.norm(path[-1]-tracker.nodes[0]) + np.linalg.norm(path[0]-tracker.nodes[-1])
        print(index, "n",len(path),"length",length,"step",np.median(path_steps),np.max(path_steps),"ep",direct_ep,reverse_ep,"debug",tracker._last_current_path_debug,"obs",len(obs),"trial_ordered_cm",metric["ordered_error_m"]*100,"trial_endpoint_cm",metric["endpoint_error_m"]*100,"obs_err_cm",visible_err*100)
    global_cap.release(); wrist_cap.release(); trajectory.close(); renderer.close(); env.close()


if __name__ == "__main__":
    main()
