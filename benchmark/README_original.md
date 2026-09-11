# DLO RGB-D Position Benchmark

This standalone benchmark estimates 14 uniformly spaced 3-D DLO positions from
one RGB-D frame. It deliberately lives outside `panda_cable_grasp`. The
estimator itself has no simulator/RL dependency; only the recorded-data
evaluation script imports the project environment to reconstruct RGB-D from
saved MuJoCo states.

The baseline follows the common centerline pipeline used by FASTDLO/mBEST-style
DLO perception:

1. HSV segmentation of the high-saturation blue cable;
2. binary-mask skeletonization;
3. ordered skeleton traversal, using minimum bending at projected crossings;
4. RGB-depth back-projection into camera coordinates;
5. cubic-spline smoothing and uniform 3-D arc-length resampling.

The simulated cable radius is 14 mm. The low-level estimator's `ray` mode
shifts the back-projected visible surface by a calibrated effective 20 mm
along the viewing ray so that the reported state represents the cable
centerline. The temporal evaluator defaults to the validated `adaptive` mode,
which blends a local tangent-normal correction only during non-rigid motion.
Override the offset with `--cable-radius-m` for another cable, or use zero for
raw surface points.

For the occlusion/shape-motion experiment, the `adaptive` correction
uses a local tangent-normal direction only when the consecutive visible curve
has a non-rigid Kabsch residual above a threshold; otherwise it keeps the ray
correction. The normal displacement is blended 50% with the ray displacement.
The following command makes all validated parameters explicit for
reproducibility:

```powershell
& $python run_temporal_benchmark.py `
  --cameras opst --frame-stride 5 --episodes-per-scenario 50 `
  --surface-to-center-mode adaptive `
  --adaptive-normal-residual-m 0.006 `
  --adaptive-normal-weight 0.50 `
  --sequence-disagreement-persistence 1 `
  --low-coverage-deformation-gain 0.50 `
  --low-coverage-deformation-residual-m 0.012 `
  --low-coverage-deformation-min-centroid-step-m 0.020 `
  --low-coverage-deformation-persistence 2 `
  --low-coverage-deformation-rigid-gain 0.50 `
  --cable-radius-m 0.020 `
  --output results\temporal_all_opst_deform_rigid050_p1_stride5
```

The default HSV range is `(100, 180, 150)` to `(135, 255, 255)`. The higher
saturation/value lower bounds are required by the current opposite camera,
whose blue sky otherwise forms a much larger component than the cable.

Only positions are estimated. Simulation ground truth is loaded by the
benchmark script after inference and is never passed to the estimator.

The authoritative results use the current project's 480x360
`dynamicvla_opst_camera` and `dynamicvla_wrist_camera`, reconstructed by
replaying the saved MuJoCo full states. The older archived `global_camera`
results are retained only as a legacy baseline. See
[`CAMERA_AND_SCENARIOS.md`](CAMERA_AND_SCENARIOS.md) for the camera audit and
motion-scenario results.

## Environment

The existing `dynamic` environment contains all required packages:

```powershell
$python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe"
& $python -m pip install -r requirements.txt
```

## Run

```powershell
$python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe"
& $python run_benchmark.py `
  --data-root "C:\Users\27642\Desktop\dynamic_cable\trackdlo_standalone\data\offline_sequences" `
  --output results\full
```

For a quick smoke test:

```powershell
& $python run_benchmark.py --max-frames-per-sequence 10 --output results\smoke
```

Use `--start-frame 10` to match the archived TrackDLO evaluation protocol,
which begins after the 0.8 s simulator settling interval.

Outputs:

- `summary.json`: aggregate accuracy, timing and 50 Hz deadline result;
- `per_sequence.csv`: scenario-level results;
- `per_frame.csv`: frame-level errors and stage timings;
- `overlays/`: RGB checks with predicted and ground-truth 14-point curves.

The algorithm-only latency excludes PNG file decoding, because a live control
loop receives RGB and depth arrays from the camera driver. Disk decoding latency
is reported separately.

## Current project camera benchmark

`run_recorded_benchmark.py` replays the saved MuJoCo full states from
`linux_log/expert_grasp_fix_4x50/run_20260824_113325`, reconstructs aligned RGB-D
from the current `dynamicvla_opst_camera` and `dynamicvla_wrist_camera`, and
evaluates against DLO body positions read only after inference:

```powershell
$python = "C:\ProgramData\anaconda3\envs\dynamic\python.exe"
& $python run_recorded_benchmark.py `
  --episodes-per-scenario 1000 `
  --frame-stride 10 `
  --output results\expert_grasp_fix_4x50_all_episodes_stride10
```

The algorithm timing excludes MuJoCo replay/render time. Rendering exists only
to reconstruct the RGB-D sensor arrays that were not stored in the original
log; a live camera would supply those arrays directly.

## Occlusion-completion experiment

The first exploration route for the missing DLO state is implemented in
`dlo_position/temporal_tracker.py`. It keeps a complete 14-node state in world
coordinates and, for each partial RGB-D centerline, performs:

1. contiguous arc-length interval matching against the previous state;
2. constant-velocity/motion-coherent propagation into hidden nodes;
3. adaptive observation gain (low-coverage fragments are damped to prevent
   static-scene drift);
4. an inextensible-cable length regularizer (measured cable length is 0.78 m).
5. a two-level rigid-motion gate: tight residuals for ordinary motion, plus a
   conservative large-jump branch for sudden whole-cable motion.
6. a centroid-motion prior and motion-constrained arc matching: the consecutive
   visible centerline supplies a Kabsch motion cue, but it is used only after a
   10 mm prior-motion gate; the same transform transports the complete state.
7. a low-coverage identity-jump safeguard: if a centerline fit implies an
   implausible 1.4 m or larger translation while less than half the cable is
   visible, translate the last complete state by the measured fragment
   centroid displacement instead of accepting the ambiguous arc match; for
   displacements above 30 mm, half of that measured velocity is retained for
   the next hidden-state prediction.
8. a low-coverage deformation gate: after two consecutive non-rigid visible
   centerline fits, blend the fragment centroid motion with its local Kabsch
   transform to move hidden nodes instead of freezing the state.

Run the temporal evaluator with:

```powershell
& $python run_temporal_benchmark.py `
  --cameras opst `
  --episodes-per-scenario 1000 `
  --frame-stride 10 `
  --expected-length-m 0.78 `
  --cable-radius-m 0.020 `
  --length-regularization-gain 0.30 `
  --low-coverage-gain-scale 0.50 `
  --rigid-residual-threshold-m 0.025 `
  --rigid-min-prior-step-m 0.004 `
  --sequence-min-translation-m 0.01 `
  --sequence-motion-weight 2.0 `
  --sequence-motion-min-prior-step-m 0.010 `
  --sequence-disagreement-translation-m 1.4 `
  --sequence-disagreement-persistence 2 `
  --surface-to-center-mode ray `
  --low-coverage-deformation-gain 0.0 `
  --low-coverage-deformation-rigid-gain 0.0 `
  --sequence-disagreement-use-centroid-translation `
  --sequence-disagreement-low-coverage-only `
  --sequence-disagreement-velocity-gain 0.5 `
  --sequence-disagreement-velocity-min-centroid-step-m 0.03 `
  --output results\temporal_full_opst_final6
```

The formal run is in `results/temporal_full_opst_final6`. It covers 4,668
frames from 183 readable episodes. Unlike the single-frame baseline, the
temporal output is always a complete 14-point state after tracker
initialization; `completion_coverage`, `confidence`, visible-point error and
occluded-point error are reported separately. The overall mean full-state
error is 10.13 cm (median 9.54 cm, P95 22.29 cm), with 14.17 cm mean error on
occluded points. Per-scenario full-state means are 3.37 cm (static), 6.86 cm
(rigid), 13.08 cm (shape), and 13.60 cm (combined). Mean/P95 algorithm latency
is 8.57/10.68 ms, so the 50 Hz deadline is met on this camera.

The low-confidence tail is the current research target: confidence `<0.05`
contains 2,872 successful frames and has 12.58 cm mean full-state error,
whereas confidence `>=0.05` has 5.63 cm. This motivates the next phases:
multi-component segmentation, stronger temporal/topological tracking, and
multi-camera or learned completion for long occlusions. The optional
`--use-fragments` path is retained for research comparison; on the supplied
episodes it is not enabled by default because disconnected-component matching
was less stable than the largest-component temporal baseline.

An overlay video for one episode per scenario is available at
`results/visualization/temporal_opst_final6_all_scenarios.mp4`. Green is ground truth,
blue is the visible RGB-D centerline, red is the completed 14-point state, and
magenta marks predicted hidden nodes.

The adaptive-correction visualization is available at
`results/visualization/temporal_opst_adaptive_residual006_blend050_all_scenarios.mp4`.

The earlier two-camera audit remains useful for camera selection: the fixed
opposite camera is the appropriate whole-DLO view, while the wrist camera often
sees only a local segment and has a much higher rendering/processing cost. The
temporal result above is therefore reported for `opst`; the current code does
not silently mix the wrist view into the 50 Hz claim.

### Topology/occlusion exploration (2026-09-01)

The first executable experiment for the proposed topology-aware route is now in
`dlo_position/estimator.py` and `dlo_position/temporal_tracker.py`:

- `--use-crossing-hypotheses` exposes all feasible Euler-trail candidates (up to
  128), then selects the candidate with the lowest 3-D continuity cost to the
  previous state; the default keeps the original minimum-bending path because
  the candidate branch is rarely ambiguous in this image resolution;
- `--use-fragments` keeps all sufficiently large visible components and updates
  directly observed nodes while completing only the missing arc intervals;
- the benchmark now reports projected self-crossing rate, crossing/non-crossing
  error, order-inversion fraction, and candidate count in both `summary.json`
  and `per_camera_scenario.csv`.

On the supplied run (`10` episodes per scenario, every fifth frame, 1,871
attempted / 1,832 successful frames), the default minimum-bending path achieves
8.41 cm mean full-state error, 8.34 cm on image-visible nodes, 10.24 cm on
image-occluded nodes, and 11.25 ms P95 algorithm latency. 17.2% of frames are
projected self-crossings: their mean error is 12.06 cm versus 7.66 cm on
non-crossings, and the order-inversion fraction has a 0.62 P95. This confirms
that ordering/topology, rather than raw point depth, is the dominant remaining
failure mode. Enabling `--use-crossing-hypotheses` gives 8.51 cm overall and
11.85 ms P95, so it remains an ablation rather than the default.

The all-fragments ablation with non-overlap interval assignment gives 8.54 cm
mean, 10.10 cm image-occluded error and 13.76 ms P95. It slightly helps hidden
points but is worse overall and is not the default. The two-camera fusion now
uses cross-view mask reprojection to resolve conflicting nodes; it gives 7.58 cm
on 102 frames, but 112.17 ms P95 because this replay includes serial MuJoCo
rendering, so it is an accuracy diagnostic, not a 50 Hz control-loop
implementation.

Reproducible outputs from this experiment:

- `results/temporal_sample10_single_hard_metrics/`
- `results/temporal_sample10_fragments_hard_metrics/`
- `results/temporal_fusion_improved_rayradius/`
- `results/temporal_sample10_default_final/`
- `results/temporal_sample10_crossing_hypotheses_ablation/`
- `results/temporal_sample10_fragments_joint/`
- `results/temporal_fusion_crossview_default/`
- `results/temporal_sample10_robot_mask2/`
- `results/temporal_sample10_robot_mask_multi/`
- `results/temporal_fusion_joint_imagecue_10ep_metrics/`
- `results/temporal_fusion_joint_imagecue_10ep_final/`
- `results/temporal_fusion_joint_utilization_10ep/`
- `results/temporal_opst_only_jointparams_10ep/`
- `results/temporal_wrist_only_jointparams_10ep/`
- `results/wrist_utilization_ablation_report.md`
- `results/robot_mask_experiment_report.md`
- `results/visualization/temporal_opst_improved_all_scenarios.mp4`

The fusion evaluator now has a basic cross-view reprojection-consistency gate
for conflicting nodes. A remaining next step is a genuine joint
multi-hypothesis arc assignment; the projected robot-mask branch is evaluated
below. A single-view image cannot uniquely resolve two cable branches that
overlap at the same depth; a temporal prior, complementary view, or physical
marker is required for those cases.

### Projected robot-mask experiment (2026-09-01)

The temporal benchmark now accepts `--use-robot-occlusion-mask`. It renders a
MuJoCo robot-geometry segmentation mask, reprojects the current predicted 14
nodes, and removes robot-covered nodes from the measurement update. Thus a
visible fragment that is split by the gripper is not interpolated through the
gripper as if it were continuously observed. The mask is used only as a
controlled simulation experiment; a real loop should generate the same mask
from the calibrated robot pose and camera model.

On the supplied run (four scenarios, ten episodes each, every fifth frame),
the mask changes the opposite-camera result from 8.41 cm to 8.28 cm mean error
and from 8.34 cm to 8.13 cm on image-visible nodes. Per-scene mean errors are
11.39 cm (combined), 5.27 cm (rigid), 10.28 cm (shape), and 3.60 cm (static).
The segmentation mask costs 5.60 ms mean; the end-to-end algorithm is 15.42 ms
mean / 17.70 ms P95, with 99.29% of frames below 20 ms. The gain is modest
because the projected robot mask covers only 0.58% of state nodes on average;
crossing/branch ambiguity remains the dominant error source.

The combined `--use-robot-occlusion-mask --use-multi-hypothesis` ablation gives
8.33 cm mean and 18.26 ms P95, so it is kept as an optional diagnostic rather
than enabled by default. Outputs are in
`results/temporal_sample10_robot_mask2/` and
`results/temporal_sample10_robot_mask_multi/`.

### Shared two-view arc-fusion experiment (2026-09-01)

`run_temporal_fusion_benchmark.py --fusion-mode joint` keeps one world-space
state and jointly matches the `opst` and `wrist` observations. The wrist match
uses the predicted state projection and its own cable mask as an identity cue;
the two views are allowed to overlap and are fused by residual-weighted node
measurements. On the same ten-episode/four-scenario split, mean error drops
from 8.66 cm for the pointwise two-track baseline to **8.12 cm**. Image-
occluded error drops from 11.05 cm to **9.21 cm**; per-scene means are
11.09 cm (combined), 6.13 cm (rigid), 10.04 cm (shape), and 2.83 cm (static).

The replay's serial MuJoCo RGB-D rendering is not a real-camera control-loop
cost: it has 24.1 ms mean by itself and makes the end-to-end P95 93.4 ms. With
images already acquired in parallel, the measured segmentation + skeleton +
depth + joint-tracker stages are 12.0 ms mean / 15.2 ms P95, which fits a 50 Hz
budget. The joint result is saved in
`results/temporal_fusion_joint_imagecue_10ep_metrics/`.

可视化回放（四个场景各 1 个 episode）保存在：

- `results/visualization/temporal_fusion_joint_opst_wrist_4scenes.mp4`：四场景完整双视角回放（204 帧，20 FPS）。
- `results/visualization/temporal_fusion_joint_hardcases.mp4`：shape/combined 交叉与遮挡难例（169 帧，20 FPS）。

视频标注约定：绿色=仿真真值中心线，蓝色=当前相机可见观测，红色=联合算法输出的 14 个状态点，紫色=该相机当前不可见、由时序/另一视角补全的点；左侧为固定 `opst` 相机，右侧为夹爪 `wrist` 相机。

For sizing the robot-mask work, `run_robot_occlusion_diagnostic.py` uses the
MuJoCo segmentation buffer as an evaluation oracle (it is not fed to the
estimator). On the same 1,871 opposite-camera frames, the fraction of in-frame
GT nodes hidden by robot geometry is 13.1% (static), 11.2% (rigid), 4.0%
(shape), and 4.4% (combined); the corresponding frame rates with at least one
robot-occluded node are 45.5%, 43.9%, 26.4%, and 30.8%. The segmentation pass
costs about 6 ms, so it is suitable for offline diagnosis/oracle ablation but
must be replaced by a projected robot mask from the known robot state in a
real 50 Hz loop. Results are saved in
`results/robot_occlusion_oracle_opst_stride5.json`.

### Surface-centerline correction ablation

The validated adaptive candidate was evaluated on all readable episodes at
every fifth frame and paired against the same-frame final6 ray baseline. It
uses a 6 mm non-rigid residual threshold, 50% tangent-normal/ray blending, a
single-frame identity-jump gate, and a 50% low-coverage centroid/Kabsch motion
blend. The mean full-state error changes by -1.14 cm (combined), -1.22 cm
(rigid), -1.72 cm (shape), and -0.03 cm (static) on common frames. Its
aggregate over 9,270 attempted frames is 9.14 cm mean / 8.42 cm median /
20.05 cm P95, with 8.73 ms mean and 11.02 ms P95 algorithm time; 99.93% of
successful frames are below 20 ms. A 10-episode frame-by-frame replay gives
9.14 cm mean and 8.71 ms mean / 10.56 ms P95. The temporal evaluator now uses
this configuration by default; the published ray baseline remains available
 with `--surface-to-center-mode ray` and the explicit legacy tracker gains.

### Panda scripted policy driven by visual DLO state (2026-09-06)

`run_scripted_vision.py` renders the same `opst` + `wrist` camera rig, runs the
selected adaptive dual-view TrackDLO estimator, converts its 45-node state to
world coordinates, and feeds the estimated target position/velocity to
`DynamicCableGraspPolicy`. The policy still uses only physical hand/contact
feedback for grasp confirmation; simulator cable positions are not used for
control. TrackDLO updates every 10 Panda control steps (nominal 5 Hz) while the
Panda action loop remains 50 Hz. On one paired seed (`20280804`), the visual
state policy succeeded in 1/4 scenes (25%): static 1/1, rigid 0/1, shape 0/1,
combined 0/1. The same rule policy with simulator target state succeeded in
2/4 (50%). The main visual failure was `insufficient_visible_observations`
during shape/combined motion, causing the policy to hold its last estimate.
This is a preliminary closed-loop integration result, not a claim of final
policy performance. Outputs are saved under
`results/scripted_vision_adaptive_dual_seed20280804_recheck` and the script is
`run_scripted_vision.py`.

For a repeated evaluation, each scenario was run 20 times with seeds
`20280804`--`20280823` (80 episodes total) using the same visual-state
configuration. The success counts were static **20/20 (100%)**, rigid
**5/20 (25%)**, shape **9/20 (45%)**, and combined **10/20 (50%)**, for an
overall **44/80 (55%)**. The estimator was still updated nominally at 5 Hz;
the measured TrackDLO update time averaged 374 ms (static), 320 ms (rigid),
204 ms (shape), and 126 ms (combined), so the 50 Hz action loop is maintained
by holding the latest estimate between vision updates rather than running
TrackDLO at 50 Hz. The largest visual failure mode was
`insufficient_visible_observations` (especially in shape/combined motion),
while dynamic-policy failures also included motion-boundary and timeout/slip
outcomes. Machine-readable outputs are saved under
`results/scripted_vision_adaptive_dual_20x20260906` (`summary.json` and
`per_episode.csv`).

As a paired control reference, the original scripted policy was also run with
the simulator DLO state (no visual estimator) on the same 80 seed/scenario
cells. It achieved 20/20 static, 17/20 rigid, 17/20 shape, and 15/20 combined
(69/80, **86.25%** overall). Thus the rigid visual result is not explained by
the rigid motion itself: visual state injection loses 12 of the 17 rigid
oracle successes. In the visual run, 7/20 rigid episodes never formed a
bilateral grasp; among the 13 that did, 6 hit the motion boundary, 1 timed out,
and 1 slipped during lift before task qualification. The oracle had only one
rigid episode without a bilateral candidate. The paired oracle outputs are saved under
`results/scripted_oracle_20x20260906/run_20260906_093127`.

## Method references

- Choi et al., *mBEST: Realtime Deformable Linear Object Detection Through
  Minimal Bending Energy Skeleton Pixel Traversals*, IEEE RA-L 2023,
  DOI `10.1109/LRA.2023.3290419`.
- The binary-mask, skeleton and ordered-centerline representation is also the
  standard interface used by FASTDLO-style DLO detectors. This benchmark keeps
  segmentation deliberately simple so that geometry accuracy and runtime can
  be measured before adding a learned mask model.
