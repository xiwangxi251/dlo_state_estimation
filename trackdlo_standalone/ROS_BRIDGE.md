# Legacy TrackDLO ROS bridge

This is the earlier ROS replay path retained for upstream cross-checks. The
project's normal integration is now the ROS-free module described in
`README.md`; none of the commands below are required to run it.

This bridge replays saved MuJoCo full-physics states through the fixed global
camera and supplies TrackDLO with synchronized, pixel-aligned RGB and depth. It
does not require a physical depth camera for simulation experiments.

## Data layout

`legacy_offline_bridge/export_trackdlo_offline.py` creates one directory per source trial:

```text
data/offline_sequences/<source_run>/<scenario>/<trial>/
  rgb/000000.png
  depth/000000.png
  timestamps.npy
  camera.json
  manifest.json
  ground_truth_evaluation_only.npz
  preview_alignment.mp4
```

Depth PNGs are aligned `uint16` millimetres, matching the format expected by
the official TrackDLO node. The ground-truth file is deliberately named
`evaluation_only`: neither the publisher nor TrackDLO reads it.

The checked dataset contains three trials from each of `id_static`,
`id_rigid_l1_nominal`, `id_shape_nominal_current`, and
`id_combined_l1_nominal`. It uses 640x480 at 12.5 Hz (every second frame from
the 25 Hz recordings), close to TrackDLO's published 15 Hz setup.

## Export or extend the dataset on Windows

Run from the standalone `trackdlo_standalone` directory in the `dynamic` environment:

```powershell
python legacy_offline_bridge/export_trackdlo_offline.py `
  headless_videos/<run>/<scenario>/trial_001_states.npz `
  --output-root data/offline_sequences `
  --stride 2 --preview
```

Only recordings whose model snapshot contains `global_camera` can be exported.
Use `--overwrite` intentionally when replacing an existing sequence.

## Run the official TrackDLO package on the Linux/ROS Noetic machine

TrackDLO is a ROS1 package tested by its authors on Ubuntu 20.04 and ROS
Noetic. Copy this repository (including the ignored dataset artifact) to that
machine. Build the official repository in a catkin workspace first, then source
both ROS and that workspace in every terminal.

Terminal 1:

```bash
roscore
```

Terminal 2 (official tracker):

```bash
source /opt/ros/noetic/setup.bash
source ~/trackdlo_ws/devel/setup.bash
roslaunch trackdlo trackdlo.launch \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/aligned_depth_to_color/camera_info \
  hsv_threshold_lower_limit:="90 90 80" \
  hsv_threshold_upper_limit:="130 255 255" \
  visualize_initialization_process:=false
```

Terminal 3 (record official outputs):

```bash
source /opt/ros/noetic/setup.bash
python3 legacy_offline_bridge/record_trackdlo_results.py \
  results/official_ros/id_static_trial_001
```

Terminal 4 (publish one sequence):

```bash
source /opt/ros/noetic/setup.bash
python3 legacy_offline_bridge/publish_trackdlo_offline.py \
  data/offline_sequences/<source_run>/id_static/trial_001
```

The publisher uses the official topic names, sends identical timestamps for
RGB/depth/CameraInfo, and publishes `16UC1` depth. The recorder works around an
official TrackDLO detail: `results_pc` is timestamped but `results_img` has an
empty header.

The hue and saturation limits remain the official blue-rope defaults. The
value lower bound is raised from 30 to 80 for this simulation: 30 also selects
the dark blue background, while 80 keeps the cable centerline recall at 98.38%
on the first static sequence and removes that large false-positive region. You
can reproduce the visual check before launching ROS:

```bash
python3 legacy_offline_bridge/visualize_trackdlo_hsv.py \
  data/offline_sequences/<source_run>/id_static/trial_001 \
  --lower "90 90 80"
```

## Generate the accuracy visualization

```bash
python3 legacy_offline_bridge/visualize_trackdlo_results.py \
  data/offline_sequences/<source_run>/id_static/trial_001 \
  results/official_ros/id_static_trial_001
```

Outputs:

- `trackdlo_accuracy.mp4`: left side overlays TrackDLO (red) and simulation
  ground truth (green); right side is the official TrackDLO visualization.
- `trackdlo_accuracy.csv`: per-frame node count, ordered centerline error and
  symmetric Chamfer error.
- `trackdlo_accuracy.json`: mean and 95th-percentile errors.

Start with `id_static/trial_001`. If its mask is incomplete, tune only the two
HSV limits before judging the tracker. Then run rigid, shape and combined
sequences without changing parameters so their comparison remains fair.
