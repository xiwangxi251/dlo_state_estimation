# ROS-free TrackDLO

This directory contains the archived, isolated TrackDLO integration originally
evaluated for `panda_cable_grasp`. It now lives outside that Git repository
because its dynamic-shape performance was insufficient. The tracking path does
**not** start ROS, subscribe to ROS
topics, or use simulator ground truth:

```text
RGB + aligned depth + camera intrinsics
  -> HSV cable pixels
  -> 3-D observed point cloud
  -> TrackDLO initialization / C++ tracking core
  -> ordered 3-D cable nodes
```

Simulation ground truth is opened only by the offline evaluator, after each
tracking call, to compute errors and draw the cyan reference curve.

## Directory layout

```text
trackdlo_standalone/
  cpp/                       ported TrackDLO C++ core and pybind11 wrapper
  src/trackdlo_standalone/   reusable Python package
  scripts/run_offline.py     offline evaluation entry point
  legacy_offline_bridge/     earlier HSV/ROS export and visualization tools
  tests/                     unit and regression tests
  data/                      RGB-D datasets (gitignored)
  results/                   metrics and visualizations (gitignored)
```

The C++ core is derived from the official MIT-licensed TrackDLO repository;
the upstream license is retained in `THIRD_PARTY_TRACKDLO_LICENSE.txt`. ROS,
PCL, and message-passing code are not part of the Python module. Their useful
pre/post-processing steps (visibility, self-occlusion and voxel sampling) are
implemented locally with NumPy/OpenCV.

## Build

Use the project's `dynamic` environment. Install the Python dependencies and
provide Eigen 3 headers:

```powershell
cd C:\Users\27642\Desktop\dynamic_cable\trackdlo_standalone
python -m pip install -r requirements.txt
$env:EIGEN3_INCLUDE_DIR = "C:\path\to\eigen-3.4.0"
python setup.py build_ext --inplace
```

On Ubuntu, set `EIGEN3_INCLUDE_DIR=/usr/include/eigen3` if Eigen is not found
automatically. A compiler with C++17 support is required.

## Python API

Images passed to the API are RGB arrays. Depth may be aligned `uint16`
millimetres or floating-point metres. `K` is the 3x3 RGB-camera intrinsic
matrix.

```python
from trackdlo_standalone import TrackDLOTracker

tracker = TrackDLOTracker(K)
first = tracker.initialize(rgb0, depth0)
result = tracker.update(rgb1, depth1)

nodes_xyz = result.nodes_camera       # (45, 3), camera optical coordinates
visible = result.visible_nodes
latency_ms = result.total_ms
was_reinitialized = result.reinitialized
```

The default HSV range targets the blue simulated cable. It is a sensor
parameter, so a different real cable/background requires calibration. The
configured projected cable width is 8 pixels for this 640x480 global camera;
the official demo value of 40 pixels corresponds to its much thicker rope in
the image and is not a universal algorithm constant.

## Offline evaluation and visualization

Run one exported sequence:

```powershell
python scripts/run_offline.py data/offline_sequences/<run>/<scenario>/trial_001
```

By default tracking starts at 0.8 s, after the simulator's settling phase. To
limit a smoke test:

```powershell
python scripts/run_offline.py data/offline_sequences/<run>/<scenario>/trial_001 --max-frames 20
```

The wrapper safely declines native updates with fewer than six matched guide
nodes. After three consecutive failed updates it attempts RGB-D
reinitialization, accepting it only when the recovered curve is 75%-125% of
the initial cable length. Every accepted recovery is marked in the CSV, NPZ,
video and summary. This prevents a robot-occluded fragment from being reported
as the full cable. To evaluate uninterrupted upstream tracking without this
project-level recovery, add `--no-reinitialize`.

Each result directory contains:

- `trackdlo_results.npz`: 3-D nodes, visibility, tracking/reinitialization
  status and source frame index;
- `metrics.csv`: per-frame errors, node visibility, point counts and timings;
- `summary.json`: mean/median/p95 error and latency plus frozen parameters;
- `trackdlo_overlay.mp4`: RGB projection and world-XY comparison side by side;
- `trackdlo_overlay_first.png`: quick visual check.

The principal `frame_error_m` is a symmetric point-to-piecewise-line distance,
which does not depend on how many points the simulator uses. Ordered and
endpoint errors are also reported and are invariant to reversal of cable node
order.

Summary rates distinguish initial setup, native C++ tracking updates,
successful RGB-D reinitializations, and failed frames that only retain the
last valid state. Error means over all frames are therefore not confused with
the `*_successful_mean` and `*_native_mean` subsets.

Projected self-crossings are handled during initialization by collapsing the
small junction blob and finding an edge-covering trail, rather than keeping
only the graph's longest branch. This recovers the full cable length without
using simulator state.

## Tests

```powershell
python -m unittest discover -s tests -v
```

`ROS_BRIDGE.md` documents the previous official-ROS replay route. It remains
available for cross-checking upstream behavior, but it is not needed by the
new module or by this project's offline experiments.
