# Camera and scenario audit

## Current-camera replay

The authoritative benchmark replays the saved MuJoCo full states in
`linux_log/expert_grasp_fix_4x50/run_20260824_113325` and renders aligned RGB-D
from both cameras in the current `panda_cable_grasp` model:

- `dynamicvla_opst_camera`: fixed opposite view, 480 x 360, FOV 73.7398 degrees;
- `dynamicvla_wrist_camera`: hand-mounted view, 480 x 360, FOV 73.7398 degrees.

The run's `global.mp4` is the recorded `opst_cam` stream; it is not the archived
TrackDLO `global_camera`. A first-frame replay comparison against that MP4 gave
2.57 intensity levels mean absolute pixel error, with most of the residual due
to video compression. This verifies that state replay reconstructs the recorded
view closely enough to obtain the missing aligned depth.

The estimator receives only rendered RGB-D. Ground-truth cable body positions
are read after inference and used only for evaluation. Offline replay/render
time is reported separately and is not included in algorithm latency, because
a live RGB-D camera supplies those arrays directly.

## Dataset coverage

The all-episode run uses frame stride 10 and covers 183 readable episodes:

| Scenario | Episodes | Sampled frames per camera |
|---|---:|---:|
| `id_static` | 35 | 753 |
| `id_rigid_l1_nominal` | 50 | 1,105 |
| `id_shape_nominal_current` | 49 | 1,479 |
| `id_combined_l1_nominal` | 49 | 1,331 |
| total | 183 | 4,668 |

Two zero-byte trajectories are skipped:
`id_shape_nominal_current/seed_20280829` and
`id_combined_l1_nominal/seed_20280845`.

## Single-frame camera audit

| Camera | Detection | Whole target in image | Complete length | Mean point error | Complete-curve error | Mean / P95 latency | <=20 ms | 50 Hz P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| opposite (`opst`) | 99.98% | 89.42% | 28.04% | 11.08 cm | 3.71 cm | 7.29 / 10.12 ms | 99.96% | yes |
| wrist | 84.45% | 12.00% | 10.30% | 14.41 cm | 2.54 cm | 23.09 / 51.32 ms | 53.75% | no |

“Whole target in image” checks image bounds, not occlusion. “Complete length”
means estimated curve length is 75–125% of ground-truth curve length. The high
all-frame error and low complete-length rate show that a successful blue-mask
detection often captures only part of the DLO; detection success alone must not
be interpreted as state accuracy.

## Per-scenario result

The fixed opposite camera meets the 20 ms P95 deadline in every scenario. Its
mean errors are 11.46 cm (static), 12.69 cm (rigid), 9.84 cm (shape), and
10.89 cm (combined). The wrist view frequently observes only a local cable
segment, especially in rigid and combined motion; it is better suited to local
grasp refinement than to standalone whole-DLO state estimation.

Archived 640x480/FOV-45 `global_camera` results remain under other `results/`
subdirectories only as a legacy algorithm baseline.

## Temporal full-state completion (opst)

The exploration tracker keeps the full 14-point state through partial
occlusion. The table below is from the final all-episode run with 0.78 m cable
length regularization, adaptive low-coverage updates, a calibrated 20 mm
effective RGB-D centerline offset, a centroid-motion prior, and
motion-constrained consecutive-visible-centerline rigid transport. It also
uses a two-frame low-coverage identity-jump safeguard: an implausible
sequence translation (at least 1.4 m) is replaced by the observed fragment's
centroid displacement; half of that displacement is retained as the next
hidden-state velocity when the centroid moves at least 30 mm. Errors are
computed against
all 14 ground-truth nodes; the two error columns split visible and occluded
nodes.

| Scenario | Frames | Full-state success | Mean full error | Visible error | Occluded error | Mean / P95 algorithm time |
|---|---:|---:|---:|---:|---:|---:|
| `id_static` | 753 | 100.00% | 3.37 cm | 3.04 cm | 5.26 cm | 8.18 / 9.23 ms |
| `id_rigid_l1_nominal` | 1,105 | 92.31% | 6.86 cm | 5.70 cm | 9.10 cm | 8.17 / 9.31 ms |
| `id_shape_nominal_current` | 1,479 | 99.93% | 13.08 cm | 10.79 cm | 17.91 cm | 8.93 / 11.41 ms |
| `id_combined_l1_nominal` | 1,331 | 88.43% | 13.60 cm | 10.87 cm | 18.38 cm | 8.69 / 10.67 ms |

The visual check is saved as
`results/visualization/temporal_opst_final6_all_scenarios.mp4`: green is ground truth,
blue is the visible centerline, red is the complete state, and magenta marks
the hidden predicted nodes. The all-episode aggregate is 10.13 cm mean full
state error (14.17 cm on occluded nodes), with 8.57 ms mean / 10.68 ms P95
algorithm time and 99.95% of successful frames under 20 ms. Low-confidence
frames remain the dominant accuracy failure mode, not the 50 Hz computation
budget.

## Surface-to-centerline direction ablation

The published final6 result remains the explicit `ray` baseline. The temporal
evaluator now defaults to the validated `adaptive` mode: it computes a
tangent-normal correction and blends it 50% with the ray correction only when
the consecutive visible centerlines have a non-rigid Kabsch residual above
6 mm; fragmented masks and low-residual static/rigid motion stay on the ray
correction. Low-coverage motion uses a 50% centroid/Kabsch blend, and the
identity-jump gate triggers on a single implausible fit.

The all-episode stride-5 replay is in
`results/temporal_all_opst_deform_rigid050_p1_stride5`:

| Scenario | Frames | Full-state success | Mean full error | Visible error | Occluded error | Mean / P95 algorithm time |
|---|---:|---:|---:|---:|---:|---:|
| `id_static` | 1,491 | 100.00% | 3.39 cm | 2.88 cm | 5.65 cm | 8.22 / 9.18 ms |
| `id_rigid_l1_nominal` | 2,191 | 92.74% | 5.65 cm | 4.31 cm | 8.01 cm | 8.29 / 9.41 ms |
| `id_shape_nominal_current` | 2,948 | 99.83% | 11.43 cm | 9.95 cm | 15.41 cm | 9.15 / 11.88 ms |
| `id_combined_l1_nominal` | 2,640 | 96.36% | 12.65 cm | 11.13 cm | 15.87 cm | 8.91 / 11.01 ms |

Across 9,270 attempted frames the candidate is 9.14 cm mean, 20.05 cm P95,
and 99.93% of successful frames are below 20 ms. Paired with the same frames
from the final6 ray run, the mean error changes are -1.14, -1.22, -1.72, and
-0.03 cm for combined, rigid, shape, and static respectively. These settings
are now the temporal evaluator defaults; the published ray baseline remains
available with explicit `--surface-to-center-mode ray` and legacy tracker gains.
