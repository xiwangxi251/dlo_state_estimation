from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlo_position.temporal_benchmark import run_temporal_benchmark


DEFAULT_RUN_ROOT = Path(
    r"C:\Users\27642\Desktop\dynamic_cable\linux_log\expert_grasp_fix_4x50\run_20260824_113325"
)
DEFAULT_PROJECT_SRC = Path(
    r"C:\Users\27642\Desktop\dynamic_cable\panda_cable_grasp\src"
)
DEFAULT_SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark temporal completion of the full 14-point DLO state."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--cameras", nargs="+", choices=["opst", "wrist"], default=["opst", "wrist"]
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument(
        "--tracker-method",
        choices=["temporal", "upetrack"],
        default="temporal",
        help="state completion tracker; upetrack is a lightweight reproduction of the published UPE equations",
    )
    parser.add_argument(
        "--cable-radius-m",
        type=float,
        default=0.020,
        help="effective RGB-D surface-to-centerline correction in metres",
    )
    parser.add_argument(
        "--surface-to-center-mode",
        choices=["ray", "normal", "adaptive"],
        default="adaptive",
        help="surface-to-centerline correction direction",
    )
    parser.add_argument(
        "--adaptive-normal-residual-m",
        type=float,
        default=0.006,
        help="non-rigid residual threshold for adaptive surface correction",
    )
    parser.add_argument(
        "--adaptive-normal-weight",
        type=float,
        default=0.5,
        help="normal-vs-ray blend weight in adaptive mode",
    )
    parser.add_argument("--spline-smoothing", type=float, default=0.0005)
    parser.add_argument("--expected-length-m", type=float, default=0.78)
    parser.add_argument("--min-initial-length-ratio", type=float, default=0.80)
    parser.add_argument("--observation-gain", type=float, default=1.0)
    parser.add_argument("--velocity-gain", type=float, default=0.15)
    parser.add_argument("--velocity-decay", type=float, default=0.80)
    parser.add_argument("--length-regularization-gain", type=float, default=0.30)
    parser.add_argument("--low-coverage-gain-scale", type=float, default=0.50)
    parser.add_argument("--confidence-gain-scale", type=float, default=0.0)
    parser.add_argument("--confidence-gain-floor", type=float, default=0.25)
    parser.add_argument("--confidence-gain-reference", type=float, default=0.15)
    parser.add_argument(
        "--confidence-gain-low-coverage-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rigid-transform-gain", type=float, default=1.0)
    parser.add_argument("--rigid-residual-threshold-m", type=float, default=0.025)
    parser.add_argument("--rigid-min-prior-step-m", type=float, default=0.004)
    parser.add_argument("--rigid-max-translation-m", type=float, default=0.30)
    parser.add_argument("--rigid-max-angle-deg", type=float, default=45.0)
    parser.add_argument("--arc-continuity-weight", type=float, default=1.0)
    parser.add_argument(
        "--image-match-weight",
        type=float,
        default=0.0,
        help="projected-pixel identity cue for partial arc matching (metre-equivalent weight)",
    )
    parser.add_argument("--centroid-motion-weight", type=float, default=0.50)
    parser.add_argument("--centroid-motion-slack-m", type=float, default=0.15)
    parser.add_argument("--sequence-min-translation-m", type=float, default=0.004)
    parser.add_argument("--rigid-min-match-confidence", type=float, default=0.0)
    parser.add_argument("--sequence-motion-weight", type=float, default=2.0)
    parser.add_argument("--sequence-motion-min-prior-step-m", type=float, default=0.01)
    parser.add_argument("--sequence-disagreement-translation-m", type=float, default=1.4)
    parser.add_argument("--sequence-disagreement-slack-m", type=float, default=0.15)
    parser.add_argument("--sequence-disagreement-persistence", type=int, default=1)
    parser.add_argument(
        "--sequence-disagreement-hold-previous",
        action="store_true",
        help="hold the last complete state when a persistent sequence disagreement is active",
    )
    parser.add_argument(
        "--sequence-disagreement-use-centroid-translation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="translate the last complete state by the observed fragment centroid delta on disagreement",
    )
    parser.add_argument(
        "--sequence-disagreement-low-coverage-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply sequence-disagreement handling only when the observed length ratio is below 0.5",
    )
    parser.add_argument("--sequence-disagreement-velocity-gain", type=float, default=0.5)
    parser.add_argument(
        "--sequence-disagreement-velocity-min-centroid-step-m",
        type=float,
        default=0.03,
    )
    parser.add_argument("--low-coverage-deformation-gain", type=float, default=0.5)
    parser.add_argument(
        "--low-coverage-deformation-residual-m", type=float, default=0.012
    )
    parser.add_argument(
        "--low-coverage-deformation-min-centroid-step-m", type=float, default=0.02
    )
    parser.add_argument(
        "--low-coverage-deformation-persistence", type=int, default=2
    )
    parser.add_argument(
        "--low-coverage-deformation-rigid-gain", type=float, default=0.5
    )
    parser.add_argument(
        "--low-coverage-rigid-observation-gain",
        type=float,
        default=1.0,
        help="scale direct short-fragment fusion when the consecutive fit is rigid",
    )
    parser.add_argument("--hidden-motion-gain", type=float, default=0.70)
    parser.add_argument("--reacquisition-min-observed-ratio", type=float, default=1.0)
    parser.add_argument("--reacquisition-min-coverage", type=float, default=0.85)
    parser.add_argument(
        "--reacquisition-max-match-confidence", type=float, default=-1.0
    )
    parser.add_argument("--arc-hysteresis-jump", type=float, default=0.0)
    parser.add_argument("--arc-hysteresis-margin-m", type=float, default=0.0)
    parser.add_argument(
        "--allow-partial-initialization",
        action="store_true",
        help="initialize a full state from a short first visible fragment",
    )
    parser.add_argument(
        "--use-fragments",
        action="store_true",
        help="retain all segmented cable components and fuse them temporally",
    )
    parser.add_argument(
        "--use-crossing-hypotheses",
        action="store_true",
        help="enumerate multiple Euler traversals at skeleton crossings and select by temporal continuity",
    )
    parser.add_argument(
        "--use-multi-hypothesis",
        action="store_true",
        help="keep a small beam of competing arc-identity trackers during partial observations",
    )
    parser.add_argument("--hypothesis-beam-width", type=int, default=2)
    parser.add_argument("--hypothesis-hold-weight", type=float, default=20.0)
    parser.add_argument(
        "--use-robot-occlusion-mask",
        action="store_true",
        help="Use a projected MuJoCo robot-geometry mask to protect hidden arc nodes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_temporal_benchmark(
        run_root=args.run_root.resolve(),
        project_src=args.project_src.resolve(),
        output=args.output.resolve(),
        scenarios=args.scenarios,
        cameras=args.cameras,
        episodes_per_scenario=args.episodes_per_scenario,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        sample_count=args.samples,
        tracker_method=args.tracker_method,
        cable_radius_m=args.cable_radius_m,
        surface_to_center_mode=args.surface_to_center_mode,
        adaptive_normal_residual_m=args.adaptive_normal_residual_m,
        adaptive_normal_weight=args.adaptive_normal_weight,
        spline_smoothing=args.spline_smoothing,
        expected_length_m=args.expected_length_m,
        min_initial_length_ratio=args.min_initial_length_ratio,
        observation_gain=args.observation_gain,
        velocity_gain=args.velocity_gain,
        velocity_decay=args.velocity_decay,
        length_regularization_gain=args.length_regularization_gain,
        low_coverage_gain_scale=args.low_coverage_gain_scale,
        confidence_gain_scale=args.confidence_gain_scale,
        confidence_gain_floor=args.confidence_gain_floor,
        confidence_gain_reference=args.confidence_gain_reference,
        confidence_gain_low_coverage_only=args.confidence_gain_low_coverage_only,
        rigid_transform_gain=args.rigid_transform_gain,
        rigid_residual_threshold_m=args.rigid_residual_threshold_m,
        rigid_min_prior_step_m=args.rigid_min_prior_step_m,
        rigid_max_translation_m=args.rigid_max_translation_m,
        rigid_max_angle_deg=args.rigid_max_angle_deg,
        arc_continuity_weight=args.arc_continuity_weight,
        image_match_weight=args.image_match_weight,
        centroid_motion_weight=args.centroid_motion_weight,
        centroid_motion_slack_m=args.centroid_motion_slack_m,
        sequence_min_translation_m=args.sequence_min_translation_m,
        rigid_min_match_confidence=args.rigid_min_match_confidence,
        sequence_motion_weight=args.sequence_motion_weight,
        sequence_motion_min_prior_step_m=args.sequence_motion_min_prior_step_m,
        sequence_disagreement_translation_m=args.sequence_disagreement_translation_m,
        sequence_disagreement_slack_m=args.sequence_disagreement_slack_m,
        sequence_disagreement_persistence=args.sequence_disagreement_persistence,
        sequence_disagreement_hold_previous=args.sequence_disagreement_hold_previous,
        sequence_disagreement_use_centroid_translation=args.sequence_disagreement_use_centroid_translation,
        sequence_disagreement_low_coverage_only=args.sequence_disagreement_low_coverage_only,
        sequence_disagreement_velocity_gain=args.sequence_disagreement_velocity_gain,
        sequence_disagreement_velocity_min_centroid_step_m=args.sequence_disagreement_velocity_min_centroid_step_m,
        low_coverage_deformation_gain=args.low_coverage_deformation_gain,
        low_coverage_deformation_residual_m=args.low_coverage_deformation_residual_m,
        low_coverage_deformation_min_centroid_step_m=args.low_coverage_deformation_min_centroid_step_m,
        low_coverage_deformation_persistence=args.low_coverage_deformation_persistence,
        low_coverage_deformation_rigid_gain=args.low_coverage_deformation_rigid_gain,
        low_coverage_rigid_observation_gain=args.low_coverage_rigid_observation_gain,
        hidden_motion_gain=args.hidden_motion_gain,
        reacquisition_min_observed_ratio=args.reacquisition_min_observed_ratio,
        reacquisition_min_coverage=args.reacquisition_min_coverage,
        reacquisition_max_match_confidence=args.reacquisition_max_match_confidence,
        arc_hysteresis_jump=args.arc_hysteresis_jump,
        arc_hysteresis_margin_m=args.arc_hysteresis_margin_m,
        allow_partial_initialization=args.allow_partial_initialization,
        use_fragments=args.use_fragments,
        use_crossing_hypotheses=args.use_crossing_hypotheses,
        use_multi_hypothesis=args.use_multi_hypothesis,
        hypothesis_beam_width=args.hypothesis_beam_width,
        hypothesis_hold_weight=args.hypothesis_hold_weight,
        use_robot_occlusion_mask=args.use_robot_occlusion_mask,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
