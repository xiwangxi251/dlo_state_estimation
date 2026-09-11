from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dlo_position.fusion_benchmark import run_temporal_fusion_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    os.environ.get("DLO_RUN_ROOT", REPO_ROOT / "data" / "recorded_run")
)
DEFAULT_PROJECT_SRC = Path(
    os.environ.get("PANDA_CABLE_GRASP_SRC", REPO_ROOT.parent / "panda_cable_grasp" / "src")
)
DEFAULT_SCENARIOS = [
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark opposite+wrist temporal DLO fusion.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--cable-radius-m", type=float, default=0.014)
    parser.add_argument("--expected-length-m", type=float, default=0.78)
    parser.add_argument("--min-initial-length-ratio", type=float, default=0.80)
    parser.add_argument("--observation-gain", type=float, default=1.0)
    parser.add_argument("--velocity-gain", type=float, default=0.15)
    parser.add_argument("--velocity-decay", type=float, default=0.80)
    parser.add_argument("--length-regularization-gain", type=float, default=0.30)
    parser.add_argument("--low-coverage-gain-scale", type=float, default=0.50)
    parser.add_argument("--rigid-transform-gain", type=float, default=0.0)
    parser.add_argument(
        "--fusion-mode",
        choices=["pointwise", "joint"],
        default="pointwise",
        help="pointwise keeps two independent tracks; joint updates one shared arc state",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_temporal_fusion_benchmark(
        run_root=args.run_root.resolve(),
        project_src=args.project_src.resolve(),
        output=args.output.resolve(),
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        sample_count=args.samples,
        cable_radius_m=args.cable_radius_m,
        expected_length_m=args.expected_length_m,
        min_initial_length_ratio=args.min_initial_length_ratio,
        observation_gain=args.observation_gain,
        velocity_gain=args.velocity_gain,
        velocity_decay=args.velocity_decay,
        length_regularization_gain=args.length_regularization_gain,
        low_coverage_gain_scale=args.low_coverage_gain_scale,
        rigid_transform_gain=args.rigid_transform_gain,
        fusion_mode=args.fusion_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
