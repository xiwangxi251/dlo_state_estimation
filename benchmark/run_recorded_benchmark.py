from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlo_position.recorded_benchmark import run_recorded_benchmark


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
        description="Replay recorded expert states and benchmark current project cameras."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--project-src", type=Path, default=DEFAULT_PROJECT_SRC)
    parser.add_argument(
        "--output", type=Path, default=Path("results/expert_grasp_fix_4x50")
    )
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument(
        "--cameras", nargs="+", choices=["opst", "wrist"], default=["opst", "wrist"]
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--cable-radius-m", type=float, default=0.014)
    parser.add_argument("--save-overlays-per-group", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_recorded_benchmark(
        run_root=args.run_root.resolve(),
        project_src=args.project_src.resolve(),
        output=args.output.resolve(),
        scenarios=args.scenarios,
        cameras=args.cameras,
        episodes_per_scenario=args.episodes_per_scenario,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        sample_count=args.samples,
        cable_radius_m=args.cable_radius_m,
        save_overlays_per_group=args.save_overlays_per_group,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

