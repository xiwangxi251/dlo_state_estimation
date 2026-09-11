from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trackdlo_standalone import TrackDLOConfig  # noqa: E402
from trackdlo_standalone.offline import run_sequence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ROS-free TrackDLO on one exported RGB-D sequence")
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--init-seconds", type=float, default=0.8)
    parser.add_argument(
        "--no-reinitialize",
        action="store_true",
        help="Disable RGB-D recovery and measure uninterrupted TrackDLO only",
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = ROOT / "results" / args.sequence.parent.parent.name / args.sequence.parent.name / args.sequence.name
    config = TrackDLOConfig(reinitialize_after_failures=0 if args.no_reinitialize else 3)
    summary = run_sequence(
        args.sequence,
        output,
        config=config,
        max_frames=args.max_frames,
        init_seconds=args.init_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
