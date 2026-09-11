from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlo_position.benchmark import discover_sequences, run_benchmark


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\27642\Desktop\dynamic_cable\trackdlo_standalone\data\offline_sequences"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 14-point RGB-D DLO position estimation."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("results/full"))
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--cable-radius-m",
        type=float,
        default=0.014,
        help="Known radius used to move measured surface depth to the DLO centerline",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--save-overlays", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = discover_sequences(args.data_root.resolve())
    if not sequences:
        raise SystemExit(f"No RGB-D/ground-truth sequences found under {args.data_root}")
    summary = run_benchmark(
        sequences,
        args.output.resolve(),
        sample_count=args.samples,
        start_frame=args.start_frame,
        cable_radius_m=args.cable_radius_m,
        stride=args.stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
        save_overlays=args.save_overlays,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
