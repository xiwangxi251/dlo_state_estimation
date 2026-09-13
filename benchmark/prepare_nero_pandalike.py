"""Adapt one or more NERO panda_like episodes to the TrackDLO benchmark layout.

The DynamicVLA conversion stores 25 Hz RGB videos and a compact action/hand
episode file.  DLO evaluation additionally needs the original FULLPHYSICS
trajectory so MuJoCo can replay each state and render aligned depth.  This
script joins the two representations without committing either dataset to Git.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


SCENARIOS = (
    "id_static",
    "id_rigid_l1_nominal",
    "id_shape_nominal_current",
    "id_combined_l1_nominal",
)


def select_indices(times: np.ndarray, target_fps: int = 25) -> np.ndarray:
    """Match the nearest source states selected by prepare_nero_dynamicvla."""

    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("state times must be a non-empty 1-D array")
    period = 1.0 / float(target_fps)
    targets = times[0] + np.arange(
        int(np.floor((times[-1] - times[0]) / period + 1e-8)) + 1,
        dtype=np.float64,
    ) * period
    right = np.clip(np.searchsorted(times, targets, side="left"), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(
        np.abs(times[left] - targets) <= np.abs(times[right] - targets),
        left,
        right,
    )
    return np.unique(selected.astype(np.int64))


def _find_source(source_root: Path, scenario: str, seed: int) -> Path:
    matches = sorted(source_root.rglob(f"{scenario}/seed_{seed}/trajectory.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No FULLPHYSICS trajectory for {scenario}/seed_{seed} under {source_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"More than one trajectory matched {scenario}/seed_{seed}: {matches}"
        )
    return matches[0]


def convert_episode(
    rendered_root: Path,
    source_root: Path,
    output_root: Path,
    scenario: str,
    seed: int,
) -> Path:
    rendered = rendered_root / scenario / "episodes" / f"seed_{seed}"
    source = _find_source(source_root, scenario, seed)
    for name in ("opst.mp4", "wrist.mp4", "episode.json"):
        if not (rendered / name).is_file():
            raise FileNotFoundError(f"Missing {rendered / name}")

    destination = output_root / "episodes" / "expert" / scenario / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered / "opst.mp4", destination / "global.mp4")
    shutil.copy2(rendered / "wrist.mp4", destination / "wrist.mp4")

    with np.load(source, allow_pickle=False) as recording:
        if "states" not in recording or "state_times" not in recording:
            raise ValueError(f"{source} is not a FULLPHYSICS trajectory")
        state_times = np.asarray(recording["state_times"], dtype=np.float64)
        # The rendered video was generated from state_times[:-1], one action
        # per frame, at 25 Hz.  Keep all original fields and add the indices
        # expected by run_trackdlo_current.py.
        indices = select_indices(state_times[:-1], 25)
        fields = {key: recording[key] for key in recording.files}
        fields["frame_state_indices"] = indices
        fields["frame_times"] = state_times[:-1][indices]
        fields["video_fps"] = np.asarray(25.0, dtype=np.float64)
        with (destination / "trajectory.npz").open("wb") as handle:
            np.savez_compressed(handle, **fields)

    metadata = json.loads((rendered / "episode.json").read_text(encoding="utf-8"))
    metadata["robot"] = "nero"
    metadata["target_fps"] = 25
    metadata["source_trajectory_local"] = str(source.resolve())
    metadata.setdefault("result", {})["requested_seed"] = int(seed)
    (destination / "episode.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def parse_selection(values: list[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    result: dict[str, int] = {}
    for value in values:
        try:
            scenario, seed_text = value.split("=", 1)
            result[scenario] = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"Selection must use SCENARIO=SEED, got {value!r}") from exc
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument(
        "--selection",
        nargs="+",
        default=None,
        metavar="SCENARIO=SEED",
        help="Explicit seed per scenario; otherwise the first sorted episode is used.",
    )
    args = parser.parse_args()
    rendered_root = args.rendered_root.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    selection = parse_selection(args.selection)
    converted = []
    for scenario in args.scenarios:
        if selection and scenario in selection:
            seed = selection[scenario]
        else:
            episodes = sorted(
                (rendered_root / scenario / "episodes").glob("seed_*")
            )
            if not episodes:
                raise FileNotFoundError(f"No rendered episodes for {scenario}")
            seed = int(episodes[0].name.removeprefix("seed_"))
        destination = convert_episode(
            rendered_root, source_root, output_root, scenario, seed
        )
        converted.append({"scenario": scenario, "seed": seed, "output": str(destination)})
    print(json.dumps(converted, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
