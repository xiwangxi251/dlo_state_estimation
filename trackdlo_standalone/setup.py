from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup

try:
    import pybind11
except ImportError as exc:
    raise RuntimeError("Install pybind11 before building: python -m pip install pybind11") from exc


ROOT = Path(__file__).resolve().parent


def find_eigen() -> Path:
    candidates = [
        os.environ.get("EIGEN3_INCLUDE_DIR"),
        ROOT / "vendor" / "eigen",
        Path(sys.prefix) / "Library" / "include" / "eigen3",
        Path(sys.prefix) / "include" / "eigen3",
        Path("/usr/include/eigen3"),
        Path("/usr/local/include/eigen3"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "Eigen" / "Core").is_file():
            return Path(candidate)
    raise RuntimeError("Eigen3 headers not found; set EIGEN3_INCLUDE_DIR")


compile_args = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]

extension = Extension(
    "trackdlo_standalone._core",
    sources=[
        str(ROOT / "cpp" / "bindings.cpp"),
        str(ROOT / "cpp" / "core_utils.cpp"),
        str(ROOT / "cpp" / "trackdlo_core.cpp"),
    ],
    include_dirs=[pybind11.get_include(), str(find_eigen()), str(ROOT / "cpp")],
    language="c++",
    extra_compile_args=compile_args,
    # The extension has no Windows resources; suppressing the linker
    # manifest avoids an rc.exe dependency on machines where the SDK resource
    # compiler is not on the VC environment PATH.
    extra_link_args=["/MANIFEST:NO"] if os.name == "nt" else [],
)

setup(
    name="panda-trackdlo-standalone",
    version="0.1.0",
    description="ROS-free TrackDLO Python module for panda_cable_grasp",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=[extension],
    install_requires=["numpy", "opencv-python", "scipy", "scikit-image"],
    python_requires=">=3.10",
)
