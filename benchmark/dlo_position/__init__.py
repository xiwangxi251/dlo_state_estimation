"""DLO benchmark helpers.

The visual PPO adapter only needs the lightweight geometry helpers.  Keep the
optional image-processing estimators lazy so the adapter can run on jump151
without requiring scikit-image just to import the package.
"""

__all__ = []
try:  # optional dependency (scikit-image)
    from .estimator import DLOPositionEstimator, PositionEstimate
    __all__ += ["DLOPositionEstimator", "PositionEstimate"]
except ModuleNotFoundError:
    pass
try:  # optional estimator dependencies
    from .temporal_tracker import TemporalDLOTracker, TemporalEstimate
    __all__ += ["TemporalDLOTracker", "TemporalEstimate"]
except ModuleNotFoundError:
    pass
try:
    from .upe_tracker import UPETrackTracker
    __all__.append("UPETrackTracker")
except ModuleNotFoundError:
    pass
