"""
Surrogate likelihood builder for Analysis Pack Standard.

Trains and calibrates emulators for expensive or unavailable likelihoods:
- Gaussian Process surrogates
- Neural network emulators
- Parametric approximations

All surrogates include calibration and error propagation.
"""

from .builder import (
    SurrogateBuilder,
    GPSurrogate,
    NeuralSurrogate,
    ParametricSurrogate
)
from .calibration import (
    calibrate_surrogate,
    compute_calibration_curve,
    inflate_intervals
)

__all__ = [
    "SurrogateBuilder",
    "GPSurrogate",
    "NeuralSurrogate",
    "ParametricSurrogate",
    "calibrate_surrogate",
    "compute_calibration_curve",
    "inflate_intervals"
]
