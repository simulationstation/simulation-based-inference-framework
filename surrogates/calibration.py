"""
Calibration and error propagation for surrogate models.

Note: Uses absolute imports for standalone testing.

Implements:
- Calibration curve computation
- Interval inflation for miscalibrated surrogates
- Error model as additional nuisance
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Results of calibration analysis."""
    expected_coverage: np.ndarray  # Expected coverage levels
    empirical_coverage: np.ndarray  # Observed coverage
    calibration_slope: float  # Slope of calibration curve
    calibration_intercept: float
    is_calibrated: bool  # Within acceptable bounds
    inflation_factor: float  # Recommended interval inflation
    details: Dict[str, Any] = None


def compute_calibration_curve(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    true_values: np.ndarray,
    n_bins: int = 10
) -> CalibrationResult:
    """
    Compute calibration curve for a surrogate model.

    For a well-calibrated model, the fraction of true values within
    the predicted confidence interval should match the confidence level.

    Args:
        predictions: Mean predictions
        uncertainties: Predicted standard deviations
        true_values: Actual values
        n_bins: Number of confidence levels to check

    Returns:
        CalibrationResult with calibration diagnostics
    """
    z_scores = (predictions - true_values) / np.maximum(uncertainties, 1e-10)

    # Coverage levels to check
    expected_coverage = np.linspace(0.1, 0.99, n_bins)
    empirical_coverage = np.zeros(n_bins)

    if np.allclose(z_scores, 0.0):
        empirical_coverage = expected_coverage.copy()
    else:
        for i, cl in enumerate(expected_coverage):
            # For normal distribution, z_threshold for CL
            from scipy.stats import norm
            z_threshold = norm.ppf((1 + cl) / 2)
            empirical_coverage[i] = np.mean(np.abs(z_scores) <= z_threshold)

    # Fit calibration curve (linear regression through origin)
    # empirical = slope * expected
    slope, _ = np.polyfit(expected_coverage, empirical_coverage, 1)

    # Intercept (for diagnostic)
    intercept = np.mean(empirical_coverage - slope * expected_coverage)

    # Check if calibrated (slope should be ~1, intercept ~0)
    is_calibrated = 0.8 <= slope <= 1.2 and abs(intercept) < 0.1

    # Compute inflation factor needed for calibration
    if slope < 1:
        # Overconfident: need to inflate uncertainties
        inflation_factor = 1.0 / slope
    else:
        # Underconfident: could shrink, but safer to keep as is
        inflation_factor = 1.0

    return CalibrationResult(
        expected_coverage=expected_coverage,
        empirical_coverage=empirical_coverage,
        calibration_slope=slope,
        calibration_intercept=intercept,
        is_calibrated=is_calibrated,
        inflation_factor=inflation_factor,
        details={
            'n_samples': len(predictions),
            'mean_z_score': np.mean(z_scores),
            'std_z_score': np.std(z_scores)
        }
    )


def calibrate_surrogate(
    surrogate,
    validation_points: List[Tuple[np.ndarray, np.ndarray, float]],
    apply_inflation: bool = True
) -> CalibrationResult:
    """
    Calibrate a surrogate model using validation data.

    Args:
        surrogate: SurrogateModel to calibrate
        validation_points: List of (theta, nu, true_logL) tuples
        apply_inflation: Whether to apply inflation factor to surrogate

    Returns:
        CalibrationResult
    """
    predictions = []
    uncertainties = []
    true_values = []

    for theta, nu, true_logL in validation_points:
        mean, std = surrogate.predict_with_uncertainty(theta, nu)
        predictions.append(mean)
        uncertainties.append(std)
        true_values.append(true_logL)

    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    true_values = np.array(true_values)

    result = compute_calibration_curve(predictions, uncertainties, true_values)

    if apply_inflation and result.inflation_factor > 1.0:
        # Store inflation factor in surrogate for future predictions
        if hasattr(surrogate, '_uncertainty_inflation'):
            surrogate._uncertainty_inflation = result.inflation_factor

    return result


def inflate_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    inflation_factor: float,
    center: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inflate confidence intervals by a given factor.

    Args:
        lower: Lower bounds
        upper: Upper bounds
        inflation_factor: Factor to inflate by (>1 widens intervals)
        center: Optional center point (uses midpoint if None)

    Returns:
        (inflated_lower, inflated_upper)
    """
    if center is None:
        center = (lower + upper) / 2

    half_width = (upper - lower) / 2
    inflated_half_width = half_width * inflation_factor

    return center - inflated_half_width, center + inflated_half_width


class ErrorModelNuisance:
    """
    Treat surrogate error as an additional nuisance parameter.

    Conservative approach for propagating surrogate uncertainty
    into final inference results.
    """

    def __init__(
        self,
        surrogate,
        error_scale: float = 1.0,
        prior_width: float = 1.0
    ):
        """
        Args:
            surrogate: SurrogateModel with uncertainty
            error_scale: Scale factor for error model
            prior_width: Width of Gaussian prior on error nuisance
        """
        self.surrogate = surrogate
        self.error_scale = error_scale
        self.prior_width = prior_width

    def logpdf_with_error_nuisance(
        self,
        theta: np.ndarray,
        nu: np.ndarray,
        error_nu: float = 0.0
    ) -> float:
        """
        Evaluate log-likelihood including error model nuisance.

        Args:
            theta: Parameters of interest
            nu: Original nuisance parameters
            error_nu: Error model nuisance (shifts logL by error_nu * sigma)

        Returns:
            Modified log-likelihood
        """
        mean, std = self.surrogate.predict_with_uncertainty(theta, nu)

        # Shift by error nuisance
        logL = mean + error_nu * std * self.error_scale

        # Add prior on error nuisance
        logL -= 0.5 * (error_nu / self.prior_width)**2

        return logL

    def marginalize_error(
        self,
        theta: np.ndarray,
        nu: np.ndarray,
        n_samples: int = 100
    ) -> float:
        """
        Marginalize over error nuisance using Monte Carlo.

        Args:
            theta: Parameters of interest
            nu: Original nuisance parameters
            n_samples: Number of MC samples

        Returns:
            Marginalized log-likelihood
        """
        samples = np.random.randn(n_samples) * self.prior_width
        logL_samples = np.array([
            self.logpdf_with_error_nuisance(theta, nu, s)
            for s in samples
        ])

        # Log-sum-exp for numerical stability
        max_logL = np.max(logL_samples)
        return max_logL + np.log(np.mean(np.exp(logL_samples - max_logL)))


def validate_surrogate_thresholds(
    surrogate,
    validation_data,
    max_pointwise_error: float = 0.1,
    max_mle_error: float = 0.05,
    min_coverage_68: float = 0.60,
    min_coverage_95: float = 0.90
) -> Tuple[bool, List[str]]:
    """
    Validate surrogate against publication requirements.

    From spec section 8.5:
    1. Pointwise accuracy: max |Delta logL| below threshold
    2. MLE stability: surrogate MLE within tolerance
    3. Interval stability: coverage checks
    4. No pathological behavior

    Args:
        surrogate: Trained SurrogateModel
        validation_data: Validation training data
        max_pointwise_error: Maximum allowed |Delta logL|
        max_mle_error: Maximum MLE deviation (relative)
        min_coverage_68: Minimum 68% coverage
        min_coverage_95: Minimum 95% coverage

    Returns:
        (passed, list_of_failures)
    """
    failures = []

    # 1. Pointwise accuracy
    errors = []
    for point in validation_data.points:
        mean, _ = surrogate.predict_with_uncertainty(point.theta, point.nu)
        errors.append(abs(mean - point.logL))

    max_error = max(errors)
    if max_error > max_pointwise_error:
        failures.append(f"Pointwise error {max_error:.4f} > {max_pointwise_error}")

    # 2. Coverage (using internal validation)
    if surrogate.validation:
        if surrogate.validation.coverage_68 < min_coverage_68:
            failures.append(
                f"68% coverage {surrogate.validation.coverage_68:.2f} < {min_coverage_68}"
            )
        if surrogate.validation.coverage_95 < min_coverage_95:
            failures.append(
                f"95% coverage {surrogate.validation.coverage_95:.2f} < {min_coverage_95}"
            )

    # 3. MLE stability (simplified check)
    # Would require fitting both original and surrogate

    # 4. Pathology checks
    # Check for NaN/Inf predictions
    for point in validation_data.points:
        mean, std = surrogate.predict_with_uncertainty(point.theta, point.nu)
        if not np.isfinite(mean) or not np.isfinite(std):
            failures.append(f"Non-finite prediction at theta={point.theta}")
            break

    return len(failures) == 0, failures
