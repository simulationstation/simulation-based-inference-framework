"""
Tests for surrogate model builders.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.likelihoods import PoissonLikelihood
from surrogates.builder import (
    SurrogateBuilder,
    GPSurrogate,
    NeuralSurrogate,
    ParametricSurrogate,
    TrainingData,
    TrainingPoint
)
from surrogates.calibration import (
    compute_calibration_curve,
    calibrate_surrogate,
    inflate_intervals
)


class TestTrainingData:
    """Tests for training data generation."""

    def test_lhs_sampling(self):
        """Test Latin hypercube sampling."""
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            return np.ones(4) * 100 * theta[0]

        model = PoissonLikelihood(observed=observed, model_func=model_func)
        model._poi_names = ['mu']
        model._nuisance_names = []

        builder = SurrogateBuilder(model, surrogate_type='gp')
        data = builder.generate_training_data(
            n_samples=50,
            method='lhs',
            theta_bounds=[(0.5, 2.0)],
            nu_bounds=[],
            seed=42
        )

        assert len(data) == 50
        assert len(data.points[0].theta) == 1
        assert all(np.isfinite(p.logL) for p in data.points)

    def test_training_data_to_arrays(self):
        """Test conversion to numpy arrays."""
        data = TrainingData()
        for i in range(10):
            data.points.append(TrainingPoint(
                theta=np.array([i * 0.1]),
                nu=np.array([0.0]),
                logL=-i
            ))

        X, y = data.to_arrays()

        assert X.shape == (10, 2)
        assert y.shape == (10,)


class TestGPSurrogate:
    """Tests for Gaussian Process surrogate."""

    def test_training_and_prediction(self):
        """Test GP training and prediction."""
        # Create simple training data
        data = TrainingData()
        for i in range(20):
            x = np.random.rand() * 2 + 0.5
            data.points.append(TrainingPoint(
                theta=np.array([x]),
                nu=np.array([]),
                logL=-0.5 * (x - 1.0)**2
            ))

        gp = GPSurrogate(kernel='rbf')
        gp.train(data)

        assert gp.is_trained

        # Predict at training point
        mean, std = gp.predict_with_uncertainty(np.array([1.0]), np.array([]))

        assert np.isfinite(mean)
        assert std >= 0

    def test_uncertainty_increases_extrapolation(self):
        """Test that uncertainty increases for extrapolation."""
        # Training data in [0.5, 1.5]
        data = TrainingData()
        for x in np.linspace(0.5, 1.5, 20):
            data.points.append(TrainingPoint(
                theta=np.array([x]),
                nu=np.array([]),
                logL=-0.5 * x**2
            ))

        gp = GPSurrogate(kernel='rbf')
        gp.train(data)

        # Uncertainty at training region
        _, std_in = gp.predict_with_uncertainty(np.array([1.0]), np.array([]))

        # Uncertainty outside training region
        _, std_out = gp.predict_with_uncertainty(np.array([3.0]), np.array([]))

        # Should have higher uncertainty extrapolating
        assert std_out > std_in


class TestNeuralSurrogate:
    """Tests for neural network surrogate."""

    def test_training_and_prediction(self):
        """Test neural surrogate training."""
        data = TrainingData()
        for i in range(50):
            x = np.random.rand() * 2
            data.points.append(TrainingPoint(
                theta=np.array([x]),
                nu=np.array([]),
                logL=-x**2
            ))

        nn = NeuralSurrogate(
            hidden_layers=[16, 16],
            n_ensemble=3,
            epochs=100
        )
        nn.train(data)

        assert nn.is_trained

        mean, std = nn.predict_with_uncertainty(np.array([1.0]), np.array([]))

        assert np.isfinite(mean)
        assert std >= 0


class TestParametricSurrogate:
    """Tests for parametric surrogate."""

    def test_polynomial_fit(self):
        """Test polynomial surrogate."""
        # Quadratic function
        data = TrainingData()
        for x in np.linspace(-2, 2, 30):
            data.points.append(TrainingPoint(
                theta=np.array([x]),
                nu=np.array([]),
                logL=1.0 - x**2 + 0.5 * x
            ))

        poly = ParametricSurrogate(order=2)
        poly.train(data)

        assert poly.is_trained

        # Should fit well for quadratic
        mean, _ = poly.predict_with_uncertainty(np.array([0.0]), np.array([]))
        assert abs(mean - 1.0) < 0.1


class TestSurrogateBuilder:
    """Tests for the surrogate builder."""

    def test_build_gp(self):
        """Test building GP surrogate."""
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            return np.ones(4) * 100 * theta[0]

        model = PoissonLikelihood(observed=observed, model_func=model_func)
        model._poi_names = ['mu']
        model._nuisance_names = []

        builder = SurrogateBuilder(model, surrogate_type='gp')
        builder.generate_training_data(
            n_samples=30,
            theta_bounds=[(0.5, 2.0)],
            nu_bounds=[]
        )

        surrogate = builder.build(validation_split=0.2)

        assert surrogate.is_trained
        assert surrogate.validation is not None

    def test_validation(self):
        """Test surrogate validation."""
        # Create well-behaved training data
        data = TrainingData()
        for x in np.linspace(0.5, 2.0, 40):
            data.points.append(TrainingPoint(
                theta=np.array([x]),
                nu=np.array([]),
                logL=-10 * (x - 1.0)**2
            ))

        observed = np.array([100])

        def model_func(theta, nu):
            return np.array([100 * theta[0]])

        model = PoissonLikelihood(observed=observed, model_func=model_func)

        builder = SurrogateBuilder(model, surrogate_type='gp')
        builder.training_data = data

        surrogate = builder.build(validation_split=0.25)

        # Check validation metrics exist
        assert surrogate.validation.mae >= 0
        assert surrogate.validation.rmse >= 0
        assert 0 <= surrogate.validation.coverage_68 <= 1


class TestCalibration:
    """Tests for calibration functions."""

    def test_calibration_curve_perfect(self):
        """Test calibration for perfectly calibrated predictions."""
        np.random.seed(42)
        n = 100

        # Perfect predictions with correct uncertainty
        true_values = np.random.randn(n)
        predictions = true_values  # Perfect mean
        uncertainties = np.ones(n)  # Correct std

        result = compute_calibration_curve(predictions, uncertainties, true_values)

        # Should be well calibrated
        assert result.calibration_slope > 0.8
        assert result.calibration_slope < 1.2

    def test_calibration_curve_overconfident(self):
        """Test detection of overconfident predictions."""
        np.random.seed(42)
        n = 100

        true_values = np.random.randn(n)
        predictions = true_values + np.random.randn(n) * 0.5  # Some noise
        uncertainties = np.ones(n) * 0.1  # Underestimated uncertainty

        result = compute_calibration_curve(predictions, uncertainties, true_values)

        # Should detect overconfidence (low empirical coverage)
        assert result.calibration_slope < 1.0
        assert result.inflation_factor > 1.0

    def test_interval_inflation(self):
        """Test interval inflation."""
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 3.0])

        new_lower, new_upper = inflate_intervals(lower, upper, inflation_factor=2.0)

        # Width should double
        orig_width = upper - lower
        new_width = new_upper - new_lower

        np.testing.assert_allclose(new_width, orig_width * 2)

        # Center should be preserved
        orig_center = (lower + upper) / 2
        new_center = (new_lower + new_upper) / 2

        np.testing.assert_allclose(new_center, orig_center)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
