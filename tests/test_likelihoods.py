"""
Tests for likelihood model implementations.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.likelihoods import (
    PoissonLikelihood,
    GaussianLikelihood,
    HybridLikelihood,
    BreitWignerModel,
    FitHealth,
    CHI2_DOF_LOW,
    CHI2_DOF_HIGH
)


class TestPoissonLikelihood:
    """Tests for Poisson likelihood."""

    def test_basic_evaluation(self):
        """Test basic log-likelihood evaluation."""
        observed = np.array([10, 20, 15, 25])

        def model_func(theta, nu):
            return observed * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )

        # At mu=1, model equals data
        ll = model.logpdf(np.array([1.0]), np.array([]))
        assert np.isfinite(ll)

        # NLL should be non-negative
        nll = model.nll(np.array([1.0]), np.array([]))
        assert nll == -ll

    def test_gradient(self):
        """Test numerical gradient computation."""
        observed = np.array([10, 20, 15])

        def model_func(theta, nu):
            return observed * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )
        model._poi_names = ['mu']
        model._nuisance_names = []

        grad_theta, grad_nu = model.gradient(np.array([1.0]), np.array([]))

        # Gradient should exist
        assert len(grad_theta) == 1
        assert np.isfinite(grad_theta[0])

    def test_fit_health_assessment(self):
        """Test fit health diagnostics."""
        observed = np.array([100, 110, 95, 105, 100])

        def model_func(theta, nu):
            return np.ones_like(observed) * 100 * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )

        health = model.assess_fit_health(np.array([1.0]), np.array([]))

        assert health.chi2 >= 0
        assert health.dof > 0
        assert health.status in ["HEALTHY", "UNDERCONSTRAINED", "MODEL_MISMATCH"]

    def test_zero_counts(self):
        """Test handling of zero counts."""
        observed = np.array([0, 10, 0, 15])

        def model_func(theta, nu):
            return np.array([5, 10, 3, 15]) * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )

        ll = model.logpdf(np.array([1.0]), np.array([]))
        assert np.isfinite(ll)


class TestGaussianLikelihood:
    """Tests for Gaussian likelihood."""

    def test_basic_evaluation(self):
        """Test basic log-likelihood evaluation."""
        observed = np.array([1.0, 2.0, 1.5])
        errors = np.array([0.1, 0.2, 0.15])

        def model_func(theta, nu):
            return np.ones_like(observed) * theta[0]

        model = GaussianLikelihood(
            observed=observed,
            errors=errors,
            model_func=model_func
        )

        ll = model.logpdf(np.array([1.5]), np.array([]))
        assert np.isfinite(ll)

    def test_covariance_matrix(self):
        """Test with full covariance matrix."""
        observed = np.array([1.0, 2.0])
        errors = np.array([0.1, 0.2])
        covariance = np.array([
            [0.01, 0.005],
            [0.005, 0.04]
        ])

        def model_func(theta, nu):
            return np.ones_like(observed) * theta[0]

        model = GaussianLikelihood(
            observed=observed,
            errors=errors,
            model_func=model_func,
            covariance=covariance
        )

        assert model.use_full_covariance
        ll = model.logpdf(np.array([1.5]), np.array([]))
        assert np.isfinite(ll)


class TestHybridLikelihood:
    """Tests for Hybrid likelihood."""

    def test_combination(self):
        """Test combining Poisson and Gaussian channels."""
        # Poisson channel
        poisson_obs = np.array([10, 20])

        def poisson_model(theta, nu):
            return poisson_obs * theta[0]

        poisson = PoissonLikelihood(
            observed=poisson_obs,
            model_func=poisson_model
        )

        # Gaussian channel
        gaussian_obs = np.array([1.0, 2.0])
        gaussian_err = np.array([0.1, 0.2])

        def gaussian_model(theta, nu):
            return gaussian_obs * theta[0]

        gaussian = GaussianLikelihood(
            observed=gaussian_obs,
            errors=gaussian_err,
            model_func=gaussian_model
        )

        hybrid = HybridLikelihood(
            poisson_channels=[poisson],
            gaussian_channels=[gaussian]
        )

        ll = hybrid.logpdf(np.array([1.0]), np.array([]))
        assert np.isfinite(ll)

        health = hybrid.assess_fit_health(np.array([1.0]), np.array([]))
        assert health.status in ["HEALTHY", "UNDERCONSTRAINED", "MODEL_MISMATCH"]


class TestBreitWignerModel:
    """Tests for Breit-Wigner amplitude model."""

    def test_amplitude(self):
        """Test BW amplitude computation."""
        bw = BreitWignerModel(M1=9.4, G1=0.02, M2=10.0, G2=0.05)

        masses = np.linspace(8, 11, 50)
        intensity = bw.amplitude(masses, c1=1.0, r=0.5, phi=0.0, scale=100)

        assert len(intensity) == 50
        assert np.all(intensity > 0)
        assert np.all(np.isfinite(intensity))

    def test_resonance_peak(self):
        """Test that intensity peaks near resonance masses."""
        bw = BreitWignerModel(M1=9.4, G1=0.02, M2=10.0, G2=0.05)

        masses = np.linspace(8, 11, 200)
        intensity = bw.amplitude(masses, c1=1.0, r=0.0, phi=0.0, scale=100)

        # Peak should be near M1
        peak_idx = np.argmax(intensity)
        peak_mass = masses[peak_idx]
        assert abs(peak_mass - 9.4) < 0.1


class TestFitHealth:
    """Tests for fit health assessment."""

    def test_healthy_fit(self):
        """Test healthy chi2/dof classification."""
        health = FitHealth(chi2=15, dof=10, chi2_per_dof=1.5, deviance=15, deviance_per_dof=1.5)
        health.assess()
        assert health.status == "HEALTHY"
        assert health.is_healthy

    def test_underconstrained_fit(self):
        """Test underconstrained detection."""
        health = FitHealth(chi2=2, dof=10, chi2_per_dof=0.2, deviance=2, deviance_per_dof=0.2)
        health.assess()
        assert health.status == "UNDERCONSTRAINED"
        assert not health.is_healthy

    def test_model_mismatch(self):
        """Test model mismatch detection."""
        health = FitHealth(chi2=50, dof=10, chi2_per_dof=5.0, deviance=50, deviance_per_dof=5.0)
        health.assess()
        assert health.status == "MODEL_MISMATCH"
        assert not health.is_healthy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
