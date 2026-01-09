"""
Tests for statistical inference machinery.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.likelihoods import PoissonLikelihood, GaussianLikelihood
from runtime.inference import (
    compute_mle,
    profile_likelihood,
    profile_likelihood_ratio,
    upper_limit,
    confidence_interval,
    combine_models,
    coverage_test
)


class TestMLE:
    """Tests for MLE computation."""

    def test_basic_mle(self):
        """Test MLE finds correct value for simple model."""
        # True mu = 1.5
        np.random.seed(42)
        true_mu = 1.5
        observed = np.random.poisson(100 * true_mu, size=20)

        def model_func(theta, nu):
            return np.ones(20) * 100 * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )
        model._poi_names = ['mu']
        model._nuisance_names = []
        model._poi_bounds = [(0.1, 5.0)]
        model._nuisance_bounds = []

        result = compute_mle(model, n_starts=20)

        assert result.converged
        # MLE should be close to sum(observed) / sum(expected at mu=1)
        expected_mle = np.sum(observed) / (20 * 100)
        assert abs(result.theta_hat[0] - expected_mle) < 0.1

    def test_mle_with_nuisances(self):
        """Test MLE with nuisance parameters."""
        np.random.seed(42)
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            # mu * (1 + nu[0] * 0.1)
            return np.ones(4) * 100 * theta[0] * (1 + nu[0] * 0.1)

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func,
            nuisance_sigmas=np.array([1.0])
        )
        model._poi_names = ['mu']
        model._nuisance_names = ['syst']
        model._poi_bounds = [(0.1, 5.0)]
        model._nuisance_bounds = [(-3, 3)]

        result = compute_mle(model, n_starts=20)

        assert result.converged
        assert len(result.theta_hat) == 1
        assert len(result.nu_hat) == 1


class TestProfileLikelihood:
    """Tests for profile likelihood."""

    def test_profiling(self):
        """Test nuisance profiling at fixed POI."""
        np.random.seed(42)
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            return np.ones(4) * 100 * theta[0] * (1 + nu[0] * 0.05)

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func,
            nuisance_sigmas=np.array([1.0])
        )
        model._poi_names = ['mu']
        model._nuisance_names = ['syst']
        model._poi_bounds = [(0.1, 5.0)]
        model._nuisance_bounds = [(-3, 3)]

        result = profile_likelihood(model, theta=np.array([1.0]), n_starts=10)

        assert np.isfinite(result.nll_profiled)
        assert len(result.nu_profiled) == 1


class TestProfileLikelihoodRatio:
    """Tests for profile likelihood ratio."""

    def test_plr_at_mle(self):
        """Test PLR is zero at MLE."""
        np.random.seed(42)
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            return np.ones(4) * 100 * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )
        model._poi_names = ['mu']
        model._nuisance_names = []
        model._poi_bounds = [(0.1, 5.0)]
        model._nuisance_bounds = []

        mle = compute_mle(model, n_starts=10)

        Lambda = profile_likelihood_ratio(
            model,
            theta_test=mle.theta_hat,
            theta_mle=mle.theta_hat,
            nu_mle=mle.nu_hat,
            n_starts=10
        )

        assert Lambda < 0.01  # Should be approximately zero

    def test_plr_away_from_mle(self):
        """Test PLR increases away from MLE."""
        np.random.seed(42)
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            return np.ones(4) * 100 * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )
        model._poi_names = ['mu']
        model._nuisance_names = []
        model._poi_bounds = [(0.1, 5.0)]
        model._nuisance_bounds = []

        mle = compute_mle(model, n_starts=10)

        # Test at value away from MLE
        test_theta = mle.theta_hat + 0.5

        Lambda = profile_likelihood_ratio(
            model,
            theta_test=test_theta,
            theta_mle=mle.theta_hat,
            nu_mle=mle.nu_hat,
            n_starts=10
        )

        assert Lambda > 0  # Should be positive away from MLE


class TestLimits:
    """Tests for limit computation."""

    def test_upper_limit(self):
        """Test upper limit computation."""
        np.random.seed(42)
        observed = np.array([5, 8, 3, 6])  # Small counts

        def model_func(theta, nu):
            return np.ones(4) * 5 * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )
        model._poi_names = ['mu']
        model._nuisance_names = []
        model._poi_bounds = [(0.01, 10.0)]
        model._nuisance_bounds = []

        result = upper_limit(
            model,
            poi_index=0,
            cl=0.95,
            method="asymptotic_cls",
            n_starts=20,
            poi_range=(0.1, 5.0)
        )

        assert result.observed > 0
        assert np.isfinite(result.observed)
        assert result.cl == 0.95


class TestConfidenceInterval:
    """Tests for confidence interval computation."""

    def test_interval_contains_mle(self):
        """Test that interval contains MLE."""
        np.random.seed(42)
        observed = np.array([100, 110, 95, 105])

        def model_func(theta, nu):
            return np.ones(4) * 100 * theta[0]

        model = PoissonLikelihood(
            observed=observed,
            model_func=model_func
        )
        model._poi_names = ['mu']
        model._nuisance_names = []
        model._poi_bounds = [(0.1, 5.0)]
        model._nuisance_bounds = []

        mle = compute_mle(model, n_starts=10)

        lower, upper = confidence_interval(
            model,
            poi_index=0,
            cl=0.68,
            n_starts=10,
            poi_range=(0.5, 2.0)
        )

        assert lower < mle.theta_hat[0] < upper


class TestCombination:
    """Tests for model combination."""

    def test_combine_independent(self):
        """Test combining two independent models."""
        obs1 = np.array([100, 110])
        obs2 = np.array([50, 55])

        def model1_func(theta, nu):
            return np.ones(2) * 100 * theta[0]

        def model2_func(theta, nu):
            return np.ones(2) * 50 * theta[0]

        model1 = PoissonLikelihood(observed=obs1, model_func=model1_func)
        model1._poi_names = ['mu1']
        model1._nuisance_names = []

        model2 = PoissonLikelihood(observed=obs2, model_func=model2_func)
        model2._poi_names = ['mu2']
        model2._nuisance_names = []

        combined = combine_models([model1, model2])

        assert combined.n_pois == 2

        # Combined log-likelihood should be sum
        ll1 = model1.logpdf(np.array([1.0]), np.array([]))
        ll2 = model2.logpdf(np.array([1.0]), np.array([]))
        ll_combined = combined.logpdf(np.array([1.0, 1.0]), np.array([]))

        assert abs(ll_combined - (ll1 + ll2)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
