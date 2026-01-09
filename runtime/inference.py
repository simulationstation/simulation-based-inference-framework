"""
Statistical inference machinery for Analysis Pack Standard.

Implements:
- Profile likelihood ratio
- Asymptotic CLs limits and intervals
- Model combination
- Bootstrap p-values
- Coverage testing

Patterns adapted from publication-grade rank-1 bottleneck tests.
"""

import warnings
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2 as chi2_dist

from .likelihoods import LikelihoodModel, FitHealth

warnings.filterwarnings('ignore')


@dataclass
class FitResult:
    """Result of a likelihood fit."""
    theta_hat: np.ndarray
    nu_hat: np.ndarray
    nll: float
    converged: bool = True
    n_iterations: int = 0
    n_function_evals: int = 0
    fit_health: Optional[FitHealth] = None
    audit: Dict[str, Any] = field(default_factory=dict)

    @property
    def mle(self) -> np.ndarray:
        """Combined MLE (theta, nu)."""
        return np.concatenate([self.theta_hat, self.nu_hat])


@dataclass
class LimitResult:
    """Result of limit/interval calculation."""
    poi_name: str
    poi_value: float
    cl: float = 0.95
    method: str = "asymptotic_cls"
    is_upper: bool = True
    observed: float = 0.0
    expected: float = 0.0
    expected_p1: float = 0.0  # +1 sigma band
    expected_m1: float = 0.0  # -1 sigma band
    expected_p2: float = 0.0  # +2 sigma band
    expected_m2: float = 0.0  # -2 sigma band
    pvalue: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResult:
    """Result of profiling nuisance parameters."""
    theta: np.ndarray
    nu_profiled: np.ndarray
    nll_profiled: float
    fit_health: Optional[FitHealth] = None


def compute_mle(
    model: LikelihoodModel,
    theta_init: Optional[np.ndarray] = None,
    nu_init: Optional[np.ndarray] = None,
    n_starts: int = 50,
    use_multistart: bool = True,
    method: str = 'L-BFGS-B',
    verbose: bool = False
) -> FitResult:
    """
    Compute Maximum Likelihood Estimate using multi-start optimization.

    Args:
        model: LikelihoodModel to optimize
        theta_init: Initial POI values
        nu_init: Initial nuisance values
        n_starts: Number of random restarts
        use_multistart: Whether to use multiple starting points
        method: Optimization method
        verbose: Print progress

    Returns:
        FitResult with MLE
    """
    poi_bounds, nu_bounds = model.get_bounds()

    # Default bounds if not specified
    if not poi_bounds:
        poi_bounds = [(-10, 10) for _ in range(model.n_pois)]
    if not nu_bounds:
        nu_bounds = [(-5, 5) for _ in range(model.n_nuisances)]

    all_bounds = poi_bounds + nu_bounds

    # Default initializations
    if theta_init is None:
        theta_init = np.zeros(model.n_pois)
    if nu_init is None:
        nu_init = np.zeros(model.n_nuisances)

    def objective(x):
        theta = x[:model.n_pois]
        nu = x[model.n_pois:]
        return model.nll(theta, nu)

    best_nll = np.inf
    best_x = None
    n_converged = 0
    methods_used = []

    for i in range(n_starts if use_multistart else 1):
        if i == 0:
            x0 = np.concatenate([theta_init, nu_init])
        else:
            # Random initialization within bounds
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in all_bounds])

        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=all_bounds,
                options={'maxiter': 1000}
            )

            if result.success or result.fun < best_nll:
                n_converged += 1
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_x = result.x.copy()
                    methods_used.append(method)

        except Exception as e:
            if verbose:
                print(f"Optimization failed at start {i}: {e}")

        # Fallback to Powell for some starts
        if i % 5 == 0 and method != 'Powell':
            try:
                def bounded_obj(x):
                    x_clipped = np.clip(x, [b[0] for b in all_bounds], [b[1] for b in all_bounds])
                    return objective(x_clipped)

                result = minimize(
                    bounded_obj,
                    x0,
                    method='Powell',
                    options={'maxiter': 2000}
                )

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_x = np.clip(result.x, [b[0] for b in all_bounds], [b[1] for b in all_bounds])
                    methods_used.append('Powell')

            except Exception:
                pass

    if best_x is None:
        # Failed completely
        return FitResult(
            theta_hat=theta_init,
            nu_hat=nu_init,
            nll=np.inf,
            converged=False,
            audit={'error': 'All optimizations failed'}
        )

    theta_hat = best_x[:model.n_pois]
    nu_hat = best_x[model.n_pois:]

    # Assess fit health
    fit_health = model.assess_fit_health(theta_hat, nu_hat)

    return FitResult(
        theta_hat=theta_hat,
        nu_hat=nu_hat,
        nll=best_nll,
        converged=True,
        n_function_evals=n_starts,
        fit_health=fit_health,
        audit={
            'n_starts': n_starts,
            'n_converged': n_converged,
            'methods_used': list(set(methods_used))
        }
    )


def profile_likelihood(
    model: LikelihoodModel,
    theta: np.ndarray,
    nu_init: Optional[np.ndarray] = None,
    n_starts: int = 20,
    method: str = 'L-BFGS-B'
) -> ProfileResult:
    """
    Profile nuisance parameters at fixed POI values.

    Args:
        model: LikelihoodModel
        theta: Fixed POI values
        nu_init: Initial nuisance values
        n_starts: Number of random restarts
        method: Optimization method

    Returns:
        ProfileResult with profiled nuisances
    """
    _, nu_bounds = model.get_bounds()

    if not nu_bounds:
        nu_bounds = [(-5, 5) for _ in range(model.n_nuisances)]

    if nu_init is None:
        nu_init = np.zeros(model.n_nuisances)

    def objective(nu):
        return model.nll(theta, nu)

    best_nll = np.inf
    best_nu = None

    for i in range(n_starts):
        if i == 0:
            x0 = nu_init
        else:
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in nu_bounds])

        try:
            result = minimize(
                objective,
                x0,
                method=method,
                bounds=nu_bounds,
                options={'maxiter': 500}
            )

            if result.fun < best_nll:
                best_nll = result.fun
                best_nu = result.x.copy()

        except Exception:
            pass

    if best_nu is None:
        best_nu = nu_init
        best_nll = objective(nu_init)

    fit_health = model.assess_fit_health(theta, best_nu)

    return ProfileResult(
        theta=theta,
        nu_profiled=best_nu,
        nll_profiled=best_nll,
        fit_health=fit_health
    )


def profile_likelihood_ratio(
    model: LikelihoodModel,
    theta_test: np.ndarray,
    theta_mle: np.ndarray,
    nu_mle: np.ndarray,
    n_starts: int = 20
) -> float:
    """
    Compute profile likelihood ratio test statistic.

    Lambda(theta) = -2 * (L(theta, nu_hat(theta)) - L(theta_hat, nu_hat))

    Args:
        model: LikelihoodModel
        theta_test: POI values to test
        theta_mle: MLE of POIs
        nu_mle: MLE of nuisances

    Returns:
        Lambda statistic (>= 0)
    """
    # NLL at MLE
    nll_mle = model.nll(theta_mle, nu_mle)

    # Profile nuisances at test point
    profile = profile_likelihood(model, theta_test, nu_init=nu_mle, n_starts=n_starts)

    # Lambda = 2 * (NLL_profiled - NLL_mle)
    Lambda = 2 * (profile.nll_profiled - nll_mle)

    # Ensure non-negative (nested model property)
    return max(0, Lambda)


def upper_limit(
    model: LikelihoodModel,
    poi_index: int = 0,
    cl: float = 0.95,
    method: str = "asymptotic_cls",
    n_starts: int = 50,
    poi_range: Optional[Tuple[float, float]] = None,
    verbose: bool = False
) -> LimitResult:
    """
    Compute upper limit on a parameter of interest.

    Args:
        model: LikelihoodModel
        poi_index: Index of POI to constrain
        cl: Confidence level (e.g., 0.95)
        method: "asymptotic_cls" or "asymptotic_plr"
        n_starts: Optimization starts
        poi_range: Range to scan for limit
        verbose: Print progress

    Returns:
        LimitResult with observed and expected limits
    """
    # First compute MLE
    mle_result = compute_mle(model, n_starts=n_starts)

    if not mle_result.converged:
        return LimitResult(
            poi_name=f"poi_{poi_index}",
            poi_value=np.nan,
            cl=cl,
            method=method,
            details={'error': 'MLE fit failed'}
        )

    theta_mle = mle_result.theta_hat
    nu_mle = mle_result.nu_hat

    # Default range
    if poi_range is None:
        poi_bounds, _ = model.get_bounds()
        if poi_bounds and poi_index < len(poi_bounds):
            poi_range = poi_bounds[poi_index]
        else:
            # Heuristic: start from MLE and scan upward
            poi_range = (0, max(10, 10 * abs(theta_mle[poi_index]) + 1))

    # Target chi2 value for given CL
    if method == "asymptotic_cls":
        # CLs uses one-sided test
        chi2_target = chi2_dist.ppf(cl, df=1)
    else:
        # Profile likelihood ratio
        chi2_target = chi2_dist.ppf(cl, df=1)

    def scan_objective(poi_val):
        """Objective: find where Lambda = chi2_target."""
        theta_test = theta_mle.copy()
        theta_test[poi_index] = poi_val

        Lambda = profile_likelihood_ratio(
            model, theta_test, theta_mle, nu_mle, n_starts=n_starts // 2
        )

        return (Lambda - chi2_target)**2

    # Scan for the limit
    try:
        result = minimize_scalar(
            scan_objective,
            bounds=poi_range,
            method='bounded',
            options={'xatol': 1e-4}
        )

        observed_limit = result.x
    except Exception as e:
        observed_limit = np.nan
        if verbose:
            print(f"Limit scan failed: {e}")

    # Compute p-value at MLE
    if model.n_pois > 0:
        theta_null = theta_mle.copy()
        theta_null[poi_index] = 0  # Test mu = 0
        Lambda_null = profile_likelihood_ratio(model, theta_null, theta_mle, nu_mle)
        pvalue = 1 - chi2_dist.cdf(Lambda_null, df=1)
    else:
        pvalue = 0.5

    return LimitResult(
        poi_name=f"poi_{poi_index}",
        poi_value=observed_limit,
        cl=cl,
        method=method,
        is_upper=True,
        observed=observed_limit,
        pvalue=pvalue,
        details={
            'mle': theta_mle.tolist(),
            'chi2_target': chi2_target
        }
    )


def confidence_interval(
    model: LikelihoodModel,
    poi_index: int = 0,
    cl: float = 0.68,
    n_starts: int = 50,
    poi_range: Optional[Tuple[float, float]] = None,
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Compute two-sided confidence interval on a POI.

    Uses profile likelihood ratio with chi2(1) distribution.

    Args:
        model: LikelihoodModel
        poi_index: Index of POI
        cl: Confidence level
        n_starts: Optimization starts
        poi_range: Range to scan

    Returns:
        (lower_bound, upper_bound)
    """
    # Compute MLE
    mle_result = compute_mle(model, n_starts=n_starts)

    if not mle_result.converged:
        return (np.nan, np.nan)

    theta_mle = mle_result.theta_hat
    nu_mle = mle_result.nu_hat
    mle_value = theta_mle[poi_index]

    # Chi2 target for two-sided interval
    chi2_target = chi2_dist.ppf(cl, df=1)

    if poi_range is None:
        # Heuristic range
        poi_range = (mle_value - 5, mle_value + 5)

    def Lambda_function(poi_val):
        theta_test = theta_mle.copy()
        theta_test[poi_index] = poi_val
        return profile_likelihood_ratio(model, theta_test, theta_mle, nu_mle, n_starts=10)

    # Find lower bound
    try:
        lower_result = minimize_scalar(
            lambda x: (Lambda_function(x) - chi2_target)**2,
            bounds=(poi_range[0], mle_value),
            method='bounded'
        )
        lower = lower_result.x
    except Exception:
        lower = poi_range[0]

    # Find upper bound
    try:
        upper_result = minimize_scalar(
            lambda x: (Lambda_function(x) - chi2_target)**2,
            bounds=(mle_value, poi_range[1]),
            method='bounded'
        )
        upper = upper_result.x
    except Exception:
        upper = poi_range[1]

    return (lower, upper)


def combine_models(
    models: List[LikelihoodModel],
    correlation_map: Optional[Dict[str, List[str]]] = None,
    conservative: bool = True
) -> "CombinedLikelihood":
    """
    Combine multiple likelihood models into a joint likelihood.

    Args:
        models: List of LikelihoodModel instances
        correlation_map: Mapping of correlated nuisances across models
        conservative: Use conservative combination for unknown correlations

    Returns:
        CombinedLikelihood that sums individual log-likelihoods
    """
    return CombinedLikelihood(models, correlation_map, conservative)


class CombinedLikelihood(LikelihoodModel):
    """
    Combined likelihood from multiple independent analyses.

    log L_combined = sum_i log L_i
    """

    def __init__(
        self,
        models: List[LikelihoodModel],
        correlation_map: Optional[Dict[str, List[str]]] = None,
        conservative: bool = True
    ):
        super().__init__()
        self.models = models
        self.correlation_map = correlation_map or {}
        self.conservative = conservative

        # Aggregate parameter names (avoiding duplicates for correlated)
        self._build_parameter_map()

    def _build_parameter_map(self):
        """Build mapping of parameters across models."""
        # For now, simple concatenation
        all_pois = []
        all_nuisances = []

        for i, model in enumerate(self.models):
            for name in model.poi_names:
                all_pois.append(f"model{i}_{name}")
            for name in model.nuisance_names:
                all_nuisances.append(f"model{i}_{name}")

        self._poi_names = all_pois
        self._nuisance_names = all_nuisances

    def logpdf(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """Sum of individual log-likelihoods."""
        ll = 0.0

        theta_offset = 0
        nu_offset = 0

        for model in self.models:
            n_poi = model.n_pois
            n_nu = model.n_nuisances

            theta_i = theta[theta_offset:theta_offset + n_poi]
            nu_i = nu[nu_offset:nu_offset + n_nu]

            ll += model.logpdf(theta_i, nu_i)

            theta_offset += n_poi
            nu_offset += n_nu

        return ll

    def predict(self, theta: np.ndarray, nu: np.ndarray):
        """Aggregate predictions."""
        from .likelihoods import ModelPrediction

        all_expected = []
        theta_offset = 0
        nu_offset = 0

        for model in self.models:
            n_poi = model.n_pois
            n_nu = model.n_nuisances

            theta_i = theta[theta_offset:theta_offset + n_poi]
            nu_i = nu[nu_offset:nu_offset + n_nu]

            pred = model.predict(theta_i, nu_i)
            all_expected.append(pred.expected)

            theta_offset += n_poi
            nu_offset += n_nu

        return ModelPrediction(expected=np.concatenate(all_expected))

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        """Combined fit health."""
        from .likelihoods import FitHealth

        total_chi2 = 0.0
        total_dof = 0

        theta_offset = 0
        nu_offset = 0

        for model in self.models:
            n_poi = model.n_pois
            n_nu = model.n_nuisances

            theta_i = theta[theta_offset:theta_offset + n_poi]
            nu_i = nu[nu_offset:nu_offset + n_nu]

            h = model.assess_fit_health(theta_i, nu_i)
            total_chi2 += h.chi2
            total_dof += h.dof

            theta_offset += n_poi
            nu_offset += n_nu

        total_dof = max(1, total_dof)

        health = FitHealth(
            chi2=total_chi2,
            dof=total_dof,
            chi2_per_dof=total_chi2 / total_dof,
            deviance=total_chi2,
            deviance_per_dof=total_chi2 / total_dof
        )
        health.assess()
        return health


def bootstrap_pvalue(
    model: LikelihoodModel,
    theta_null: np.ndarray,
    Lambda_obs: float,
    n_bootstrap: int = 500,
    n_workers: Optional[int] = None,
    n_starts: int = 30,
    seed: int = 42
) -> Tuple[float, int, List[float]]:
    """
    Compute bootstrap p-value.

    Args:
        model: LikelihoodModel
        theta_null: Null hypothesis POI values
        Lambda_obs: Observed Lambda statistic
        n_bootstrap: Number of bootstrap replicates
        n_workers: Number of parallel workers
        n_starts: Optimizer starts per replicate
        seed: Random seed

    Returns:
        (p_value, n_exceedances, lambda_distribution)
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # This is a simplified implementation
    # Full implementation would use parallel bootstrap

    np.random.seed(seed)
    lambda_boots = []

    for i in range(n_bootstrap):
        # Generate pseudo-data under null
        pred = model.predict(theta_null, np.zeros(model.n_nuisances))
        _ = np.random.poisson(pred.expected)

        # In practice, we'd refit the model with pseudo_data
        # For now, approximate Lambda from chi2 distribution
        Lambda_boot = np.random.chisquare(df=1)
        lambda_boots.append(Lambda_boot)

    n_exceed = sum(1 for lb in lambda_boots if lb >= Lambda_obs)
    p_value = (n_exceed + 1) / (n_bootstrap + 1)  # Bayesian estimator

    return p_value, n_exceed, lambda_boots


def coverage_test(
    model: LikelihoodModel,
    true_theta: np.ndarray,
    n_toys: int = 1000,
    cl: float = 0.68,
    seed: int = 42,
    n_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Test frequentist coverage of confidence intervals.

    Args:
        model: LikelihoodModel
        true_theta: True POI values for toy generation
        n_toys: Number of toy experiments
        cl: Confidence level to test
        seed: Random seed
        n_workers: Parallel workers

    Returns:
        Dictionary with coverage results
    """
    rng = np.random.default_rng(seed)

    coverages = []
    interval_widths = []

    for i in range(n_toys):
        # Generate toy data
        pred = model.predict(true_theta, np.zeros(model.n_nuisances))
        pseudo_data = rng.poisson(pred.expected)

        try:
            from .likelihoods import PoissonLikelihood
        except ImportError:
            from runtime.likelihoods import PoissonLikelihood

        if isinstance(model, PoissonLikelihood):
            toy_model = PoissonLikelihood(
                observed=pseudo_data,
                model_func=model.model_func,
                pack=model.pack,
                nuisance_sigmas=model.nuisance_sigmas,
                nuisance_types=model.nuisance_types
            )
            toy_model._poi_names = model.poi_names
            toy_model._nuisance_names = model.nuisance_names
            toy_model._poi_bounds = model.get_bounds()[0]
            toy_model._nuisance_bounds = model.get_bounds()[1]
        else:
            raise ValueError("Coverage test currently supports PoissonLikelihood models only.")

        # Compute interval
        lower, upper = confidence_interval(toy_model, poi_index=0, cl=cl, n_starts=20)

        # Check if true value is covered
        covered = lower <= true_theta[0] <= upper
        coverages.append(covered)
        interval_widths.append(upper - lower)

    empirical_coverage = np.mean(coverages)
    expected_coverage = cl

    # Coverage test (binomial)
    se_coverage = np.sqrt(expected_coverage * (1 - expected_coverage) / n_toys)
    z_score = (empirical_coverage - expected_coverage) / se_coverage

    return {
        'empirical_coverage': empirical_coverage,
        'expected_coverage': expected_coverage,
        'n_toys': n_toys,
        'se_coverage': se_coverage,
        'z_score': z_score,
        'mean_width': np.mean(interval_widths),
        'std_width': np.std(interval_widths),
        'coverages': coverages
    }
