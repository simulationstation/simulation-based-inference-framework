"""
Likelihood model implementations for Analysis Pack Standard.

Supports Poisson, Gaussian, and Hybrid likelihood models with:
- Vectorized evaluation
- Automatic gradient computation (when available)
- Numerical stability safeguards
- Fit health diagnostics

Patterns adapted from publication-grade rank-1 bottleneck tests.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import special

# Fit health thresholds (from reference implementation)
CHI2_DOF_LOW = 0.5    # Below = underconstrained
CHI2_DOF_HIGH = 3.0   # Above = model mismatch
DEVIANCE_DOF_HIGH = 3.0


@dataclass
class FitHealth:
    """Fit health diagnostics."""
    chi2: float = 0.0
    dof: int = 1
    chi2_per_dof: float = 0.0
    deviance: float = 0.0
    deviance_per_dof: float = 0.0
    status: str = "UNKNOWN"
    reason: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.status == "HEALTHY"

    def assess(self) -> None:
        """Assess fit health based on chi2/dof and deviance/dof."""
        if self.chi2_per_dof < CHI2_DOF_LOW:
            self.status = "UNDERCONSTRAINED"
            self.reason = f"chi2/dof={self.chi2_per_dof:.2f} < {CHI2_DOF_LOW}"
        elif self.chi2_per_dof > CHI2_DOF_HIGH:
            self.status = "MODEL_MISMATCH"
            self.reason = f"chi2/dof={self.chi2_per_dof:.2f} > {CHI2_DOF_HIGH}"
        elif self.deviance_per_dof > DEVIANCE_DOF_HIGH:
            self.status = "MODEL_MISMATCH"
            self.reason = f"deviance/dof={self.deviance_per_dof:.2f} > {DEVIANCE_DOF_HIGH}"
        else:
            self.status = "HEALTHY"
            self.reason = f"chi2/dof={self.chi2_per_dof:.2f}, deviance/dof={self.deviance_per_dof:.2f}"


@dataclass
class ModelPrediction:
    """Model prediction at a given parameter point."""
    expected: np.ndarray  # Expected counts/values
    signal: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    gradients: Optional[np.ndarray] = None


class LikelihoodModel(ABC):
    """
    Abstract base class for likelihood models.

    A likelihood model can evaluate log L(data | theta, nu), compute
    gradients, profile nuisances, and assess fit health.
    """

    def __init__(self, pack: Any = None):
        self.pack = pack
        self._poi_names: List[str] = []
        self._nuisance_names: List[str] = []
        self._poi_bounds: List[Tuple[float, float]] = []
        self._nuisance_bounds: List[Tuple[float, float]] = []

    @property
    def n_pois(self) -> int:
        return len(self._poi_names)

    @property
    def n_nuisances(self) -> int:
        return len(self._nuisance_names)

    @property
    def n_params(self) -> int:
        return self.n_pois + self.n_nuisances

    @property
    def poi_names(self) -> List[str]:
        return self._poi_names

    @property
    def nuisance_names(self) -> List[str]:
        return self._nuisance_names

    @abstractmethod
    def logpdf(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """
        Evaluate log-likelihood at given parameters.

        Args:
            theta: Parameters of interest
            nu: Nuisance parameters

        Returns:
            Log-likelihood value
        """
        pass

    def nll(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """Negative log-likelihood (for minimization)."""
        return -self.logpdf(theta, nu)

    @abstractmethod
    def predict(self, theta: np.ndarray, nu: np.ndarray) -> ModelPrediction:
        """
        Compute model prediction at given parameters.

        Args:
            theta: Parameters of interest
            nu: Nuisance parameters

        Returns:
            ModelPrediction with expected values
        """
        pass

    def gradient(self, theta: np.ndarray, nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of log-likelihood.

        Default implementation uses numerical differentiation.

        Returns:
            (grad_theta, grad_nu)
        """
        eps = 1e-6

        grad_theta = np.zeros(self.n_pois)
        grad_nu = np.zeros(self.n_nuisances)

        ll0 = self.logpdf(theta, nu)

        for i in range(self.n_pois):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            ll_plus = self.logpdf(theta_plus, nu)
            grad_theta[i] = (ll_plus - ll0) / eps

        for i in range(self.n_nuisances):
            nu_plus = nu.copy()
            nu_plus[i] += eps
            ll_plus = self.logpdf(theta, nu_plus)
            grad_nu[i] = (ll_plus - ll0) / eps

        return grad_theta, grad_nu

    @abstractmethod
    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        """Assess fit quality at given parameters."""
        pass

    def constraint_nll(self, nu: np.ndarray) -> float:
        """
        Negative log-likelihood of nuisance constraint terms.

        Default: standard normal constraints (nu ~ N(0,1))
        """
        return 0.5 * np.sum(nu**2)

    def get_bounds(self) -> Tuple[List[Tuple], List[Tuple]]:
        """Get parameter bounds for optimization."""
        return self._poi_bounds, self._nuisance_bounds


class PoissonLikelihood(LikelihoodModel):
    """
    Poisson likelihood for count data.

    L(n | mu) = prod_i Poisson(n_i | mu_i)

    With nuisance constraints for systematics.
    """

    def __init__(
        self,
        observed: np.ndarray,
        model_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        pack: Any = None,
        nuisance_sigmas: Optional[np.ndarray] = None,
        nuisance_types: Optional[List[str]] = None
    ):
        """
        Args:
            observed: Observed counts per bin
            model_func: Function (theta, nu) -> expected counts
            pack: Optional AnalysisPack reference
            nuisance_sigmas: Uncertainties for nuisance constraints
            nuisance_types: Constraint types ('normal', 'lognormal', etc.)
        """
        super().__init__(pack)
        self.observed = observed
        self.model_func = model_func
        self.nuisance_sigmas = nuisance_sigmas if nuisance_sigmas is not None else np.array([])
        self.nuisance_types = nuisance_types if nuisance_types is not None else []

    def logpdf(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """Poisson log-likelihood with nuisance constraints."""
        mu = self.model_func(theta, nu)
        mu = np.maximum(mu, 1e-10)  # Prevent log(0)

        # Poisson log-likelihood: sum(n*log(mu) - mu - log(n!))
        # We use the saturated form to avoid log(n!)
        ll = 0.0
        for i, (n, m) in enumerate(zip(self.observed, mu)):
            if n > 0:
                ll += n * np.log(m) - m - special.gammaln(n + 1)
            else:
                ll += -m

        # Add constraint terms
        ll -= self.constraint_nll(nu)

        return ll

    def predict(self, theta: np.ndarray, nu: np.ndarray) -> ModelPrediction:
        """Compute expected counts."""
        expected = self.model_func(theta, nu)
        return ModelPrediction(expected=expected)

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        """Assess Poisson fit health using Pearson chi2 and deviance."""
        mu = self.model_func(theta, nu)
        mu = np.maximum(mu, 1e-10)

        # Pearson chi-squared
        chi2 = np.sum((self.observed - mu)**2 / mu)

        # Poisson deviance: 2 * sum(n*log(n/mu) - (n - mu))
        deviance = 0.0
        for n, m in zip(self.observed, mu):
            if n > 0:
                deviance += 2 * (n * np.log(n / m) - (n - m))
            else:
                deviance += 2 * m

        n_bins = len(self.observed)
        n_params = len(theta) + len(nu)
        dof = max(1, n_bins - n_params)

        health = FitHealth(
            chi2=chi2,
            dof=dof,
            chi2_per_dof=chi2 / dof,
            deviance=deviance,
            deviance_per_dof=deviance / dof
        )
        health.assess()
        return health

    def constraint_nll(self, nu: np.ndarray) -> float:
        """Compute nuisance constraint NLL."""
        if len(nu) == 0:
            return 0.0

        nll = 0.0
        for i, n_val in enumerate(nu):
            if i < len(self.nuisance_types):
                ntype = self.nuisance_types[i]
            else:
                ntype = 'normal'

            sigma = self.nuisance_sigmas[i] if i < len(self.nuisance_sigmas) else 1.0

            if ntype == 'normal':
                nll += 0.5 * (n_val / sigma)**2
            elif ntype == 'lognormal':
                # nu is log(multiplier), expect nu ~ N(0, sigma)
                nll += 0.5 * (n_val / sigma)**2
            elif ntype == 'fixed':
                # Fixed parameters don't contribute
                pass

        return nll


class GaussianLikelihood(LikelihoodModel):
    """
    Gaussian likelihood for continuous measurements.

    L(x | mu, sigma) = prod_i N(x_i | mu_i, sigma_i)

    Supports full covariance matrices.
    """

    def __init__(
        self,
        observed: np.ndarray,
        errors: np.ndarray,
        model_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        pack: Any = None,
        covariance: Optional[np.ndarray] = None,
        nuisance_sigmas: Optional[np.ndarray] = None
    ):
        """
        Args:
            observed: Observed values
            errors: Measurement errors (diagonal of covariance)
            model_func: Function (theta, nu) -> expected values
            pack: Optional AnalysisPack reference
            covariance: Full covariance matrix (if correlated)
            nuisance_sigmas: Uncertainties for nuisance constraints
        """
        super().__init__(pack)
        self.observed = observed
        self.errors = errors
        self.model_func = model_func
        self.nuisance_sigmas = nuisance_sigmas if nuisance_sigmas is not None else np.array([])

        if covariance is not None:
            self.covariance = covariance
            self.use_full_covariance = True
            # Compute inverse and log determinant
            self._cov_inv = np.linalg.inv(covariance)
            self._cov_logdet = np.linalg.slogdet(covariance)[1]
        else:
            self.covariance = np.diag(errors**2)
            self.use_full_covariance = False
            self._cov_inv = np.diag(1.0 / errors**2)
            self._cov_logdet = 2 * np.sum(np.log(errors))

    def logpdf(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """Gaussian log-likelihood with optional correlations."""
        mu = self.model_func(theta, nu)
        residual = self.observed - mu

        # Multivariate Gaussian log-likelihood
        n = len(self.observed)
        ll = -0.5 * (n * np.log(2 * np.pi) + self._cov_logdet +
                     residual @ self._cov_inv @ residual)

        # Add constraint terms
        ll -= self.constraint_nll(nu)

        return ll

    def predict(self, theta: np.ndarray, nu: np.ndarray) -> ModelPrediction:
        """Compute expected values."""
        expected = self.model_func(theta, nu)
        return ModelPrediction(expected=expected)

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        """Assess Gaussian fit health using chi-squared."""
        mu = self.model_func(theta, nu)
        residual = self.observed - mu

        # Chi-squared = r^T C^-1 r
        chi2 = residual @ self._cov_inv @ residual

        n_bins = len(self.observed)
        n_params = len(theta) + len(nu)
        dof = max(1, n_bins - n_params)

        health = FitHealth(
            chi2=chi2,
            dof=dof,
            chi2_per_dof=chi2 / dof,
            deviance=chi2,  # For Gaussian, deviance = chi2
            deviance_per_dof=chi2 / dof
        )
        health.assess()
        return health

    def constraint_nll(self, nu: np.ndarray) -> float:
        """Standard normal constraints on nuisances."""
        if len(nu) == 0:
            return 0.0

        nll = 0.0
        for i, n_val in enumerate(nu):
            sigma = self.nuisance_sigmas[i] if i < len(self.nuisance_sigmas) else 1.0
            nll += 0.5 * (n_val / sigma)**2

        return nll


class HybridLikelihood(LikelihoodModel):
    """
    Hybrid likelihood combining Poisson and Gaussian components.

    Typically used for:
    - Poisson counts in signal/control regions
    - Gaussian constraints on systematics
    """

    def __init__(
        self,
        poisson_channels: List[PoissonLikelihood],
        gaussian_channels: List[GaussianLikelihood],
        pack: Any = None
    ):
        """
        Args:
            poisson_channels: List of Poisson likelihood components
            gaussian_channels: List of Gaussian likelihood components
            pack: Optional AnalysisPack reference
        """
        super().__init__(pack)
        self.poisson_channels = poisson_channels
        self.gaussian_channels = gaussian_channels

    def logpdf(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """Combined log-likelihood from all channels."""
        ll = 0.0

        for ch in self.poisson_channels:
            ll += ch.logpdf(theta, nu)

        for ch in self.gaussian_channels:
            ll += ch.logpdf(theta, nu)

        return ll

    def predict(self, theta: np.ndarray, nu: np.ndarray) -> ModelPrediction:
        """Aggregate predictions from all channels."""
        all_expected = []

        for ch in self.poisson_channels:
            pred = ch.predict(theta, nu)
            all_expected.append(pred.expected)

        for ch in self.gaussian_channels:
            pred = ch.predict(theta, nu)
            all_expected.append(pred.expected)

        return ModelPrediction(expected=np.concatenate(all_expected))

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        """Combined fit health assessment."""
        total_chi2 = 0.0
        total_deviance = 0.0
        total_dof = 0

        for ch in self.poisson_channels:
            h = ch.assess_fit_health(theta, nu)
            total_chi2 += h.chi2
            total_deviance += h.deviance
            total_dof += h.dof

        for ch in self.gaussian_channels:
            h = ch.assess_fit_health(theta, nu)
            total_chi2 += h.chi2
            total_deviance += h.deviance
            total_dof += h.dof

        total_dof = max(1, total_dof)

        health = FitHealth(
            chi2=total_chi2,
            dof=total_dof,
            chi2_per_dof=total_chi2 / total_dof,
            deviance=total_deviance,
            deviance_per_dof=total_deviance / total_dof
        )
        health.assess()
        return health


class BreitWignerModel:
    """
    Two-resonance Breit-Wigner amplitude model.

    Adapted from publication-grade rank-1 test implementation.
    Used for particle resonance fitting.
    """

    def __init__(
        self,
        M1: float = 9.4,
        G1: float = 0.02,
        M2: float = 10.0,
        G2: float = 0.05
    ):
        """
        Args:
            M1, G1: Mass and width of first resonance (GeV)
            M2, G2: Mass and width of second resonance (GeV)
        """
        self.M1 = M1
        self.G1 = G1
        self.M2 = M2
        self.G2 = G2

    def breit_wigner(self, m: np.ndarray, M: float, Gamma: float) -> np.ndarray:
        """Relativistic Breit-Wigner amplitude."""
        s = m**2
        M2 = M**2
        return M * Gamma / (M2 - s + 1j * M * Gamma)

    def amplitude(
        self,
        m: np.ndarray,
        c1: float,
        r: float,
        phi: float,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Compute intensity from two-resonance interference.

        Args:
            m: Mass values (GeV)
            c1: Overall coupling
            r: Ratio magnitude |R| = |c2/c1|
            phi: Relative phase
            scale: Overall scale factor

        Returns:
            Intensity (|amplitude|^2 * scale)
        """
        R = r * np.exp(1j * phi)

        BW1 = self.breit_wigner(m, self.M1, self.G1)
        BW2 = self.breit_wigner(m, self.M2, self.G2)

        amp = c1 * (BW1 + R * BW2)
        intensity = np.abs(amp)**2 * scale

        return np.maximum(intensity, 1e-10)


def create_model(pack) -> LikelihoodModel:
    """
    Create a likelihood model from an analysis pack.

    Factory function that inspects the pack type and creates
    the appropriate likelihood model.

    Args:
        pack: AnalysisPack object

    Returns:
        LikelihoodModel instance
    """
    likelihood_type = pack.likelihood_type

    if likelihood_type == "poisson":
        return _create_poisson_model(pack)
    elif likelihood_type == "gaussian":
        return _create_gaussian_model(pack)
    elif likelihood_type == "hybrid":
        return _create_hybrid_model(pack)
    elif likelihood_type == "surrogate":
        try:
            from surrogates import load_surrogate
        except ImportError:
            from ..surrogates import load_surrogate
        return load_surrogate(pack)
    else:
        raise ValueError(f"Unsupported likelihood type: {likelihood_type}")


def _create_poisson_model(pack) -> PoissonLikelihood:
    """Create Poisson model from pack."""
    if not pack.channels:
        raise ValueError("Pack has no channel data")

    nuisance_names = pack.get_nuisance_names()
    nuisance_types = [p.constraint for p in pack.nuisances]
    nuisance_sigmas = np.array([p.sigma for p in pack.nuisances]) if pack.n_nuisances > 0 else np.array([])

    templates = _load_templates(pack.path)
    poisson_channels = []

    for ch_spec in pack.likelihood_spec.get("channels", []):
        ch_name = ch_spec["name"]
        if ch_name not in pack.channels:
            continue
        channel = pack.channels[ch_name]

        signal, background = _extract_yields(pack, ch_spec, channel, templates)
        effects = _extract_systematics(ch_spec, templates)

        def model_func(theta, nu, signal=signal, background=background, effects=effects):
            mu = theta[0] if len(theta) > 0 else 1.0
            signal_scale = 1.0
            background_scale = 1.0
            signal_shape = signal.copy()
            background_shape = background.copy()

            for i, name in enumerate(nuisance_names):
                if name not in effects:
                    continue
                effect = effects[name]
                sigma = effect["sigma"]
                constraint = effect["constraint"]
                target = effect["target"]

                if effect["type"] == "shape":
                    alpha = nu[i] * sigma
                    delta = effect["delta"]
                    if target in ("signal", "both"):
                        signal_shape = signal_shape + alpha * delta
                    if target in ("background", "both"):
                        background_shape = background_shape + alpha * delta
                    continue

                if constraint == "lognormal":
                    factor = np.exp(nu[i] * sigma)
                else:
                    factor = 1.0 + nu[i] * sigma

                if target == "signal":
                    signal_scale *= factor
                elif target == "background":
                    background_scale *= factor
                elif target == "both":
                    signal_scale *= factor
                    background_scale *= factor

            signal_scaled = np.maximum(signal_shape * signal_scale, 0.0)
            background_scaled = np.maximum(background_shape * background_scale, 0.0)
            expected = mu * signal_scaled + background_scaled
            return np.maximum(expected, 0.0)

        model = PoissonLikelihood(
            observed=channel.observed,
            model_func=model_func,
            pack=pack,
            nuisance_sigmas=nuisance_sigmas,
            nuisance_types=nuisance_types
        )
        _apply_pack_metadata(model, pack)
        poisson_channels.append(model)

    if len(poisson_channels) == 1:
        return poisson_channels[0]

    combined = HybridLikelihood(poisson_channels=poisson_channels, gaussian_channels=[], pack=pack)
    _apply_pack_metadata(combined, pack)
    return combined


def _create_gaussian_model(pack) -> GaussianLikelihood:
    """Create Gaussian model from pack."""
    if not pack.channels:
        raise ValueError("Pack has no channel data")

    channel_name = list(pack.channels.keys())[0]
    channel = pack.channels[channel_name]

    def model_func(theta, nu):
        mu = theta[0] if len(theta) > 0 else 1.0
        return channel.observed * mu

    nuisance_sigmas = np.ones(pack.n_nuisances)

    model = GaussianLikelihood(
        observed=channel.observed,
        errors=channel.errors,
        model_func=model_func,
        pack=pack,
        nuisance_sigmas=nuisance_sigmas
    )
    _apply_pack_metadata(model, pack)
    return model


def _create_hybrid_model(pack) -> HybridLikelihood:
    """Create hybrid model from pack."""
    poisson_channels = []
    gaussian_channels = []

    for ch_spec in pack.likelihood_spec.get("channels", []):
        ch_name = ch_spec["name"]
        ch_type = ch_spec.get("type", "poisson")

        if ch_name in pack.channels:
            channel = pack.channels[ch_name]

            def model_func(theta, nu, ch=channel):
                mu = theta[0] if len(theta) > 0 else 1.0
                return ch.observed * mu

            if ch_type == "poisson":
                poisson_channels.append(PoissonLikelihood(
                    observed=channel.observed,
                    model_func=model_func,
                    pack=pack
                ))
            else:
                gaussian_channels.append(GaussianLikelihood(
                    observed=channel.observed,
                    errors=channel.errors,
                    model_func=model_func,
                    pack=pack
                ))

    model = HybridLikelihood(
        poisson_channels=poisson_channels,
        gaussian_channels=gaussian_channels,
        pack=pack
    )
    _apply_pack_metadata(model, pack)
    return model


def _apply_pack_metadata(model: LikelihoodModel, pack: Any) -> None:
    """Attach parameter metadata from pack to a model instance."""
    model._poi_names = pack.get_poi_names()
    model._nuisance_names = pack.get_nuisance_names()
    model._poi_bounds = [tuple(p.range) for p in pack.pois]
    model._nuisance_bounds = [tuple(p.range) for p in pack.nuisances]


def _extract_systematics(
    channel_spec: Dict[str, Any],
    templates: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, Dict[str, Any]]:
    """Extract systematic effects from channel specification."""
    effects: Dict[str, Dict[str, Any]] = {}
    for syst in channel_spec.get("systematics", []):
        name = syst.get("name")
        if not name:
            continue
        effect_type = syst.get("type", "norm")
        delta = None
        if effect_type == "shape":
            up_name = syst.get("up")
            down_name = syst.get("down")
            if not up_name or not down_name:
                raise ValueError(f"Shape systematic '{name}' missing up/down templates")
            up = _resolve_template(up_name, templates)
            down = _resolve_template(down_name, templates)
            delta = 0.5 * (up - down)
        effects[name] = {
            "type": effect_type,
            "sigma": float(syst.get("sigma", 0.0)),
            "constraint": syst.get("constraint", "normal"),
            "target": syst.get("target", "both"),
            "delta": delta
        }
    return effects


def _extract_yields(
    pack: Any,
    channel_spec: Dict[str, Any],
    channel: Any,
    templates: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract signal/background yields for a channel."""
    model_spec = channel_spec.get("model", {})
    signal_spec = model_spec.get("signal", {})
    background_spec = model_spec.get("background", {})

    signal = _load_yields(pack.path, signal_spec, templates) if signal_spec else None
    background = _load_yields(pack.path, background_spec, templates) if background_spec else None

    if signal is None and background is None:
        signal = channel.observed
        background = np.zeros_like(channel.observed)
    elif signal is None:
        signal = np.zeros_like(background)
    elif background is None:
        background = np.zeros_like(signal)

    if len(signal) != len(channel.observed) or len(background) != len(channel.observed):
        raise ValueError(f"Yield length mismatch for channel {channel.name}")

    return signal, background


def _load_templates(pack_path: Path) -> Dict[str, np.ndarray]:
    templates_path = pack_path / "model" / "templates.json"
    if not templates_path.exists():
        return {}
    with open(templates_path, "r") as f:
        payload = json.load(f)
    return {name: np.array(values, dtype=float) for name, values in payload.get("templates", {}).items()}


def _resolve_template(name: str, templates: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if not templates or name not in templates:
        raise ValueError(f"Template '{name}' not found in model/templates.json")
    return templates[name]


def _load_yields(
    pack_path: Path,
    spec: Dict[str, Any],
    templates: Optional[Dict[str, np.ndarray]] = None
) -> Optional[np.ndarray]:
    """Load yield arrays from a model spec."""
    if not spec:
        return None
    if "values" in spec:
        return np.array(spec["values"], dtype=float)
    if "template" in spec:
        return _resolve_template(spec["template"], templates)
    if "file" in spec:
        file_path = pack_path / "model" / spec["file"]
        if not file_path.exists():
            raise FileNotFoundError(f"Missing model file: {file_path}")
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1, dtype=float)
        if data.ndim == 1:
            return np.array([data[1]], dtype=float)
        return data[:, 1]
    return None
