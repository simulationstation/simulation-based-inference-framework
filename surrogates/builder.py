"""
Surrogate model builders for likelihood emulation.

Implements:
- Gaussian Process surrogates with uncertainty quantification
- Neural network emulators
- Parametric surrogates (splines, polynomials)

Training data generation using Latin hypercube sampling,
active learning around MLE, and stress testing.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path

try:
    from runtime.likelihoods import LikelihoodModel, FitHealth
except ImportError:
    from ..runtime.likelihoods import LikelihoodModel, FitHealth


@dataclass
class TrainingPoint:
    """Single training point for surrogate."""
    theta: np.ndarray
    nu: np.ndarray
    logL: float
    grad_theta: Optional[np.ndarray] = None
    grad_nu: Optional[np.ndarray] = None


@dataclass
class TrainingData:
    """Collection of training points."""
    points: List[TrainingPoint] = field(default_factory=list)
    theta_bounds: List[Tuple[float, float]] = field(default_factory=list)
    nu_bounds: List[Tuple[float, float]] = field(default_factory=list)

    def __len__(self):
        return len(self.points)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to input/output arrays."""
        X = np.array([np.concatenate([p.theta, p.nu]) for p in self.points])
        y = np.array([p.logL for p in self.points])
        return X, y


@dataclass
class SurrogateValidation:
    """Validation results for a surrogate."""
    mae: float = 0.0  # Mean absolute error
    rmse: float = 0.0  # Root mean squared error
    max_error: float = 0.0  # Maximum absolute error
    mle_error: float = 0.0  # Error in MLE location
    coverage_68: float = 0.0  # 68% interval coverage
    coverage_95: float = 0.0  # 95% interval coverage
    calibration_slope: float = 1.0  # Slope of calibration curve
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class SurrogateModel(LikelihoodModel):
    """Base class for surrogate likelihood models."""

    def __init__(self, original_model: Optional[LikelihoodModel] = None):
        super().__init__()
        self.original_model = original_model
        self.training_data: Optional[TrainingData] = None
        self.validation: Optional[SurrogateValidation] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def train(self, training_data: TrainingData) -> None:
        """Train the surrogate on training data."""
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self, theta: np.ndarray, nu: np.ndarray
    ) -> Tuple[float, float]:
        """
        Predict logL with uncertainty estimate.

        Returns:
            (mean_logL, std_logL)
        """
        pass

    def logpdf(self, theta: np.ndarray, nu: np.ndarray) -> float:
        """Evaluate surrogate log-likelihood."""
        mean, _ = self.predict_with_uncertainty(theta, nu)
        return mean

    def save(self, path: str) -> None:
        """Save surrogate to disk."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        """Load surrogate from disk."""
        raise NotImplementedError


class GPSurrogate(SurrogateModel):
    """
    Gaussian Process surrogate for log-likelihood.

    Provides uncertainty quantification through GP posterior variance.
    Suitable for low-to-moderate dimensional parameter spaces.
    """

    def __init__(
        self,
        original_model: Optional[LikelihoodModel] = None,
        kernel: str = "matern52",
        noise_variance: float = 1e-6
    ):
        super().__init__(original_model)
        self.kernel_type = kernel
        self.noise_variance = noise_variance

        # Will be set during training
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._length_scales: Optional[np.ndarray] = None
        self._signal_variance: float = 1.0

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if self._length_scales is None:
            self._length_scales = np.ones(X1.shape[1])

        # Scaled distance
        X1_scaled = X1 / self._length_scales
        X2_scaled = X2 / self._length_scales

        # Squared Euclidean distance
        sq_dist = (
            np.sum(X1_scaled**2, axis=1)[:, None] +
            np.sum(X2_scaled**2, axis=1)[None, :] -
            2 * X1_scaled @ X2_scaled.T
        )
        sq_dist = np.maximum(sq_dist, 0)  # Numerical stability

        if self.kernel_type == "rbf":
            K = self._signal_variance * np.exp(-0.5 * sq_dist)
        elif self.kernel_type == "matern52":
            r = np.sqrt(sq_dist)
            sqrt5 = np.sqrt(5)
            K = self._signal_variance * (1 + sqrt5 * r + 5/3 * sq_dist) * np.exp(-sqrt5 * r)
        elif self.kernel_type == "matern32":
            r = np.sqrt(sq_dist)
            sqrt3 = np.sqrt(3)
            K = self._signal_variance * (1 + sqrt3 * r) * np.exp(-sqrt3 * r)
        else:
            K = self._signal_variance * np.exp(-0.5 * sq_dist)

        return K

    def train(self, training_data: TrainingData) -> None:
        """Train GP on training data."""
        X, y = training_data.to_arrays()

        self._X_train = X
        self._y_train = y
        self.training_data = training_data

        n_samples, n_features = X.shape

        # Initialize length scales from data
        self._length_scales = np.std(X, axis=0)
        self._length_scales = np.maximum(self._length_scales, 1e-3)

        # Signal variance from output variance
        self._signal_variance = np.var(y)

        # Compute kernel matrix
        K = self._kernel(X, X)
        K += self.noise_variance * np.eye(n_samples)

        # Invert kernel matrix
        try:
            self._K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Regularize if singular
            K += 1e-4 * np.eye(n_samples)
            self._K_inv = np.linalg.inv(K)

        # Precompute alpha = K^-1 @ y
        self._alpha = self._K_inv @ y

        self._is_trained = True

    def predict_with_uncertainty(
        self, theta: np.ndarray, nu: np.ndarray
    ) -> Tuple[float, float]:
        """GP prediction with posterior variance."""
        if not self._is_trained:
            raise RuntimeError("Surrogate must be trained before prediction")

        x = np.concatenate([theta, nu]).reshape(1, -1)

        # Cross-covariance
        k_star = self._kernel(x, self._X_train)

        # Posterior mean
        mean = (k_star @ self._alpha).squeeze().item()

        # Posterior variance
        k_ss = self._kernel(x, x)[0, 0]
        var = k_ss - k_star @ self._K_inv @ k_star.T
        var = float(var[0, 0])
        std = np.sqrt(max(var, 0))

        return mean, std

    def predict(self, theta: np.ndarray, nu: np.ndarray):
        """Model prediction."""
        try:
            from runtime.likelihoods import ModelPrediction
        except ImportError:
            from ..runtime.likelihoods import ModelPrediction
        mean, _ = self.predict_with_uncertainty(theta, nu)
        return ModelPrediction(expected=np.array([mean]))

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        """Assess surrogate prediction quality."""
        _, std = self.predict_with_uncertainty(theta, nu)

        health = FitHealth()
        if std > 1.0:
            health.status = "HIGH_UNCERTAINTY"
            health.reason = f"Surrogate std = {std:.2f} > 1.0"
        else:
            health.status = "HEALTHY"
            health.reason = f"Surrogate std = {std:.2f}"

        return health

    def save(self, path: str) -> None:
        """Save GP surrogate to disk."""
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, "X_train.npy"), self._X_train)
        np.save(os.path.join(path, "y_train.npy"), self._y_train)
        np.save(os.path.join(path, "length_scales.npy"), self._length_scales)

        config = {
            "kernel_type": self.kernel_type,
            "noise_variance": self.noise_variance,
            "signal_variance": self._signal_variance
        }
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str) -> "GPSurrogate":
        """Load GP surrogate from disk."""
        with open(os.path.join(path, "config.json"), 'r') as f:
            config = json.load(f)

        surrogate = cls(
            kernel=config["kernel_type"],
            noise_variance=config["noise_variance"]
        )

        X_train = np.load(os.path.join(path, "X_train.npy"))
        y_train = np.load(os.path.join(path, "y_train.npy"))
        surrogate._length_scales = np.load(os.path.join(path, "length_scales.npy"))
        surrogate._signal_variance = config["signal_variance"]

        # Reconstruct training
        n_samples = len(y_train)
        training_data = TrainingData()
        # Simplified reconstruction
        surrogate._X_train = X_train
        surrogate._y_train = y_train
        surrogate._is_trained = True

        # Recompute kernel inverse
        K = surrogate._kernel(X_train, X_train)
        K += surrogate.noise_variance * np.eye(n_samples)
        surrogate._K_inv = np.linalg.inv(K)
        surrogate._alpha = surrogate._K_inv @ y_train

        return surrogate


class NeuralSurrogate(SurrogateModel):
    """
    Neural network surrogate for log-likelihood.

    Uses ensemble of networks for uncertainty quantification.
    Suitable for high-dimensional parameter spaces.
    """

    def __init__(
        self,
        original_model: Optional[LikelihoodModel] = None,
        hidden_layers: List[int] = None,
        n_ensemble: int = 5,
        learning_rate: float = 0.001,
        epochs: int = 1000
    ):
        super().__init__(original_model)
        self.hidden_layers = hidden_layers or [64, 64, 32]
        self.n_ensemble = n_ensemble
        self.learning_rate = learning_rate
        self.epochs = epochs

        self._models = []
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._output_mean: float = 0.0
        self._output_std: float = 1.0

    def _build_network(self, input_dim: int) -> Any:
        """Build a simple MLP using numpy (no external deps)."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_layers:
            # Xavier initialization
            W = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            layers.append({'W': W, 'b': b})
            prev_dim = hidden_dim

        # Output layer
        W = np.random.randn(prev_dim, 1) * np.sqrt(2.0 / prev_dim)
        b = np.zeros(1)
        layers.append({'W': W, 'b': b})

        return layers

    def _forward(self, x: np.ndarray, network: List[Dict]) -> np.ndarray:
        """Forward pass through network."""
        h = x
        for i, layer in enumerate(network[:-1]):
            h = h @ layer['W'] + layer['b']
            h = np.maximum(h, 0)  # ReLU

        # Output layer (no activation)
        out = h @ network[-1]['W'] + network[-1]['b']
        return out

    def train(self, training_data: TrainingData) -> None:
        """Train ensemble of neural networks."""
        X, y = training_data.to_arrays()

        # Normalize inputs and outputs
        self._input_mean = np.mean(X, axis=0)
        self._input_std = np.std(X, axis=0)
        self._input_std = np.maximum(self._input_std, 1e-6)

        self._output_mean = np.mean(y)
        self._output_std = np.std(y)
        self._output_std = max(self._output_std, 1e-6)

        X_norm = (X - self._input_mean) / self._input_std
        y_norm = (y - self._output_mean) / self._output_std

        input_dim = X.shape[1]

        # Train ensemble with different random seeds
        self._models = []
        for i in range(self.n_ensemble):
            np.random.seed(42 + i)
            network = self._build_network(input_dim)

            # Simple SGD training
            n_samples = len(X_norm)
            batch_size = min(32, n_samples)

            for epoch in range(self.epochs):
                # Shuffle
                perm = np.random.permutation(n_samples)
                X_shuf = X_norm[perm]
                y_shuf = y_norm[perm]

                for j in range(0, n_samples, batch_size):
                    X_batch = X_shuf[j:j+batch_size]
                    y_batch = y_shuf[j:j+batch_size]

                    # Forward
                    pred = self._forward(X_batch, network)

                    # MSE gradient
                    grad_out = 2 * (pred - y_batch.reshape(-1, 1)) / len(X_batch)

                    # Backprop (simplified)
                    self._backward(network, X_batch, grad_out)

            self._models.append(network)

        self.training_data = training_data
        self._is_trained = True

    def _backward(self, network: List[Dict], X: np.ndarray, grad_out: np.ndarray):
        """Simplified backpropagation."""
        # Forward pass storing activations
        activations = [X]
        h = X
        for i, layer in enumerate(network[:-1]):
            pre_act = h @ layer['W'] + layer['b']
            h = np.maximum(pre_act, 0)
            activations.append(h)

        # Backward pass
        grad = grad_out

        # Output layer
        network[-1]['W'] -= self.learning_rate * activations[-1].T @ grad
        network[-1]['b'] -= self.learning_rate * np.sum(grad, axis=0)
        grad = grad @ network[-1]['W'].T

        # Hidden layers
        for i in range(len(network) - 2, -1, -1):
            # ReLU gradient
            grad = grad * (activations[i+1] > 0)
            network[i]['W'] -= self.learning_rate * activations[i].T @ grad
            network[i]['b'] -= self.learning_rate * np.sum(grad, axis=0)
            grad = grad @ network[i]['W'].T

    def predict_with_uncertainty(
        self, theta: np.ndarray, nu: np.ndarray
    ) -> Tuple[float, float]:
        """Ensemble prediction with uncertainty."""
        if not self._is_trained:
            raise RuntimeError("Surrogate must be trained before prediction")

        x = np.concatenate([theta, nu]).reshape(1, -1)
        x_norm = (x - self._input_mean) / self._input_std

        predictions = []
        for network in self._models:
            pred = self._forward(x_norm, network)[0, 0]
            # Denormalize
            pred = pred * self._output_std + self._output_mean
            predictions.append(pred)

        mean = np.mean(predictions)
        std = np.std(predictions)

        return float(mean), float(std)

    def predict(self, theta: np.ndarray, nu: np.ndarray):
        try:
            from runtime.likelihoods import ModelPrediction
        except ImportError:
            from ..runtime.likelihoods import ModelPrediction
        mean, _ = self.predict_with_uncertainty(theta, nu)
        return ModelPrediction(expected=np.array([mean]))

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        _, std = self.predict_with_uncertainty(theta, nu)

        health = FitHealth()
        if std > 1.0:
            health.status = "HIGH_UNCERTAINTY"
            health.reason = f"Ensemble std = {std:.2f} > 1.0"
        else:
            health.status = "HEALTHY"
            health.reason = f"Ensemble std = {std:.2f}"

        return health


class ParametricSurrogate(SurrogateModel):
    """
    Parametric surrogate using polynomial or spline approximation.

    Suitable for low-dimensional parameter spaces with smooth likelihoods.
    """

    def __init__(
        self,
        original_model: Optional[LikelihoodModel] = None,
        order: int = 2,
        method: str = "polynomial"
    ):
        super().__init__(original_model)
        self.order = order
        self.method = method
        self._coefficients: Optional[np.ndarray] = None
        self._residual_std: float = 0.0

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Generate polynomial features up to given order."""
        n_samples, n_features = X.shape
        features = [np.ones((n_samples, 1))]  # Constant term

        for d in range(1, self.order + 1):
            for i in range(n_features):
                features.append(X[:, i:i+1]**d)

        # Cross terms for order 2
        if self.order >= 2:
            for i in range(n_features):
                for j in range(i+1, n_features):
                    features.append(X[:, i:i+1] * X[:, j:j+1])

        return np.hstack(features)

    def train(self, training_data: TrainingData) -> None:
        """Fit polynomial model."""
        X, y = training_data.to_arrays()

        # Generate polynomial features
        Phi = self._polynomial_features(X)

        # Least squares fit
        try:
            self._coefficients, residuals, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        except np.linalg.LinAlgError:
            # Regularized fit
            reg = 1e-4 * np.eye(Phi.shape[1])
            self._coefficients = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
            residuals = y - Phi @ self._coefficients

        # Estimate residual std for uncertainty
        predictions = Phi @ self._coefficients
        self._residual_std = np.std(y - predictions)

        self.training_data = training_data
        self._is_trained = True

    def predict_with_uncertainty(
        self, theta: np.ndarray, nu: np.ndarray
    ) -> Tuple[float, float]:
        """Polynomial prediction with residual uncertainty."""
        if not self._is_trained:
            raise RuntimeError("Surrogate must be trained before prediction")

        x = np.concatenate([theta, nu]).reshape(1, -1)
        Phi = self._polynomial_features(x)

        mean = (Phi @ self._coefficients).squeeze().item()
        std = self._residual_std

        return mean, std

    def predict(self, theta: np.ndarray, nu: np.ndarray):
        try:
            from runtime.likelihoods import ModelPrediction
        except ImportError:
            from ..runtime.likelihoods import ModelPrediction
        mean, _ = self.predict_with_uncertainty(theta, nu)
        return ModelPrediction(expected=np.array([mean]))

    def assess_fit_health(self, theta: np.ndarray, nu: np.ndarray) -> FitHealth:
        health = FitHealth()
        health.status = "HEALTHY"
        health.reason = f"Polynomial order {self.order}"
        return health


class SurrogateBuilder:
    """
    Builder for training and validating surrogate models.

    Handles:
    - Training data generation (LHS, active learning)
    - Model training
    - Validation and calibration
    - Error model construction
    """

    def __init__(
        self,
        model: LikelihoodModel,
        surrogate_type: str = "gp",
        **kwargs
    ):
        """
        Args:
            model: Original likelihood model to emulate
            surrogate_type: "gp", "neural", or "parametric"
            **kwargs: Surrogate-specific parameters
        """
        self.model = model
        self.surrogate_type = surrogate_type
        self.kwargs = kwargs

        self.training_data = TrainingData()
        self.validation_data = TrainingData()
        self.surrogate: Optional[SurrogateModel] = None

    def generate_training_data(
        self,
        n_samples: int = 500,
        method: str = "lhs",
        theta_bounds: Optional[List[Tuple[float, float]]] = None,
        nu_bounds: Optional[List[Tuple[float, float]]] = None,
        include_gradients: bool = False,
        seed: int = 42
    ) -> TrainingData:
        """
        Generate training data from the original model.

        Args:
            n_samples: Number of training points
            method: Sampling method ("lhs", "sobol", "random")
            theta_bounds: POI bounds for sampling
            nu_bounds: Nuisance bounds for sampling
            include_gradients: Also compute gradients
            seed: Random seed

        Returns:
            TrainingData object
        """
        np.random.seed(seed)

        # Get bounds
        poi_bounds, nuisance_bounds = self.model.get_bounds()
        if theta_bounds is not None:
            poi_bounds = theta_bounds
        if nu_bounds is not None:
            nuisance_bounds = nu_bounds

        # Default bounds if not specified
        if not poi_bounds:
            poi_bounds = [(-5, 5) for _ in range(self.model.n_pois)]
        if not nuisance_bounds:
            nuisance_bounds = [(-3, 3) for _ in range(self.model.n_nuisances)]

        n_theta = len(poi_bounds)
        n_nu = len(nuisance_bounds)
        n_total = n_theta + n_nu

        # Generate samples
        if method == "lhs":
            samples = self._latin_hypercube(n_samples, n_total)
        elif method == "sobol":
            samples = self._sobol_sequence(n_samples, n_total)
        else:
            samples = np.random.rand(n_samples, n_total)

        # Scale to bounds
        all_bounds = poi_bounds + nuisance_bounds
        for i, (lo, hi) in enumerate(all_bounds):
            samples[:, i] = lo + samples[:, i] * (hi - lo)

        # Evaluate model at each point
        data = TrainingData(
            theta_bounds=poi_bounds,
            nu_bounds=nuisance_bounds
        )

        for i in range(n_samples):
            theta = samples[i, :n_theta]
            nu = samples[i, n_theta:]

            logL = self.model.logpdf(theta, nu)

            grad_theta = None
            grad_nu = None
            if include_gradients:
                grad_theta, grad_nu = self.model.gradient(theta, nu)

            data.points.append(TrainingPoint(
                theta=theta,
                nu=nu,
                logL=logL,
                grad_theta=grad_theta,
                grad_nu=grad_nu
            ))

        self.training_data = data
        return data

    def _latin_hypercube(self, n: int, d: int) -> np.ndarray:
        """Generate Latin Hypercube samples."""
        samples = np.zeros((n, d))
        for i in range(d):
            perm = np.random.permutation(n)
            samples[:, i] = (perm + np.random.rand(n)) / n
        return samples

    def _sobol_sequence(self, n: int, d: int) -> np.ndarray:
        """Generate Sobol sequence (simplified fallback to LHS)."""
        # Full Sobol would require additional library
        return self._latin_hypercube(n, d)

    def build(
        self,
        training_data: Optional[TrainingData] = None,
        validation_split: float = 0.2
    ) -> SurrogateModel:
        """
        Build and train surrogate model.

        Args:
            training_data: Training data (uses stored if None)
            validation_split: Fraction for validation

        Returns:
            Trained SurrogateModel
        """
        if training_data is None:
            training_data = self.training_data

        if len(training_data) == 0:
            raise ValueError("No training data available")

        # Split into train/validation
        n_total = len(training_data)
        n_val = int(n_total * validation_split)
        n_train = n_total - n_val

        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_data = TrainingData(
            points=[training_data.points[i] for i in train_indices],
            theta_bounds=training_data.theta_bounds,
            nu_bounds=training_data.nu_bounds
        )

        val_data = TrainingData(
            points=[training_data.points[i] for i in val_indices],
            theta_bounds=training_data.theta_bounds,
            nu_bounds=training_data.nu_bounds
        )

        self.validation_data = val_data

        # Create surrogate
        if self.surrogate_type == "gp":
            self.surrogate = GPSurrogate(
                original_model=self.model,
                **self.kwargs
            )
        elif self.surrogate_type == "neural":
            self.surrogate = NeuralSurrogate(
                original_model=self.model,
                **self.kwargs
            )
        elif self.surrogate_type == "parametric":
            self.surrogate = ParametricSurrogate(
                original_model=self.model,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown surrogate type: {self.surrogate_type}")

        # Train
        self.surrogate.train(train_data)

        # Validate
        self.surrogate.validation = self.validate(val_data)

        return self.surrogate

    def validate(self, validation_data: Optional[TrainingData] = None) -> SurrogateValidation:
        """
        Validate surrogate against held-out data.

        Args:
            validation_data: Validation data (uses stored if None)

        Returns:
            SurrogateValidation results
        """
        if validation_data is None:
            validation_data = self.validation_data

        if self.surrogate is None or not self.surrogate.is_trained:
            raise RuntimeError("Surrogate must be trained before validation")

        errors = []
        uncertainties = []
        z_scores = []

        for point in validation_data.points:
            mean, std = self.surrogate.predict_with_uncertainty(point.theta, point.nu)
            error = mean - point.logL
            errors.append(error)
            uncertainties.append(std)

            if std > 0:
                z_scores.append(error / std)

        errors = np.array(errors)
        uncertainties = np.array(uncertainties)
        z_scores = np.array(z_scores)

        # Compute metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))

        # Coverage (assuming normal errors)
        if len(z_scores) > 0:
            coverage_68 = np.mean(np.abs(z_scores) < 1.0)
            coverage_95 = np.mean(np.abs(z_scores) < 2.0)
        else:
            coverage_68 = 0.0
            coverage_95 = 0.0

        # Pass if max error < 0.1 and coverage is reasonable
        passed = max_error < 0.1 and coverage_68 > 0.5

        return SurrogateValidation(
            mae=mae,
            rmse=rmse,
            max_error=max_error,
            coverage_68=coverage_68,
            coverage_95=coverage_95,
            passed=passed,
            details={
                'n_validation': len(validation_data),
                'mean_uncertainty': np.mean(uncertainties),
                'z_score_mean': np.mean(z_scores) if len(z_scores) > 0 else 0.0,
                'z_score_std': np.std(z_scores) if len(z_scores) > 0 else 1.0
            }
        )


def load_surrogate(pack) -> SurrogateModel:
    """Load a surrogate model from an analysis pack."""
    surrogate_path = pack.path / "surrogate"

    if not surrogate_path.exists():
        raise FileNotFoundError(f"No surrogate directory in pack: {pack.path}")

    config_path = surrogate_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

        surrogate_type = config.get("type", "gp")

        if surrogate_type == "gp":
            return GPSurrogate.load(str(surrogate_path))
        elif surrogate_type == "neural":
            return NeuralSurrogate.load(str(surrogate_path))
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")
    else:
        # Try to infer from files
        if (surrogate_path / "X_train.npy").exists():
            return GPSurrogate.load(str(surrogate_path))

    raise FileNotFoundError(f"Could not load surrogate from: {surrogate_path}")
