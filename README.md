# Analysis Pack Standard (APS)

**Reusable Collider Analyses via Surrogate Likelihoods and Simulation-Based Inference at Scale**

## Overview

APS provides a standard, machine-executable representation of collider physics analysis likelihoods with:
- Clear provenance and metadata
- Explicit nuisance parameter definitions and correlations
- Versioned validation tests reproducing published results
- Optional surrogate models with quantified approximation error

## Installation

```bash
pip install -e .

# With all optional dependencies
pip install -e ".[full,dev]"
```

## Quick Start

### Loading an Analysis Pack

```python
import aps

# Load a pack
pack = aps.load("packs/cms_x6900_dijpsi")

# Get model information
print(f"Name: {pack.name}")
print(f"Likelihood type: {pack.likelihood_type}")
print(f"POIs: {pack.get_poi_names()}")

# Create executable model
model = pack.model()

# Evaluate log-likelihood
ll = model.logpdf(theta=[1.0], nu=[0.0, 0.0, 0.0])
```

### Computing Limits

```python
from runtime import compute_mle, upper_limit, confidence_interval

# Find MLE
mle = compute_mle(model, n_starts=50)
print(f"Best-fit mu: {mle.theta_hat[0]:.4f}")

# Compute 95% CL upper limit
limit = upper_limit(model, poi_index=0, cl=0.95)
print(f"95% CL limit: {limit.observed:.4f}")

# Compute confidence interval
lower, upper = confidence_interval(model, poi_index=0, cl=0.68)
print(f"68% CI: [{lower:.4f}, {upper:.4f}]")
```

### CLI Usage

```bash
# Validate a pack
aps validate packs/cms_x6900_dijpsi

# Fit a model
aps fit packs/cms_x6900_dijpsi --poi mu_signal --starts 100

# Compute limit
aps limit packs/cms_x6900_dijpsi --poi mu_signal --cl 0.95

# Combine analyses
aps combine packs/pack1 packs/pack2 --correlations corr.yaml

# Export data
aps export packs/cms_x6900_dijpsi --format json -o output.json
```

## Analysis Pack Structure

```
my_analysis_pack/
  metadata.yaml       # Name, provenance, parameters, citations
  LICENSE             # License for pack content
  data/
    observed.csv      # Observed data
    binning.json      # Binning definitions
  model/
    likelihood.json   # Likelihood specification
    schema.json       # Parameter ordering
    constraints.json  # Nuisance correlations (optional)
  validation/
    reproduce.py      # Reproduction script
    expected_results.json
    tolerances.yaml
  surrogate/          # Optional: trained surrogate
    config.json
    weights.npy
  env/
    requirements.txt
```

## Likelihood Types

- **Poisson**: Count data with Poisson statistics
- **Gaussian**: Continuous measurements with covariance
- **Hybrid**: Combination of Poisson and Gaussian channels
- **Surrogate**: Emulated likelihood (GP, neural network, parametric)

## Surrogate Models

Build fast emulators for expensive likelihoods:

```python
from surrogates import SurrogateBuilder

builder = SurrogateBuilder(model, surrogate_type='gp')

# Generate training data
builder.generate_training_data(n_samples=500, method='lhs')

# Train and validate
surrogate = builder.build()

print(f"Validation MAE: {surrogate.validation.mae:.4f}")
print(f"68% coverage: {surrogate.validation.coverage_68:.2f}")
```

## Fit Health Diagnostics

All fits include health checks:

```python
health = model.assess_fit_health(theta, nu)

print(f"Status: {health.status}")  # HEALTHY, UNDERCONSTRAINED, MODEL_MISMATCH
print(f"chi2/dof: {health.chi2_per_dof:.2f}")
```

Thresholds:
- chi2/dof < 0.5: UNDERCONSTRAINED
- chi2/dof in [0.5, 3.0]: HEALTHY
- chi2/dof > 3.0: MODEL_MISMATCH

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=aps --cov=runtime --cov=surrogates

# Run specific test file
pytest tests/test_likelihoods.py -v
```

## Contributing

See CONTRIBUTING.md for guidelines.

## License

MIT License. See LICENSE file.

## Citation

If you use APS in your research, please cite:
```
@software{aps2026,
  title = {Analysis Pack Standard: Reusable Collider Analyses},
  year = {2026},
  url = {https://github.com/analysis-pack-standard/aps}
}
```

## References

- Spec document: `spec.tex`
- Example packs: `packs/`
- API documentation: `docs/`
