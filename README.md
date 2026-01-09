# Analysis Pack Standard (APS)

**Reusable Collider Analyses via Surrogate Likelihoods and Simulation-Based Inference at Scale**

## Honesty Header

**Implemented now**
- Executable Poisson and Gaussian likelihoods with numeric templates and Gaussian-constrained nuisances.
- A deterministic vertical-slice demo that writes `REPORT.md`, `results.json`, and `scan.csv`.
- Schema validation for required files and supported template/nuisance targets.

**Planned**
- Template-likelihood combinations with correlations.
- Surrogate distillation with calibration/coverage tests.
- SBI demonstrators only where a credible forward model is public.

**Not validated (yet)**
- Surrogate/SBI accuracy or coverage guarantees.
- Fit health thresholds as publication-grade diagnostics.

## Overview

APS provides a standard, machine-executable representation of collider physics analysis likelihoods with:
- Clear provenance and metadata
- Explicit nuisance parameter definitions and correlations
- Versioned validation tests reproducing published results
- Optional surrogate scaffolding (calibration work is pending)

## Installation

```bash
pip install -e .

# With all optional dependencies
pip install -e ".[full,dev]"
```

## Quick Start

> Note: `packs/cms_x6900_dijpsi` contains formula strings that the runtime does not parse yet.
> Use `packs/toy_poisson_template_v1` for an executable example.

### Loading an Analysis Pack

```python
import aps

# Load a pack
pack = aps.load("packs/toy_poisson_template_v1")

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
aps validate packs/toy_poisson_template_v1

# Fit a model
aps fit packs/toy_poisson_template_v1 --poi mu --starts 100

# Compute limit
aps limit packs/toy_poisson_template_v1 --poi mu --cl 0.95

# Run vertical-slice demo
aps demo --pack packs/toy_poisson_template_v1 --out out/toy_demo_v1 --seed 123

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
    templates.json    # Numeric templates (optional)
    nuisances.json    # Nuisance mapping (optional)
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

Surrogate builders are included, but calibration/coverage validation is not yet end-to-end.
Use with caution until validation hooks are implemented.

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

All fits include heuristic health checks:

```python
health = model.assess_fit_health(theta, nu)

print(f"Status: {health.status}")  # HEALTHY, UNDERCONSTRAINED, MODEL_MISMATCH
print(f"chi2/dof: {health.chi2_per_dof:.2f}")
```

Thresholds are heuristics, not validated guarantees.

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
