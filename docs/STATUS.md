# APS Status Snapshot

## Truth table (runtime capabilities vs. pack content)

| Item | Status | Notes |
| --- | --- | --- |
| Poisson count likelihoods with numeric templates | ✅ Executable | Uses `runtime/likelihoods.py` with CSV/JSON templates. |
| Gaussian likelihoods (diagonal or full covariance) | ✅ Executable | Implemented in `runtime/likelihoods.py`. |
| Hybrid Poisson+Gaussian models | ✅ Executable | Combines channels in `runtime/likelihoods.py`. |
| Parametric formula strings (e.g., Breit–Wigner) | ❌ Not executable | No formula parser; packs using `formula` are rejected by validation. |
| Surrogate/SBI packs with calibration/coverage | ❌ Not validated | Surrogate scaffolding exists but is not calibrated end-to-end. |
| Coverage guarantees with toy refits | ⚠️ Minimal smoke coverage | Only a lightweight toy-coverage harness is provided. |

## What the toy pack exercises

| Pack | Exercised features |
| --- | --- |
| `packs/toy_poisson_template_v1` | Poisson counts, one POI (`mu`), three Gaussian nuisances (global norm, background norm, background shape morph), profile-likelihood scan, deterministic demo outputs. |
