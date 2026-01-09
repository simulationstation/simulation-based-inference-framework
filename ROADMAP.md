# Roadmap

## Phase 1: Template-likelihood packs (now)
- Ship executable template packs with deterministic validation harnesses.
- Enforce lint/tests/demo runs in CI.
- Support basic profile-likelihood scans and reporting.
- Add combination support later (correlations explicitly modeled).

## Phase 2: Surrogate distillation + calibration
- Train surrogate likelihoods with explicit error models.
- Require calibration/coverage checks before any surrogate claims.
- Add surrogate trust-region metadata in packs.

## Phase 3: SBI demonstrator (only with credible public models)
- Provide a minimal SBI pack only where a public forward model is available.
- Include calibration/coverage tests for the SBI pipeline.
