# REVIEW

## What’s already good
- The repository has a clear specification document (`spec.tex`) and a starter schema implementation in `aps/schemas.py`.
- A minimal runtime exists with Poisson/Gaussian/Hybrid likelihoods, basic profiling/limits, and a CLI scaffold.
- There is at least one example pack with the expected directory structure.

## What’s missing
- A reproducible, end-to-end “vertical slice” demo that produces `out/` artifacts and a short report.
- A canonical toy analysis pack that is fully executable without relying on hidden model formulas.
- CI enforcement for linting, tests, and a demo smoke run.
- A project roadmap and a formal review of where the repo diverges from the spec.
- Pre-commit hooks and contribution basics.

## What’s incorrect or misleading
- The README advertises features (e.g., model formulas, validated surrogates, fit health diagnostics with thresholds) that are not wired into the runtime.
- The existing pack references parametric formulas (Breit–Wigner interference) that the runtime does not parse or evaluate.
- Coverage/validation claims are not backed by a runnable harness that compares against expected results.

## Highest-risk items (scientific + software)
1. **Scientific integrity risk:** Surrogate/SBI claims are present but not backed by calibration or coverage checks that actually run.
2. **Reproducibility risk:** There is no CI enforcement; regressions can slip in unnoticed.
3. **Spec/runtime divergence:** The pack format in `spec.tex` is richer than what the runtime can execute.
4. **Misleading documentation:** Users could believe the provided pack is executable when it is not.
5. **Nuisance modeling ambiguity:** Systematics are not consistently represented between metadata and runtime.

## Next 10 tasks (priority order)
1. Add a fully executable toy Analysis Pack with explicit signal/background yields and nuisance effects.
2. Implement a vertical-slice demo that loads a pack, fits, scans a POI, and writes results + report to `out/`.
3. Add CI: lint, tests, and demo smoke test.
4. Tighten schema validation for model/yield specifications and nuisance effect targets.
5. Add a validation harness that compares demo outputs to golden values with tolerances.
6. Add an “honesty header” to documentation clarifying limitations of public data and surrogate claims.
7. Implement correlation handling for combined models or explicitly mark it unsupported.
8. Improve `coverage_test` to generate pseudo-data and re-fit (currently a placeholder).
9. Add a minimal data catalog with provenance/sha256 if any real public data is introduced.
10. Add user-facing API docs for pack creation and model evaluation.
