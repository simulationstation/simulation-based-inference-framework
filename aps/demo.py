"""
Vertical-slice demo runner for APS packs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import aps
from runtime.inference import compute_mle, profile_likelihood


def run_demo(
    pack_path: str,
    out_dir: str,
    seed: int = 123,
    scan_min: float = 0.0,
    scan_max: float = 2.0,
    scan_steps: int = 21,
    n_starts: int = 50
) -> dict[str, Any]:
    """Run the vertical-slice demo and write artifacts."""
    np.random.seed(seed)

    pack = aps.load(pack_path, validate=True)
    model = pack.model()

    result = compute_mle(model, n_starts=n_starts)
    if not result.converged:
        raise RuntimeError("MLE fit failed")

    scan_mu = np.linspace(scan_min, scan_max, scan_steps)
    nll_mle = result.nll
    scan_rows = []

    for mu in scan_mu:
        theta = np.array([mu])
        profiled = profile_likelihood(model, theta=theta, nu_init=result.nu_hat, n_starts=20)
        delta_nll = profiled.nll_profiled - nll_mle
        qmu = max(0.0, 2.0 * delta_nll)
        scan_rows.append({
            "mu": float(mu),
            "delta_nll": float(delta_nll),
            "qmu": float(qmu)
        })

    mu_hat = float(result.theta_hat[0]) if result.theta_hat.size else float("nan")
    mu_uncertainty = _estimate_uncertainty(scan_rows, target_q=1.0)
    interval_95 = _estimate_interval(scan_rows, target_q=3.84)

    nuisances = []
    for idx, nuisance in enumerate(pack.nuisances):
        pull = result.nu_hat[idx] / nuisance.sigma if nuisance.sigma != 0 else float("nan")
        nuisances.append({
            "name": nuisance.name,
            "value": float(result.nu_hat[idx]),
            "sigma": float(nuisance.sigma),
            "pull": float(pull)
        })

    git_hash = _get_git_hash()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {
        "pack": {
            "name": pack.name,
            "version": pack.version,
            "path": str(Path(pack_path).as_posix())
        },
        "run": {
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_hash": git_hash
        },
        "fit": {
            "mu_hat": mu_hat,
            "nll": float(nll_mle),
            "nu_hat": [float(x) for x in result.nu_hat],
            "fit_health": asdict(result.fit_health) if result.fit_health else None
        },
        "scan": {
            "mu": [row["mu"] for row in scan_rows],
            "delta_nll": [row["delta_nll"] for row in scan_rows],
            "qmu": [row["qmu"] for row in scan_rows]
        },
        "summary": {
            "mu_uncertainty": mu_uncertainty,
            "interval_95": interval_95
        },
        "nuisances": nuisances
    }

    _write_scan_csv(out_path / "scan.csv", scan_rows)
    _write_results(out_path / "results.json", results)
    _write_report(out_path / "REPORT.md", pack, scan_rows, results, nuisances)

    return results


def _estimate_uncertainty(
    scan_rows: list[dict[str, float]],
    target_q: float = 1.0
) -> float | None:
    interval = _estimate_interval(scan_rows, target_q)
    if interval is None:
        return None
    lower, upper = interval
    return 0.5 * (upper - lower)


def _estimate_interval(
    scan_rows: list[dict[str, float]],
    target_q: float = 3.84
) -> tuple[float, float] | None:
    sorted_rows = sorted(scan_rows, key=lambda x: x["mu"])
    mu_vals = np.array([row["mu"] for row in sorted_rows])
    q_vals = np.array([row["qmu"] for row in sorted_rows])

    below = q_vals <= target_q
    if not np.any(below):
        return None

    indices = np.where(below)[0]
    lower_idx = indices[0]
    upper_idx = indices[-1]

    lower = _interpolate_crossing(mu_vals, q_vals, lower_idx, direction=-1, target=target_q)
    upper = _interpolate_crossing(mu_vals, q_vals, upper_idx, direction=1, target=target_q)

    return (lower, upper)


def _interpolate_crossing(
    mu_vals: np.ndarray,
    q_vals: np.ndarray,
    idx: int,
    direction: int,
    target: float
) -> float:
    neighbor = idx + direction
    if neighbor < 0 or neighbor >= len(mu_vals):
        return float(mu_vals[idx])

    x0, y0 = mu_vals[idx], q_vals[idx]
    x1, y1 = mu_vals[neighbor], q_vals[neighbor]
    if y1 == y0:
        return float(x0)
    slope = (y1 - y0) / (x1 - x0)
    return float(x0 + (target - y0) / slope)


def _write_scan_csv(path: Path, scan_rows: list[dict[str, float]]) -> None:
    with open(path, "w") as f:
        f.write("mu,delta_nll,qmu\n")
        for row in scan_rows:
            f.write(f"{row['mu']},{row['delta_nll']},{row['qmu']}\n")


def _write_results(path: Path, results: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _write_report(
    path: Path,
    pack: aps.loaders.AnalysisPack,
    scan_rows: list[dict[str, float]],
    results: dict[str, Any],
    nuisances: list[dict[str, float]]
) -> None:
    total_counts = 0
    n_bins = 0
    for channel in pack.channels.values():
        total_counts += float(np.sum(channel.observed))
        n_bins += int(channel.nbins)

    mu_hat = results["fit"]["mu_hat"]
    mu_uncertainty = results["summary"]["mu_uncertainty"]
    interval_95 = results["summary"]["interval_95"]

    with open(path, "w") as f:
        f.write("# APS Demo Report\n\n")
        f.write("## Pack\n")
        f.write(f"- Name: {pack.name}\n")
        f.write(f"- Version: {pack.version}\n")
        f.write(f"- Path: {pack.path.as_posix()}\n\n")

        f.write("## Dataset Summary\n")
        f.write(f"- Channels: {len(pack.channels)}\n")
        f.write(f"- Total bins: {n_bins}\n")
        f.write(f"- Total observed counts: {int(total_counts)}\n\n")

        f.write("## Fit Summary\n")
        if mu_uncertainty is not None:
            f.write(f"- Best-fit mu: {mu_hat:.4f} ± {mu_uncertainty:.4f} (approx. 68% CL)\n")
        else:
            f.write(f"- Best-fit mu: {mu_hat:.4f}\n")
        if interval_95 is not None:
            f.write(
                f"- Approx. 95% CL interval: [{interval_95[0]:.4f}, {interval_95[1]:.4f}]\n"
            )
        f.write("\n")

        f.write("## Nuisance Parameters (profiled)\n")
        f.write("| Name | Value | Pull |\n")
        f.write("| --- | --- | --- |\n")
        for entry in nuisances:
            f.write(f"| {entry['name']} | {entry['value']:.4f} | {entry['pull']:.2f} |\n")
        f.write("\n")

        f.write("## Notes\n")
        f.write("- Profile likelihood scan over mu with Gaussian nuisance constraints.\n")
        f.write("- Asymptotic chi-square thresholds used for interval estimates.\n")


def _get_git_hash() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
