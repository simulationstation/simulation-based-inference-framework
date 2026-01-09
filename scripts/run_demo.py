#!/usr/bin/env python3
"""Run a minimal vertical-slice demo for APS."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aps
from runtime import compute_mle, profile_likelihood, upper_limit


def run_demo(pack_path: str, out_dir: Path, poi_index: int, n_points: int) -> dict:
    pack = aps.load(pack_path)
    model = pack.model()

    poi_bounds, _ = model.get_bounds()
    if poi_bounds and poi_index < len(poi_bounds):
        scan_min, scan_max = poi_bounds[poi_index]
    else:
        scan_min, scan_max = (0.0, 5.0)

    mle = compute_mle(model, n_starts=50)
    limit = upper_limit(model, poi_index=poi_index, cl=0.95, n_starts=50)

    scan_values = np.linspace(scan_min, scan_max, n_points)
    scan = []
    for val in scan_values:
        theta = np.zeros(model.n_pois)
        theta[poi_index] = val
        profile = profile_likelihood(model, theta=theta, n_starts=25)
        scan.append({
            "poi": float(val),
            "nll": float(profile.nll_profiled),
            "delta_nll": float(profile.nll_profiled - mle.nll)
        })

    results = {
        "pack": pack.name,
        "version": pack.version,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "poi": model.poi_names[poi_index] if model.poi_names else f"poi_{poi_index}",
        "mle": {
            "theta": mle.theta_hat.tolist(),
            "nu": mle.nu_hat.tolist(),
            "nll": float(mle.nll)
        },
        "limit_95": float(limit.observed),
        "scan": scan
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    report_lines = [
        f"# APS Demo Report: {pack.name}",
        "",
        f"Generated: {results['timestamp']}",
        "",
        "## Inputs",
        f"- Pack: `{pack_path}`",
        f"- POI: `{results['poi']}`",
        f"- Scan range: [{scan_min:.2f}, {scan_max:.2f}] with {n_points} points",
        "",
        "## Results",
        f"- MLE mu: {results['mle']['theta'][0]:.4f}",
        f"- NLL at MLE: {results['mle']['nll']:.4f}",
        f"- 95% CL upper limit: {results['limit_95']:.4f}",
        "",
        "## Artifacts",
        "- `results.json`: numerical outputs and full scan data."
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report_lines))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run APS vertical-slice demo")
    parser.add_argument("--pack", default="packs/toy_counting", help="Path to analysis pack")
    parser.add_argument("--out", default="out/demo_toy_counting", help="Output directory")
    parser.add_argument("--poi-index", type=int, default=0, help="POI index to scan")
    parser.add_argument("--n-points", type=int, default=21, help="Number of scan points")
    args = parser.parse_args()

    results = run_demo(args.pack, Path(args.out), args.poi_index, args.n_points)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
