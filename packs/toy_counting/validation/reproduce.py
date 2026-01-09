#!/usr/bin/env python3
"""Reproduce reference results for the toy_counting pack."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import aps
from runtime import compute_mle, upper_limit

PACK_PATH = Path(__file__).resolve().parents[1]
OUT_PATH = PACK_PATH / "validation" / "expected_results.json"


def main() -> None:
    pack = aps.load(str(PACK_PATH))
    model = pack.model()

    mle = compute_mle(model, n_starts=50)
    limit = upper_limit(model, poi_index=0, cl=0.95, n_starts=50)

    results = {
        "mle_mu": float(mle.theta_hat[0]),
        "nll": float(mle.nll),
        "limit_95": float(limit.observed),
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
