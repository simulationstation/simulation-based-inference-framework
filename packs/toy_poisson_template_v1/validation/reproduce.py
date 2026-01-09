"""Reproduce expected outputs for toy_poisson_template_v1."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from aps.demo import run_demo


DEFAULT_SCAN = {
    "scan_min": 0.0,
    "scan_max": 2.0,
    "scan_steps": 11
}


def run_reproduction(
    pack_path: str,
    expected_path: str,
    tolerances_path: str,
    seed: int = 123
) -> Tuple[bool, List[str]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_demo(
            pack_path=pack_path,
            out_dir=tmpdir,
            seed=seed,
            scan_min=DEFAULT_SCAN["scan_min"],
            scan_max=DEFAULT_SCAN["scan_max"],
            scan_steps=DEFAULT_SCAN["scan_steps"],
            n_starts=50
        )

    expected = _load_json(expected_path)
    tolerances = _load_yaml(tolerances_path)

    errors = []
    errors.extend(_compare_scalar("mu_hat", results["fit"]["mu_hat"], expected["mu_hat"], tolerances))
    errors.extend(_compare_scalar("nll", results["fit"]["nll"], expected["nll"], tolerances))
    errors.extend(_compare_array("nu_hat", results["fit"]["nu_hat"], expected["nu_hat"], tolerances))

    errors.extend(
        _compare_array(
            "scan.mu",
            results["scan"]["mu"],
            expected["scan"]["mu"],
            tolerances
        )
    )
    errors.extend(
        _compare_array(
            "scan.qmu",
            results["scan"]["qmu"],
            expected["scan"]["qmu"],
            tolerances
        )
    )
    errors.extend(
        _compare_array(
            "scan.delta_nll",
            results["scan"]["delta_nll"],
            expected["scan"]["delta_nll"],
            tolerances
        )
    )

    return len(errors) == 0, errors


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _compare_scalar(name: str, actual: float, expected: float, tolerances: Dict[str, Any]) -> List[str]:
    tol = tolerances.get(name, tolerances.get("default", 0.0))
    if abs(actual - expected) > tol:
        return [f"{name} mismatch: {actual} vs {expected} (tol={tol})"]
    return []


def _compare_array(
    name: str,
    actual: List[float],
    expected: List[float],
    tolerances: Dict[str, Any]
) -> List[str]:
    tol = tolerances.get(name, tolerances.get("default", 0.0))
    if len(actual) != len(expected):
        return [f"{name} length mismatch: {len(actual)} vs {len(expected)}"]
    errors = []
    for idx, (a_val, e_val) in enumerate(zip(actual, expected)):
        if abs(a_val - e_val) > tol:
            errors.append(f"{name}[{idx}] mismatch: {a_val} vs {e_val} (tol={tol})")
    return errors


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    pack_path = repo_root / "packs" / "toy_poisson_template_v1"
    expected_path = pack_path / "validation" / "expected.json"
    tolerances_path = pack_path / "validation" / "tolerances.yaml"

    ok, errors = run_reproduction(
        pack_path=str(pack_path),
        expected_path=str(expected_path),
        tolerances_path=str(tolerances_path)
    )

    if not ok:
        for err in errors:
            print(err)
        raise SystemExit(1)

    print("Reproduction passed.")
