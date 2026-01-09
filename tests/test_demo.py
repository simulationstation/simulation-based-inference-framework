"""Integration test for aps demo command logic."""

import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aps.demo import run_demo


def test_demo_run_outputs():
    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "packs" / "toy_poisson_template_v1"
    expected_path = pack_path / "validation" / "expected.json"
    tolerances_path = pack_path / "validation" / "tolerances.yaml"

    expected = json.loads(expected_path.read_text())
    tolerances = yaml.safe_load(tolerances_path.read_text())

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_demo(
            pack_path=str(pack_path),
            out_dir=tmpdir,
            seed=123,
            scan_min=0.0,
            scan_max=2.0,
            scan_steps=11,
            n_starts=50
        )

        out_path = Path(tmpdir)
        assert (out_path / "results.json").exists()
        assert (out_path / "REPORT.md").exists()
        assert (out_path / "scan.csv").exists()

    tol = tolerances.get("mu_hat", tolerances.get("default", 0.0))
    assert abs(results["fit"]["mu_hat"] - expected["mu_hat"]) <= tol
