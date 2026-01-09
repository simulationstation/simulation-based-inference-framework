"""Tests for toy pack reproduction harness."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packs.toy_poisson_template_v1.validation.reproduce import run_reproduction


def test_reproduce_toy_pack():
    repo_root = Path(__file__).resolve().parents[1]
    pack_path = repo_root / "packs" / "toy_poisson_template_v1"
    expected_path = pack_path / "validation" / "expected.json"
    tolerances_path = pack_path / "validation" / "tolerances.yaml"

    ok, errors = run_reproduction(
        pack_path=str(pack_path),
        expected_path=str(expected_path),
        tolerances_path=str(tolerances_path)
    )

    assert ok, "\n".join(errors)
