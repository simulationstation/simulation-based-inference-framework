"""Tests for analysis pack loading."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aps


def test_load_toy_pack():
    pack = aps.load("packs/toy_counting")
    assert pack.name == "toy_counting_v1"
    assert pack.likelihood_type == "poisson"
    assert pack.n_pois == 1
    assert pack.n_nuisances == 2
    assert "counting" in pack.channels


def test_pack_model_execution():
    pack = aps.load("packs/toy_counting")
    model = pack.model()
    ll = model.logpdf(theta=[1.0], nu=[0.0, 0.0])
    assert ll == ll
