"""Tests for the toy_poisson_template_v1 pack."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aps
from runtime.inference import profile_likelihood


def test_toy_pack_validation():
    pack = aps.load("packs/toy_poisson_template_v1", validate=True)
    assert pack.name == "toy_poisson_template_v1"
    assert pack.n_pois == 1
    assert pack.n_nuisances == 3
    assert "toy_channel" in pack.channels


def test_toy_pack_logl_finite():
    pack = aps.load("packs/toy_poisson_template_v1", validate=True)
    model = pack.model()

    for mu in np.linspace(0.0, 2.0, 5):
        ll = model.logpdf(theta=np.array([mu]), nu=np.zeros(model.n_nuisances))
        assert np.isfinite(ll)


def test_toy_pack_profile_nuisances():
    pack = aps.load("packs/toy_poisson_template_v1", validate=True)
    model = pack.model()
    profiled = profile_likelihood(model, theta=np.array([1.0]), n_starts=10)
    assert np.isfinite(profiled.nll_profiled)
    assert len(profiled.nu_profiled) == model.n_nuisances


def test_toy_pack_scan_curve():
    pack = aps.load("packs/toy_poisson_template_v1", validate=True)
    model = pack.model()
    mu_values = np.linspace(0.0, 2.0, 7)
    nll_values = [model.nll(theta=np.array([mu]), nu=np.zeros(model.n_nuisances)) for mu in mu_values]
    assert all(np.isfinite(nll) for nll in nll_values)
