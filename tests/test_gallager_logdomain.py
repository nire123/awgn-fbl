"""Tests for the log-domain Gallager evaluation."""

import numpy as np
import pytest

from awgn_fbl import GallagerAchievable


pytestmark = pytest.mark.filterwarnings("ignore")


def test_log_matches_linear_where_representable():
    g = GallagerAchievable(n=500, snr_db=3.0)
    for R in [0.2, 0.4, 0.6, 0.8]:
        lin = g.achievable_error(R)
        if lin > 1e-12:
            assert np.exp(g.log_achievable_error(R)) == pytest.approx(lin, rel=1e-9)


def test_log_finite_arbitrarily_deep():
    g = GallagerAchievable(n=2000, snr_db=9.0)
    lp = g.log_achievable_error(0.6 * g.capacity)
    assert np.isfinite(lp) and lp < -300      # far past linear underflow


def test_log_clamped_at_one():
    g = GallagerAchievable(n=200, snr_db=0.0)
    # above capacity the exponent goes non-positive; the bound saturates at 1
    assert g.log_achievable_error(1.2 * g.capacity) == pytest.approx(0.0, abs=1e-9)
    # and log P_e is never positive (a probability bound)
    for R in np.linspace(0.05, 1.5 * g.capacity, 20):
        assert g.log_achievable_error(R) <= 1e-9


def test_rate_monotone_in_eps_including_deep():
    g = GallagerAchievable(n=2000, snr_db=9.0)
    rates = [g.achievable_rate(e) for e in [1e-200, 1e-100, 1e-30, 1e-9, 1e-3]]
    assert all(b > a for a, b in zip(rates, rates[1:]))


def test_rate_below_capacity_deep_eps():
    g = GallagerAchievable(n=2000, snr_db=9.0)
    for e in [1e-50, 1e-150, 1e-300]:
        assert 0 < g.achievable_rate(e) < g.capacity


def test_reference_point_logM_1225():
    # Polyanskiy's Gallager reference: n=3000, 0 dB, eps=1e-6 -> log2 M ~ 1225
    R = GallagerAchievable(n=3000, snr_db=0.0).achievable_rate(1e-6)
    assert R * 3000 == pytest.approx(1225, abs=5)
