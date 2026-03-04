"""Tests for FFT-based autocorrelation and integrated autocorrelation time.

Red-phase tests for Phase 2.0 (Steps 2.0.1–2.0.3).
All imports are lazy (inside test functions) so pytest can collect
even before the implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — synthetic signal generators
# ---------------------------------------------------------------------------


def _generate_ar1(phi: float, n: int, seed: int = 42) -> np.ndarray:
    """Generate an AR(1) process: x_t = phi * x_{t-1} + eps_t.

    Analytical ACF: rho(t) = phi^|t|.
    Analytical tau_int: 0.5 + phi / (1 - phi).
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


# ---------------------------------------------------------------------------
# acf_fft tests (Step 2.0.1)
# ---------------------------------------------------------------------------


def test_acf_fft_white_noise_normalized() -> None:
    """White noise ACF: rho(0)=1, rho(t>0) near zero."""
    from pbc_datagen.autocorrelation import acf_fft

    rng = np.random.default_rng(123)
    x = rng.standard_normal(10_000)

    rho = acf_fft(x)

    # rho(0) must be exactly 1 (normalization)
    assert rho[0] == pytest.approx(1.0, abs=1e-12)

    # rho(t>0) should be near zero — within ±3/sqrt(N) ≈ 0.03
    bound = 3.0 / np.sqrt(len(x))
    assert np.all(np.abs(rho[1:100]) < bound), (
        f"White noise ACF exceeds ±{bound:.4f} in first 100 lags"
    )


def test_acf_fft_ar1_matches_analytical() -> None:
    """AR(1) with phi=0.5: first 15 lags of rho(t) match phi^t."""
    from pbc_datagen.autocorrelation import acf_fft

    phi = 0.5
    x = _generate_ar1(phi, n=50_000, seed=99)

    rho = acf_fft(x)

    # Compare first 15 lags (phi^15 ≈ 3e-5, well above noise floor)
    for t in range(15):
        expected = phi**t
        assert rho[t] == pytest.approx(expected, abs=0.02), (
            f"ACF mismatch at lag {t}: got {rho[t]:.4f}, expected {expected:.4f}"
        )


def test_acf_fft_constant_input_raises() -> None:
    """Constant input has zero variance — acf_fft must raise ValueError."""
    from pbc_datagen.autocorrelation import acf_fft

    x = np.ones(100)

    with pytest.raises(ValueError, match="[Vv]ariance|[Cc]onstant"):
        acf_fft(x)


# ---------------------------------------------------------------------------
# tau_int tests (Step 2.0.2)
# ---------------------------------------------------------------------------


def test_tau_int_white_noise() -> None:
    """White noise: tau_int should be approximately 0.5.

    For uncorrelated data, rho(t>0) ≈ 0 so the sum contributes nothing.
    tau_int = 0.5 + sum(~0) ≈ 0.5.
    """
    from pbc_datagen.autocorrelation import tau_int

    rng = np.random.default_rng(456)
    x = rng.standard_normal(10_000)

    tau = tau_int(x)

    assert tau == pytest.approx(0.5, abs=0.15), f"White noise tau_int = {tau:.3f}, expected ~0.5"


def test_tau_int_ar1_recovers_known_value() -> None:
    """AR(1) with phi=0.5: analytical tau_int = 0.5 + 0.5/0.5 = 1.5."""
    from pbc_datagen.autocorrelation import tau_int

    phi = 0.5
    expected_tau = 0.5 + phi / (1.0 - phi)  # = 1.5

    x = _generate_ar1(phi, n=50_000, seed=77)

    tau = tau_int(x)

    assert tau == pytest.approx(expected_tau, rel=0.15), (
        f"AR(1) phi={phi} tau_int = {tau:.3f}, expected {expected_tau:.3f}"
    )


def test_tau_int_ar1_higher_correlation() -> None:
    """AR(1) with phi=0.9: analytical tau_int = 0.5 + 9.0 = 9.5.

    Stronger correlation → larger tau_int. Needs a longer series for
    the zero-crossing cutoff to work well.
    """
    from pbc_datagen.autocorrelation import tau_int

    phi = 0.9
    expected_tau = 0.5 + phi / (1.0 - phi)  # = 9.5

    x = _generate_ar1(phi, n=200_000, seed=88)

    tau = tau_int(x)

    assert tau == pytest.approx(expected_tau, rel=0.20), (
        f"AR(1) phi={phi} tau_int = {tau:.3f}, expected {expected_tau:.3f}"
    )


# ---------------------------------------------------------------------------
# tau_int_multi tests (Step 2.0.3)
# ---------------------------------------------------------------------------


def test_tau_int_multi_returns_per_key_and_bottleneck() -> None:
    """tau_int_multi on a dict of two signals returns correct per-key
    tau_int values and identifies the bottleneck (maximum).

    Construct two AR(1) signals with different phi values so one has a
    clearly larger tau_int. The bottleneck must be the slower one.
    """
    from pbc_datagen.autocorrelation import tau_int_multi

    phi_fast = 0.3  # tau = 0.5 + 0.3/0.7 ≈ 0.93
    phi_slow = 0.8  # tau = 0.5 + 0.8/0.2 = 4.5

    expected_fast = 0.5 + phi_fast / (1.0 - phi_fast)
    expected_slow = 0.5 + phi_slow / (1.0 - phi_slow)

    sweep_dict: dict[str, np.ndarray] = {
        "energy": _generate_ar1(phi_slow, n=100_000, seed=10),
        "abs_m": _generate_ar1(phi_fast, n=100_000, seed=20),
    }

    per_obs, tau_max = tau_int_multi(sweep_dict)

    # Check return structure
    assert set(per_obs.keys()) == {"energy", "abs_m"}

    # Per-observable values within tolerance
    assert per_obs["energy"] == pytest.approx(expected_slow, rel=0.20), (
        f"energy tau = {per_obs['energy']:.3f}, expected {expected_slow:.3f}"
    )
    assert per_obs["abs_m"] == pytest.approx(expected_fast, rel=0.25), (
        f"abs_m tau = {per_obs['abs_m']:.3f}, expected {expected_fast:.3f}"
    )

    # Bottleneck must be the slow observable
    assert tau_max == pytest.approx(per_obs["energy"], abs=1e-12), (
        f"tau_max = {tau_max:.3f} but energy tau = {per_obs['energy']:.3f}"
    )
    assert tau_max > per_obs["abs_m"], (
        "tau_max should be strictly larger than the fast observable's tau_int"
    )


# ---------------------------------------------------------------------------
# tau_int_batch tests
# ---------------------------------------------------------------------------


def test_tau_int_batch_matches_sequential() -> None:
    """tau_int_batch must agree with sequential tau_int on each row."""
    from pbc_datagen.autocorrelation import tau_int, tau_int_batch

    rng = np.random.default_rng(42)
    n_series, n = 200, 500
    phi = 0.7
    X = np.zeros((n_series, n))
    X[:, 0] = rng.standard_normal(n_series)
    for t in range(1, n):
        X[:, t] = phi * X[:, t - 1] + rng.standard_normal(n_series)

    batch_result = tau_int_batch(X)
    seq_result = np.array([tau_int(X[i]) for i in range(n_series)])

    np.testing.assert_allclose(batch_result, seq_result, rtol=1e-10)


def test_tau_int_batch_constant_rows() -> None:
    """Constant rows must return 0.5 (the sentinel value)."""
    from pbc_datagen.autocorrelation import tau_int_batch

    X = np.ones((50, 100))
    result = tau_int_batch(X)
    np.testing.assert_allclose(result, 0.5)
