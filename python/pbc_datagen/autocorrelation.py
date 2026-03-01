"""FFT-based autocorrelation and integrated autocorrelation time (τ_int).

Phase 2.0 — Steps 2.0.1–2.0.3.

Math reference (from PLAN.md):

    Autocorrelation via Wiener-Khinchin theorem:
        1. Center: x = O - mean(O)
        2. FFT with zero-padding: X = fft(x, n=2N)
        3. Power spectrum: S = |X|²
        4. ACF = ifft(S)[:N].real / ifft(S)[0].real   →  ρ(0) = 1

    Integrated autocorrelation time:
        τ_int = 1/2 + Σ_{t=1}^{t_cut} ρ(t)
        where t_cut = first lag where ρ(t) ≤ 0
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def acf_fft(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalized autocorrelation function via FFT.

    Zero-pads to 2N to compute *linear* (not circular) correlation.
    Returns array of length N with ρ(0) = 1.

    Raises:
        ValueError: If the input has zero variance (constant signal).
    """
    n = len(x)
    x_centered = np.asarray(x, dtype=np.float64) - np.mean(x)

    # Sum of squares = unnormalized variance × N
    var = np.dot(x_centered, x_centered)
    if var == 0.0:
        msg = "Constant input: zero variance, ACF is undefined"
        raise ValueError(msg)

    # Wiener-Khinchin: ACF = IFFT(|FFT(x)|²)
    ft = np.fft.fft(x_centered, n=2 * n)
    power = ft.real**2 + ft.imag**2  # |X|², avoids complex abs overhead
    acf = np.fft.ifft(power)[:n].real

    # Normalize so ρ(0) = 1
    acf /= acf[0]
    return acf


def tau_int(x: npt.NDArray[np.float64]) -> float:
    """Integrated autocorrelation time via first zero crossing.

    τ_int = 0.5 + Σ_{t=1}^{t_cut} ρ(t)

    where t_cut is the first lag where ρ(t) ≤ 0.  Simple and robust:
    no tuning parameters, works well when N >> τ_int.
    """
    rho = acf_fft(x)

    total = 0.5
    for t in range(1, len(rho)):
        if rho[t] <= 0.0:
            break
        total += rho[t]
    return float(total)


def tau_int_multi(
    sweep_dict: dict[str, npt.NDArray[np.float64]],
) -> tuple[dict[str, float], float]:
    """τ_int for every observable in a sweep() result dict.

    Returns:
        (per_obs, tau_max) where per_obs maps observable name → τ_int,
        and tau_max is the bottleneck (maximum across all observables).
        Thin at intervals ≥ 3 × tau_max for independent snapshots.
    """
    per_obs = {key: tau_int(series) for key, series in sweep_dict.items()}
    tau_max = max(per_obs.values())
    return per_obs, tau_max
