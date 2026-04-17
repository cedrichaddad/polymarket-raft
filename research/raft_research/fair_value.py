"""Parametric fair-value backbone (§11.1).

    p_0(t) = Phi( (log(S_t/K) - b*tau) / (sigma_per_sec * sqrt(tau)) )
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def fair_prob(
    s_t: np.ndarray,
    k: np.ndarray,
    tau_seconds: np.ndarray,
    sigma_per_sec: np.ndarray | float,
    drift: float = 0.0,
) -> np.ndarray:
    s_t = np.asarray(s_t, dtype=float)
    k = np.asarray(k, dtype=float)
    tau_seconds = np.asarray(tau_seconds, dtype=float)
    sigma = np.asarray(sigma_per_sec, dtype=float) if np.ndim(sigma_per_sec) else np.full_like(s_t, sigma_per_sec)
    valid = (s_t > 0) & (k > 0) & (tau_seconds > 0) & (sigma > 0)
    z = np.zeros_like(s_t, dtype=float)
    safe_sigma = np.where(sigma > 0, sigma, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = safe_sigma * np.sqrt(np.where(tau_seconds > 0, tau_seconds, 1e-12))
        z = (np.log(np.where(s_t > 0, s_t, 1.0) / np.where(k > 0, k, 1.0)) - drift * tau_seconds) / denom
    out = np.where(valid, norm.cdf(z), np.nan)
    return out
