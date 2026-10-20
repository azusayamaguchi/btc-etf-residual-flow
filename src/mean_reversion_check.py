"""
Does the BTC price mean-revert? Checks before fitting an Ornstein-Uhlenbeck (OU) process.

  1. ADF  (H0: unit root / random walk)   on log price
  2. KPSS (H0: stationary / mean-reverting) on log price
  3. Variance ratio (Lo-MacKinlay, heteroskedasticity-robust): VR(q) < 1 -> mean reversion
  4. OU half-life and its false-positive check: the half-life estimated from
     dlogP_t = a + b logP_{t-1} + e is biased toward mean reversion in finite samples,
     so compare it with the distribution obtained from random walks of the same length.

Usage:
    python src/mean_reversion_check.py --start 2018-01-01
    python src/mean_reversion_check.py --demo     # a random walk and an OU process, for comparison
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def ou_fit(logp: np.ndarray) -> tuple[float, float]:
    """Regress dlogP on lagged logP; return slope b and half-life in days (inf if b >= 0)."""
    y, x = np.diff(logp), logp[:-1]
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    half_life = -np.log(2) / np.log(1 + b) if -1 < b < 0 else np.inf
    return float(b), float(half_life)


def half_life_null(logp: np.ndarray, n_sim: int = 2000, block: int = 20, seed: int = 0) -> np.ndarray:
    """Slopes b from random walks built by resampling blocks of the observed daily log returns
    (keeps fat tails and volatility clustering; the random block order removes any slow
    mean reversion of the level)."""
    rng = np.random.default_rng(seed)
    r = np.diff(logp)
    n = len(r)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_sim)
    for i in range(n_sim):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        path = logp[0] + np.concatenate([[0.0], np.cumsum(r[idx])])
        out[i] = ou_fit(path)[0]
    return out


def variance_ratio(logp: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay variance ratio VR(q) and heteroskedasticity-robust z statistic."""
    r = np.diff(logp)
    n, mu = len(r), r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = logp[q:] - logp[:-q]
    m = q * (n - q + 1) * (1 - q / n)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    dev2 = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = np.sum(dev2[j:] * dev2[:-j]) / np.sum(dev2) ** 2 * n
        theta += (2 * (q - j) / q) ** 2 * delta
    return float(vr), float((vr - 1) / np.sqrt(theta / n))


def unit_root_tests(logp: np.ndarray) -> None:
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError:
        print("statsmodels not installed: skipping ADF / KPSS")
        return
    adf = adfuller(logp, regression="c", autolag="AIC")
    kp = kpss(logp, regression="c", nlags="auto")
    print(f"ADF  (H0 random walk) : stat {adf[0]:.2f}, p = {adf[1]:.3f}  "
          f"-> {'reject: evidence of mean reversion' if adf[1] < 0.05 else 'cannot reject random walk'}")
    print(f"KPSS (H0 stationary)  : stat {kp[0]:.2f}, p = {kp[1]:.3f}  "
          f"-> {'reject: not mean-reverting' if kp[1] < 0.05 else 'cannot reject stationarity'}")


def check(name: str, price: pd.Series) -> None:
    logp = np.log(price.dropna().to_numpy())
    print(f"\n===== {name}: {len(logp)} days")
    unit_root_tests(logp)
    for q in (5, 20, 60, 120):
        vr, z = variance_ratio(logp, q)
        print(f"VR({q:3d}) = {vr:.2f}  z = {z:+.2f}  "
              f"{'(mean reversion)' if z < -1.96 else '(momentum)' if z > 1.96 else ''}")
    b, hl = ou_fit(logp)
    null_b = half_life_null(logp)
    p = float(np.mean(null_b <= b))                  # how often a random walk looks this mean-reverting
    hl_null = -np.log(2) / np.log(1 + np.median(null_b[null_b < 0]))
    print(f"OU fit: b = {b:.5f}, half-life {hl:,.0f} days")
    print(f"random walks of the same length: median half-life {hl_null:,.0f} days; "
          f"P(b as negative as observed) = {p:.2f}  "
          f"-> {'mean reversion beyond chance' if p < 0.05 else 'consistent with a random walk'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--price-csv", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        rng = np.random.default_rng(1)
        n, idx = 2000, pd.date_range("2018-01-01", periods=2000)
        rw = np.cumsum(rng.standard_t(4, n) * 0.025)
        ou = np.zeros(n)
        for t in range(1, n):                         # true half-life about 70 days
            ou[t] = ou[t - 1] * (1 - 0.01) + rng.standard_t(4) * 0.025
        check("random walk (no mean reversion)", pd.Series(np.exp(10 + rw), index=idx))
        check("OU process (true half-life ~70 days)", pd.Series(np.exp(10 + ou), index=idx))
        return

    from yardstick import load_daily_close
    check("BTC-USD daily close", load_daily_close(args.start, args.end, args.price_csv))


if __name__ == "__main__":
    main()
