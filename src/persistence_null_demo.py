"""
Testing walk-forward persistence: why the baseline must not be built by block bootstrap
WITH replacement, and how to build it.

    python src/persistence_null_demo.py            # experiments on one path (a few minutes)
    python src/persistence_null_demo.py --quick    # fewer paths
    python src/persistence_null_demo.py --calibrate --n-obs 100 --n-sim 100 --jobs 16 --out data/calib
        # calibration of the baselines on no-edge i.i.d. and GARCH paths (hours on one core)

1. The statistic: rank correlation of rules' information ratio between each training window
   and the following test window (walk-forward), averaged over windows.
2. The truth: on independent random walks (no edge) the statistic is centred on zero.
3. Baselines built from ONE observed path:
     block bootstrap WITH replacement   -> the same multi-day segments appear in both training
                                           and test windows -> spurious persistence (bias up)
     block shuffle WITHOUT replacement  -> no duplicated segment -> close to the truth
4. A placebo: random long/flat rules have no persistence by construction.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

TRAIN, TEST, WARMUP, COST_BPS = 365, 90, 200, 10.0


# ------------------------------------------------------------ rules and their returns
def ma_rules(close: pd.Series) -> dict[str, pd.Series]:
    """72 moving-average crossover rules (long/flat and long/short)."""
    out = {}
    for f, s in itertools.product([2, 3, 5, 8, 10, 15, 20, 30, 40, 50], [10, 20, 30, 50, 75, 100, 150, 200]):
        if f < s:
            up = (close.rolling(f).mean() > close.rolling(s).mean()).astype(float)
            out[f"ma_{f}_{s}_long"], out[f"ma_{f}_{s}_ls"] = up, 2 * up - 1
    return out


def random_rules(index: pd.DatetimeIndex, n: int = 72, seed: int = 1) -> dict[str, pd.Series]:
    """Placebo: random long/flat regimes, meaningless by construction."""
    out = {}
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        hold = rng.integers(3, 60)
        state = (np.cumsum(rng.random(len(index)) < 1 / hold) + rng.integers(0, 2)) % 2
        out[f"rand_{k}"] = pd.Series(state.astype(float), index=index)
    return out


def active_returns(close: pd.Series, rules: dict[str, pd.Series]) -> pd.DataFrame:
    """Return of each rule minus buy-and-hold. A position decided at close t earns day t+1."""
    ret = close.pct_change()
    cols = {}
    for k, pos in rules.items():
        p = pos.shift(1).fillna(0.0)
        cols[k] = p * ret - p.diff().abs().fillna(p.abs()) * COST_BPS / 1e4
    return pd.DataFrame(cols).iloc[WARMUP:].sub(ret.iloc[WARMUP:], axis=0).fillna(0.0)


# ------------------------------------------------------------ the statistic
def ir(x: pd.DataFrame) -> pd.Series:
    """Information ratio (mean / s.d. of active return) of every rule. A rule that is long for
    the whole window equals buy-and-hold (active return 0, s.d. 0): its IR is set to 0."""
    sd = x.std(ddof=1)
    return (x.mean() / sd.where(sd > 0)).fillna(0.0)


def persistence(active: pd.DataFrame) -> float:
    """Mean over walk-forward windows of the rank correlation between training-window IR and
    next-window IR. > 0: past winners keep winning; < 0: rankings reverse."""
    rho = []
    for t0 in range(TRAIN, len(active) - TEST + 1, TEST):
        a, b = ir(active.iloc[t0 - TRAIN:t0]), ir(active.iloc[t0:t0 + TEST])
        rho.append(stats.spearmanr(a, b)[0])
    return float(np.mean(rho))


# ------------------------------------------------------------ paths
def random_walk(n: int, seed: int) -> pd.Series:
    """No edge by construction: fat-tailed (t, 4 d.f.) daily returns, ~3% daily volatility."""
    r = np.random.default_rng(seed).standard_t(4, n) * 0.03 / np.sqrt(2)
    return pd.Series(30000 * np.exp(np.cumsum(r)), index=pd.date_range("2021-01-01", periods=n))


def garch_walk(n: int, seed: int, omega: float = 1.8e-5, alpha: float = 0.08, beta: float = 0.90) -> pd.Series:
    """No edge, but volatility clustering: GARCH(1,1) with t (5 d.f.) innovations, zero mean,
    unconditional daily volatility 3% (omega / (1 - alpha - beta) = 0.03 ** 2)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(5, n) / np.sqrt(5 / 3)
    h, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(h) * z[t]
        h = omega + alpha * r[t] ** 2 + beta * h
    return pd.Series(30000 * np.exp(np.cumsum(r)), index=pd.date_range("2021-01-01", periods=n))


def block_bootstrap(r: np.ndarray, block: int, rng) -> np.ndarray:
    """Circular block bootstrap WITH replacement: a block can be drawn several times."""
    n = len(r)
    st = rng.integers(0, n, int(np.ceil(n / block)))
    return r[((st[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n]


def block_shuffle(r: np.ndarray, block: int, rng) -> np.ndarray:
    """Non-overlapping blocks shuffled WITHOUT replacement: every block appears exactly once."""
    blocks = [r[i:i + block] for i in range(0, len(r), block)]
    return np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])


def baseline(close: pd.Series, resample, block: int, n_sim: int, seed: int = 0) -> np.ndarray:
    """Persistence statistic on n_sim paths rebuilt from the observed daily returns."""
    rng = np.random.default_rng(seed)
    r = close.pct_change().dropna().to_numpy()
    out = []
    for _ in range(n_sim):
        path = pd.Series(close.iloc[0] * np.concatenate([[1.0], np.cumprod(1 + resample(r, block, rng))]),
                         index=close.index)
        out.append(persistence(active_returns(path, ma_rules(path))))
    return np.array(out)


# ------------------------------------------------------------ experiments
METHODS = {"boot20": (block_bootstrap, 20), "boot60": (block_bootstrap, 60),
           "shuffle1": (block_shuffle, 1), "shuffle20": (block_shuffle, 20), "shuffle60": (block_shuffle, 60)}
DGPS = {"iid": random_walk, "garch": garch_walk}


def _truth_one(args):
    dgp, n, seed = args
    c = DGPS[dgp](n, seed)
    return persistence(active_returns(c, ma_rules(c)))


def _obs_one(args):
    """One observed path: its statistic and, for each baseline method, the share of baseline
    paths at or below it (one-sided p-value for 'reversal')."""
    dgp, n, k, methods, n_sim = args
    close = DGPS[dgp](n, k)
    obs = persistence(active_returns(close, ma_rules(close)))
    row = {"dgp": dgp, "path": k, "observed": obs}
    for m in methods:
        fn, b = METHODS[m]
        base = baseline(close, fn, b, n_sim, seed=k)
        row[f"{m}_mean"] = base.mean()
        row[f"{m}_p"] = np.mean(base <= obs)
    return row


def calibrate(dgp: str, n: int, n_truth: int, n_obs: int, n_sim: int, methods: list[str],
              jobs: int, out_csv: str | None) -> None:
    """True distribution of the statistic for a no-edge process, and how often each baseline
    built from ONE observed path flags 'reversal' (p < 0.05) or 'persistence' (p > 0.95)."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        truth = np.array(list(ex.map(_truth_one, [(dgp, n, 10_000 + s) for s in range(n_truth)])))
        rows = list(ex.map(_obs_one, [(dgp, n, k, methods, n_sim) for k in range(n_obs)]))
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"saved {out_csv}")
    lo_t, hi_t = np.percentile(truth, [5, 95])
    print(f"truth ({n_truth} paths): mean {truth.mean():+.3f}  5-95% {lo_t:+.2f} .. {hi_t:+.2f}")
    print(f"exact p from the truth: P(p<0.05) {np.mean([np.mean(truth <= o) < 0.05 for o in df.observed]):.2f}"
          f"  (sanity check: should be about 0.05)")
    print(f"{'baseline':12s}{'bias':>8s}{'P<0.05':>9s}{'P>0.95':>9s}   (over {n_obs} observed paths, "
          f"{n_sim} baseline paths each; nominal 0.05)")
    se = np.sqrt(0.05 * 0.95 / n_obs)
    for m in methods:
        print(f"{m:12s}{(df[f'{m}_mean'] - truth.mean()).mean():+8.3f}{(df[f'{m}_p'] < 0.05).mean():9.2f}"
              f"{(df[f'{m}_p'] > 0.95).mean():9.2f}")
    print(f"(binomial s.e. of a rejection rate at 5%: {se:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=1500, help="days per path")
    ap.add_argument("--calibrate", action="store_true", help="calibration of the baselines (slow)")
    ap.add_argument("--dgp", choices=["iid", "garch", "both"], default="both")
    ap.add_argument("--n-truth", type=int, default=200)
    ap.add_argument("--n-obs", type=int, default=100, help="observed paths for the calibration")
    ap.add_argument("--n-sim", type=int, default=100, help="baseline paths per observed path")
    ap.add_argument("--methods", default="boot20,shuffle1,shuffle20,shuffle60")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--out", default=None, help="CSV prefix for the per-path results")
    args = ap.parse_args()
    if args.calibrate:
        for dgp in (["iid", "garch"] if args.dgp == "both" else [args.dgp]):
            print(f"\n==== {dgp}")
            calibrate(dgp, args.n, args.n_truth, args.n_obs, args.n_sim, args.methods.split(","),
                      args.jobs, f"{args.out}_{dgp}.csv" if args.out else None)
        return
    n_true, n_sim = (30, 40) if args.quick else (200, 200)

    print(f"== 1. Truth: {n_true} independent random walks (no edge)")
    truth = np.array([persistence(active_returns(c, ma_rules(c)))
                      for c in (random_walk(args.n, s) for s in range(1000, 1000 + n_true))])
    print(f"mean {truth.mean():+.3f}  s.d. {truth.std():.3f}  5-95% {np.percentile(truth, 5):+.2f} .. "
          f"{np.percentile(truth, 95):+.2f}")

    close = random_walk(args.n, 0)
    obs = persistence(active_returns(close, ma_rules(close)))
    print(f"\n== 2. Baselines rebuilt from ONE no-edge path (observed statistic {obs:+.3f})")
    print(f"{'method':38s}{'mean':>8s}{'5-95%':>17s}{'P(base<=obs)':>14s}")
    for name, fn, blocks in [("block bootstrap WITH replacement", block_bootstrap, (20, 60)),
                             ("block shuffle WITHOUT replacement", block_shuffle, (1, 20, 60))]:
        for b in blocks:
            base = baseline(close, fn, b, n_sim)
            lo, hi = np.percentile(base, [5, 95])
            print(f"{name + f', block {b}':38s}{base.mean():+8.3f}{lo:+9.2f} .. {hi:+.2f}"
                  f"{np.mean(base <= obs):14.2f}")
    print(f"truth (independent random walks)     {truth.mean():+8.3f}{np.percentile(truth, 5):+9.2f} .. "
          f"{np.percentile(truth, 95):+.2f}{np.mean(truth <= obs):14.2f}")

    print("\n== 3. Placebo: random rules on the same path")
    print(f"persistence of random rules: {persistence(active_returns(close, random_rules(close.index))):+.3f}")


if __name__ == "__main__":
    main()
