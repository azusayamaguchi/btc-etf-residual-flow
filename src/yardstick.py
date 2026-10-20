"""
Yardstick for backtest reliability: the evaluation protocol used in every later article.

  1. Timestamp alignment  : a position decided at close t earns the return of day t+1 only.
  2. Walk-forward          : select on past data, evaluate on the following unseen window,
                             with purging (label overlap) and embargo (serial dependence).
  3. Trial log             : every strategy tried is recorded (data/trials.csv).
  4. Deflated Sharpe Ratio : the best Sharpe is deflated for the number of trials, skewness
                             and kurtosis (Bailey & Lopez de Prado, 2014).
  5. Block bootstrap       : confidence intervals that keep serial dependence.
  6. Costs                 : proportional cost on every change of position.
  7. Benchmark             : buy-and-hold over the same days. PSR/DSR and the walk-forward
                             selection use the excess return over buy-and-hold.

Usage:
    python src/yardstick.py --demo                      # synthetic random walk (no edge by design)
    python src/yardstick.py --start 2022-01-01          # BTC-USD daily closes from Coinbase
    python src/yardstick.py --price-csv data/btc_daily.csv --cost-bps 10

The brute-force search over "meaningless" rules (moving-average crossovers and random
regime signals) is the demonstration: the best in-sample rule looks good, the yardstick
says it is not.
"""
from __future__ import annotations

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ANN = 365                     # crypto trades every day
EULER_GAMMA = 0.5772156649


# ============================================================ data
def load_daily_close(start: str, end: str | None, price_csv: str | None,
                     product: str = "BTC-USD") -> pd.Series:
    """Daily close at 00:00 UTC (the usual crypto daily bar)."""
    if price_csv:
        s = pd.read_csv(price_csv, index_col=0, parse_dates=True).iloc[:, 0]
        return s.astype(float).rename("close")
    from btc_etf_flow_analysis import load_hourly
    h = load_hourly(start, end, product=product).tz_convert("UTC")
    daily = h[h.index.hour == 0]                       # candle ending at 00:00 UTC
    daily.index = daily.index.normalize().tz_localize(None)
    return daily.rename("close")


def make_demo(n: int = 1500, seed: int = 0) -> pd.Series:
    """Geometric random walk with BTC-like volatility: no rule can have a true edge."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(df=4, size=n) * 0.03 / np.sqrt(2)          # fat tails, ~3% daily vol
    return pd.Series(30000 * np.exp(np.cumsum(r)), index=pd.date_range("2021-01-01", periods=n), name="close")


# ============================================================ strategies (1. timing, 6. costs)
def strategy_returns(close: pd.Series, position: pd.Series, cost_bps: float) -> pd.Series:
    """Position decided at close t (from data <= t) earns the close-to-close return of t+1."""
    ret = close.pct_change()
    pos = position.shift(1).fillna(0.0)                 # no look-ahead
    turnover = pos.diff().abs().fillna(pos.abs())
    return (pos * ret - turnover * cost_bps / 1e4).fillna(0.0)


def ma_rules(close: pd.Series) -> dict[str, pd.Series]:
    fast = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    slow = [10, 20, 30, 50, 75, 100, 150, 200]
    out = {}
    for f, s in itertools.product(fast, slow):
        if f >= s:
            continue
        up = (close.rolling(f).mean() > close.rolling(s).mean()).astype(float)
        out[f"ma_{f}_{s}_long"] = up                            # long / flat
        out[f"ma_{f}_{s}_ls"] = 2 * up - 1                      # long / short
    return out


ANCHOR = pd.date_range("2010-01-01", "2040-12-31", freq="D")   # fixed calendar for random rules


def random_rules(index: pd.DatetimeIndex, n: int = 300, seed: int = 1) -> dict[str, pd.Series]:
    """Random long/flat regimes with random average holding period: meaningless by construction.
    Each rule has its own random stream and is drawn on a fixed calendar, then cut to `index`,
    so the rules do not change when the data period is extended or shortened."""
    out = {}
    for k in range(n):
        rng = np.random.default_rng([seed, k])
        hold = rng.integers(3, 60)
        flips = rng.random(len(ANCHOR)) < 1 / hold
        state = (np.cumsum(flips) + rng.integers(0, 2)) % 2
        out[f"rand_{k:03d}_h{hold}"] = pd.Series(state.astype(float), index=ANCHOR).reindex(index)
    return out


def random_rules_legacy(index: pd.DatetimeIndex, n: int = 300, seed: int = 1) -> dict[str, pd.Series]:
    """Previous generator (the rules depend on the length of `index`). Kept only to reproduce
    results published before the change: use it with the same --start and --end."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n):
        hold = rng.integers(3, 60)
        flips = rng.random(len(index)) < 1 / hold
        state = (np.cumsum(flips) + rng.integers(0, 2)) % 2
        out[f"rand_{k:03d}_h{hold}"] = pd.Series(state.astype(float), index=index)
    return out


# ============================================================ Sharpe, PSR, DSR (4)
def sharpe(r: pd.Series | np.ndarray) -> float:
    """Non-annualised Sharpe ratio of daily returns."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    return r.mean() / sd if sd > 0 else 0.0


def psr(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark), adjusting for skew and kurtosis."""
    r = np.asarray(r, dtype=float)
    t = len(r)
    sr = sharpe(r)
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)                # normal = 3
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / denom))


def expected_max_sharpe(sr_var: float, n_trials: float) -> float:
    """Expected maximum Sharpe among n independent trials with true SR = 0."""
    n = max(n_trials, 2.0)
    return np.sqrt(sr_var) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n)
                              + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * np.e)))


def effective_trials(returns: pd.DataFrame) -> float:
    """Rough number of independent trials from the average pairwise correlation:
    N_eff = rho + (1 - rho) * N  (approximation; correlated rules count as fewer trials)."""
    c = returns.corr().to_numpy()
    n = c.shape[0]
    rho = (c.sum() - n) / (n * (n - 1))
    rho = float(np.clip(rho, 0, 1))
    return rho + (1 - rho) * n


def dsr(best: np.ndarray, all_sr: np.ndarray, n_trials: float) -> tuple[float, float]:
    """Deflated Sharpe Ratio: PSR of the best strategy against the expected max SR under the null."""
    sr0 = expected_max_sharpe(np.var(all_sr, ddof=1), n_trials)
    return psr(best, sr0), sr0


# ============================================================ walk-forward with purge / embargo (2)
def purged_splits(n: int, test_len: int, horizon: int = 1, embargo: int = 0,
                  train_len: int | None = None, scheme: str = "walk_forward"):
    """
    Yield (train_idx, test_idx).
      horizon : label length in days. A training label that ends inside the test window
                overlaps it, so the last (horizon - 1) training days before the test are purged.
      embargo : days dropped right AFTER each test window from training (matters for k-fold,
                where training data can come after the test; ignored by walk-forward).
      scheme  : "walk_forward" (train strictly before test) or "kfold" (train on both sides).
    """
    purge = max(horizon - 1, 0)
    starts = range(train_len or test_len, n - test_len + 1, test_len) if scheme == "walk_forward" \
        else range(0, n - test_len + 1, test_len)
    for t0 in starts:
        test = np.arange(t0, t0 + test_len)
        lo = 0 if (scheme == "kfold" or train_len is None) else max(0, t0 - train_len)
        before = np.arange(lo, max(t0 - purge, lo))
        if scheme == "kfold":
            # labels after the test start after it, so no purge is needed there; the embargo
            # drops days whose features are serially dependent on the test window
            after = np.arange(min(t0 + test_len + embargo, n), n)
            train = np.concatenate([before, after])
        else:
            train = before
        if len(train) > 0:
            yield train, test


def walk_forward_selection(rule_returns: pd.DataFrame, train_len: int, test_len: int,
                           horizon: int = 1, bench: pd.Series | None = None) -> tuple[pd.Series, list[str]]:
    """In each window pick the rule with the best in-sample Sharpe (of the excess return over
    `bench` if given); record its out-of-sample returns."""
    oos, picks = [], []
    for tr, te in purged_splits(len(rule_returns), test_len, horizon, train_len=train_len):
        ins = rule_returns.iloc[tr]
        if bench is not None:
            ins = ins.sub(bench.iloc[tr], axis=0)
        best = ins.apply(sharpe).idxmax()
        picks.append(best)
        oos.append(rule_returns[best].iloc[te])
    return pd.concat(oos), picks


# ============================================================ block bootstrap (5)
def block_bootstrap_sharpe(r: np.ndarray, block: int = 20, n_boot: int = 2000,
                           bench: np.ndarray | None = None, seed: int = 0) -> dict:
    """Circular block bootstrap of the annualised Sharpe (and of the difference vs. a benchmark)."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)
    n = len(r)
    n_blocks = int(np.ceil(n / block))
    sr, diff = [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        sr.append(sharpe(r[idx]) * np.sqrt(ANN))
        if bench is not None:
            diff.append(sr[-1] - sharpe(np.asarray(bench)[idx]) * np.sqrt(ANN))
    out = {"sr_ci": np.percentile(sr, [2.5, 97.5]), "p_sr_le_0": float(np.mean(np.array(sr) <= 0))}
    if bench is not None:
        out["diff_ci"] = np.percentile(diff, [2.5, 97.5])
        out["p_diff_le_0"] = float(np.mean(np.array(diff) <= 0))
    return out


# ============================================================ structural-break episodes
def load_episodes(path: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """CSV with columns start,end (inclusive dates), e.g. GSADF-detected episodes.
    Used ONLY to describe the data (attribution, stratification), never inside a trading rule:
    full-sample dating is ex-post information."""
    df = pd.read_csv(path)
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(df["start"], df["end"])]


def episode_flag(index: pd.DatetimeIndex, episodes) -> pd.Series:
    flag = pd.Series(False, index=index)
    for a, b in episodes:
        flag |= (index >= a) & (index <= b)
    return flag


def episode_attribution(bh: pd.Series, best: pd.Series, oos: pd.Series, episodes) -> None:
    """How much of the performance comes from the episode days?"""
    flag = episode_flag(bh.index, episodes)
    print(f"episode days: {int(flag.sum())} of {len(flag)}")
    for a, b in episodes:
        m = (bh.index >= a) & (bh.index <= b)
        if not m.any():
            print(f"  {a.date()} to {b.date()}: outside the evaluation period "
                  f"({bh.index.min().date()} onward, after the 200-day warm-up)")
            continue
        part = "" if (bh.index[m].min() == a and bh.index[m].max() == b) else "  (partly outside)"
        print(f"  {a.date()} to {b.date()}: BTC {np.prod(1 + bh[m]) - 1:+.1%}, "
              f"best rule excess {np.prod(1 + best[m]) - np.prod(1 + bh[m]):+.1%}{part}")
    rows = []
    for lab, m in [("inside episodes", flag), ("outside episodes", ~flag)]:
        o = oos.reindex(bh.index[m]).dropna()
        rows.append({"days": lab,
                     "BTC log return": np.log1p(bh[m]).sum(),
                     "best-rule excess (log)": (np.log1p(best[m]) - np.log1p(bh[m])).sum(),
                     "WF-OOS excess (log)": (np.log1p(o) - np.log1p(bh.reindex(o.index))).sum()})
    print(pd.DataFrame(rows).set_index("days").round(3).to_string())


def leave_one_episode_out(rets: pd.DataFrame, bh: pd.Series, episodes) -> None:
    """Re-run the full search with each episode removed: does the conclusion depend on one episode?"""
    print(f"{'removed':25s}{'best rule':16s}{'excess SR':>10s}{'DSR':>7s}{'BH SR':>7s}")
    inside = [(a, b) for a, b in episodes if ((rets.index >= a) & (rets.index <= b)).any()]
    for a, b in [(None, None)] + inside:
        keep = pd.Series(True, index=rets.index) if a is None else ~((rets.index >= a) & (rets.index <= b))
        ex = rets[keep].sub(bh[keep], axis=0)
        srs = ex.apply(sharpe)
        best = srs.idxmax()
        d, _ = dsr(ex[best].to_numpy(), srs.to_numpy(), len(srs))
        lab = "none" if a is None else f"{a.date()}..{b.date()}"
        print(f"{lab:25s}{best:16s}{srs[best] * np.sqrt(ANN):10.2f}{d:7.3f}{sharpe(bh[keep]) * np.sqrt(ANN):7.2f}")


def stratified_block_bootstrap_diff(r: np.ndarray, bench: np.ndarray, flag: np.ndarray,
                                    block: int = 20, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Block bootstrap that resamples episode days and other days separately (keeps the share of
    episode days fixed). Returns the 95% CI of the annualised Sharpe difference vs. the benchmark."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(flag), np.flatnonzero(~flag)]
    diffs = []
    for _ in range(n_boot):
        idx = []
        for g in groups:
            if len(g) == 0:
                continue
            nb = int(np.ceil(len(g) / block))
            st = rng.integers(0, len(g), nb)
            idx.append(g[((st[:, None] + np.arange(block)[None, :]).ravel()[:len(g)]) % len(g)])
        idx = np.concatenate(idx)
        diffs.append((sharpe(r[idx]) - sharpe(bench[idx])) * np.sqrt(ANN))
    return np.percentile(diffs, [2.5, 97.5])


def persistence_test(rets: pd.DataFrame, bh: pd.Series, train_len: int, test_len: int, episodes) -> None:
    """Rank correlation between rules' in-sample and next-window excess Sharpe.
    No edge -> ~0 everywhere. Regime-dependent edge -> positive when the test window has no
    episode, ~0 or negative when it does. Random rules are the built-in placebo."""
    flag = episode_flag(rets.index, episodes)
    ex = rets.sub(bh, axis=0)
    is_rand = ex.columns.str.startswith("rand_")
    rows = []
    for tr, te in purged_splits(len(ex), test_len, 1, train_len=train_len):
        a, b = ex.iloc[tr].apply(sharpe), ex.iloc[te].apply(sharpe)
        rows.append({"test start": ex.index[te[0]].date(),
                     "test has episode": bool(flag.iloc[te].any()),
                     "MA rules": stats.spearmanr(a[~is_rand], b[~is_rand])[0],
                     "random rules": stats.spearmanr(a[is_rand], b[is_rand])[0]})
    tab = pd.DataFrame(rows)
    print(tab.round(2).to_string(index=False))
    print("\nmean rank correlation:")
    print(tab.groupby("test has episode")[["MA rules", "random rules"]].agg(["mean", "count"]).round(2))


# ============================================================ structural-break onsets only
def load_onsets(path: str) -> list[pd.Timestamp]:
    """CSV with a 'start' column (GSADF origination times, UTC). Each onset is mapped to the
    first daily close (00:00 UTC) at or after it, i.e. the first bar whose return can see it.
    Used ONLY to describe the data, never inside a trading rule."""
    t = pd.to_datetime(pd.read_csv(path)["start"], utc=True)
    return [x.tz_convert(None).ceil("D") for x in t]


def onset_persistence(rets: pd.DataFrame, bh: pd.Series, train_len: int, test_len: int, onsets) -> None:
    """Rank correlation of rules' information ratio between the training window and the next
    test window, classified by where a structural-break onset falls:
      'onset in test'  : the regime changes after the rule was chosen
      'onset in train' : the training window already contains a change
      'no onset'       : neither window contains an onset
    No edge -> ~0 everywhere; regime-dependent edge -> positive without an onset in the test
    window, ~0 or negative with one. Random rules are the built-in placebo."""
    ex = rets.sub(bh, axis=0)
    is_rand = ex.columns.str.startswith("rand_")
    rows = []
    for tr, te in purged_splits(len(ex), test_len, 1, train_len=train_len):
        a0, a1, b1 = ex.index[tr[0]], ex.index[tr[-1]], ex.index[te[-1]]
        in_test = any(a1 < o <= b1 for o in onsets)
        in_train = any(a0 <= o <= a1 for o in onsets)
        group = ("onset in test" if in_test else "onset in train" if in_train else "no onset") if onsets \
            else "all windows"
        a, b = ex.iloc[tr].apply(sharpe), ex.iloc[te].apply(sharpe)
        rows.append({"test start": ex.index[te[0]].date(), "group": group,
                     "MA rules": stats.spearmanr(a[~is_rand], b[~is_rand])[0],
                     "random rules": stats.spearmanr(a[is_rand], b[is_rand])[0]})
    tab = pd.DataFrame(rows)
    print(tab.round(2).to_string(index=False))
    print("\nmean rank correlation by group:")
    print(tab.groupby("group")[["MA rules", "random rules"]].agg(["mean", "count"]).round(2))
    return tab


def appraisal_ratio(r: pd.DataFrame, bh: pd.Series) -> pd.Series:
    """Exposure-neutral score of every rule: regress its return on buy-and-hold in the window,
    r = a + b * bh + e, and return a / s.d.(e). Rules that differ only in how long they hold
    the asset get the same score, so the ranking reflects timing, not exposure x drift."""
    x = bh.to_numpy() - bh.mean()
    y = r.to_numpy() - r.mean().to_numpy()
    b = (x @ y) / (x @ x) if (x @ x) > 0 else np.zeros(y.shape[1])
    alpha = r.mean().to_numpy() - b * bh.mean()
    sd = (y - np.outer(x, b)).std(axis=0, ddof=2)
    return pd.Series(np.where(sd > 0, alpha / np.where(sd > 0, sd, 1), 0.0), index=r.columns)


def window_rank_corr(rets: pd.DataFrame, bh: pd.Series, train_len: int, test_len: int,
                     metric: str = "ir") -> list[float]:
    """Per walk-forward window: rank correlation of the rules' score in the training window and
    in the next test window. metric 'ir': information ratio vs buy-and-hold; 'ar': appraisal ratio."""
    out = []
    for tr, te in purged_splits(len(rets), test_len, 1, train_len=train_len):
        if metric == "ir":
            ex = rets.sub(bh, axis=0)
            a, b = ex.iloc[tr].apply(sharpe), ex.iloc[te].apply(sharpe)
        else:
            a = appraisal_ratio(rets.iloc[tr], bh.iloc[tr])
            b = appraisal_ratio(rets.iloc[te], bh.iloc[te])
        out.append(stats.spearmanr(a, b)[0])
    return out


def drift_check(rets: pd.DataFrame, bh: pd.Series, train_len: int, test_len: int) -> pd.DataFrame:
    """Is the rank reversal just exposure x drift? For each window: direction of buy-and-hold in
    the training and test windows, and the MA-rule rank correlation with the information ratio
    (exposure-sensitive) and with the appraisal ratio (exposure-neutral)."""
    ma = rets.loc[:, ~rets.columns.str.startswith("rand_")]
    ir_rc = window_rank_corr(ma, bh, train_len, test_len, "ir")
    ar_rc = window_rank_corr(ma, bh, train_len, test_len, "ar")
    rows = []
    for (tr, te), a, b in zip(purged_splits(len(ma), test_len, 1, train_len=train_len), ir_rc, ar_rc):
        r_tr, r_te = np.log1p(bh.iloc[tr]).sum(), np.log1p(bh.iloc[te]).sum()
        rows.append({"test start": ma.index[te[0]].date(), "BH train": r_tr, "BH test": r_te,
                     "same direction": bool(np.sign(r_tr) == np.sign(r_te)),
                     "IR rank corr": a, "AR rank corr": b})
    tab = pd.DataFrame(rows)
    print(tab.round(2).to_string(index=False))
    g = tab.groupby("same direction")[["IR rank corr", "AR rank corr"]].agg(["mean", "count"])
    print("\nmean rank correlation by whether the training and test windows moved the same way:")
    print(g.round(2))
    rho = stats.spearmanr(tab["same direction"].astype(float), tab["IR rank corr"])[0]
    print(f"rank correlation between 'same direction' and the IR rank correlation: {rho:+.2f}")
    print(f"mean over all windows: IR {tab['IR rank corr'].mean():+.2f}, AR {tab['AR rank corr'].mean():+.2f}")
    return tab


def persistence_null(close: pd.Series, cost_bps: float, train_len: int, test_len: int,
                     n_sim: int = 200, block: int = 20, seed: int = 0, metric: str = "ir") -> np.ndarray:
    """Rank correlation of MA rules between training and test windows on price paths with no
    edge: non-overlapping blocks of the observed daily returns are shuffled (keeps the return
    distribution and dependence within a block, removes persistent structure beyond it).
    Returns (n_sim, n_windows)."""
    rng = np.random.default_rng(seed)
    r = close.pct_change().dropna().to_numpy()
    n = len(r)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    out = []
    for _ in range(n_sim):
        # shuffle non-overlapping blocks WITHOUT replacement: drawing blocks with replacement
        # repeats the same multi-day segments in training and test windows, which creates
        # spurious persistence (the baseline would be biased upward)
        idx = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        path = pd.Series(close.iloc[0] * np.concatenate([[1.0], np.cumprod(1 + r[idx])]), index=close.index)
        rets = pd.DataFrame({k: strategy_returns(path, v, cost_bps) for k, v in ma_rules(path).items()}).iloc[200:]
        bhp = path.pct_change().reindex(rets.index).fillna(0.0)
        out.append(window_rank_corr(rets, bhp, train_len, test_len, metric))
    return np.array(out)


def compare_with_null(tab: pd.DataFrame, null: np.ndarray, label: str = "") -> None:
    """Observed mean MA rank correlation per group vs. the same windows on no-edge paths."""
    print(f"no-edge baseline{label} ({null.shape[0]} shuffled paths): mean rank correlation of MA rules "
          f"per window {null.mean():+.2f} (5-95%: {np.percentile(null, 5):+.2f} to {np.percentile(null, 95):+.2f})")
    print(f"{'group':16s}{'windows':>8s}{'observed':>10s}{'null mean':>11s}{'null 5-95%':>18s}{'P(null<=obs)':>14s}")
    groups = [("all windows", np.ones(len(tab), bool))] + \
             [(g, (tab["group"] == g).to_numpy()) for g in sorted(tab["group"].unique()) if g != "all windows"]
    for g, m in groups:
        obs = tab.loc[m, "MA rules"].mean()
        nm = null[:, m].mean(axis=1)
        lo, hi = np.percentile(nm, [5, 95])
        print(f"{g:16s}{int(m.sum()):8d}{obs:+10.2f}{nm.mean():+11.2f}{lo:+9.2f} to {hi:+.2f}{np.mean(nm <= obs):14.2f}")


# ============================================================ report
def run(close: pd.Series, family: str, cost_bps: float, train_len: int, test_len: int,
        trials_csv: str, outdir: str, episodes=None, onsets=None, legacy_random: bool = False,
        n_null: int = 0, asset: str = "") -> None:
    rules = {}
    if family in ("ma", "both"):
        rules |= ma_rules(close)
    if family in ("random", "both"):
        rules |= (random_rules_legacy if legacy_random else random_rules)(close.index)
    rets = pd.DataFrame({k: strategy_returns(close, v, cost_bps) for k, v in rules.items()})
    rets = rets.iloc[200:]                               # drop the warm-up of the longest MA
    bh = close.pct_change().reindex(rets.index).fillna(0.0)
    print(f"period {rets.index.min().date()} to {rets.index.max().date()}  ({len(rets)} days), "
          f"{rets.shape[1]} rules, cost {cost_bps} bp per unit turnover")

    # 3. trial log (raw and excess over buy-and-hold)
    excess = rets.sub(bh, axis=0)                        # 7. benchmark built into the test
    sr_raw, sr_ex = rets.apply(sharpe), excess.apply(sharpe)
    log = pd.DataFrame({"sharpe_annual": sr_raw * np.sqrt(ANN),
                        "excess_sharpe_annual": sr_ex * np.sqrt(ANN),
                        "mean_daily": rets.mean(), "mean_excess_daily": excess.mean()})
    os.makedirs(os.path.dirname(trials_csv) or ".", exist_ok=True)
    log.sort_values("excess_sharpe_annual", ascending=False).to_csv(trials_csv)
    print(f"trial log: {trials_csv}")

    # 4. best rule against buy-and-hold: PSR and DSR on excess returns
    best = sr_ex.idxmax()
    n_raw, n_eff = len(sr_ex), effective_trials(excess)
    print(f"\n== Full-sample search, excess return over buy-and-hold (in-sample)")
    print(f"best rule: {best}  Sharpe {sr_raw[best] * np.sqrt(ANN):.2f} (buy-and-hold "
          f"{sharpe(bh) * np.sqrt(ANN):.2f}), excess-return Sharpe {sr_ex[best] * np.sqrt(ANN):.2f}")
    print(f"PSR of excess vs 0     : {psr(excess[best].to_numpy()):.3f}")
    for label, n in [("raw", n_raw), ("effective", n_eff)]:
        d, sr0 = dsr(excess[best].to_numpy(), sr_ex.to_numpy(), n)
        print(f"DSR, {label:9s} N={n:7.1f}: {d:.3f}   (expected max excess SR under null "
              f"{sr0 * np.sqrt(ANN):.2f} annualised)")
    best_raw = sr_raw.idxmax()
    d0, _ = dsr(rets[best_raw].to_numpy(), sr_raw.to_numpy(), n_raw)
    print(f"(reference, zero benchmark: best {best_raw} Sharpe {sr_raw[best_raw] * np.sqrt(ANN):.2f}, "
          f"DSR {d0:.3f} - too lenient when rules share the market drift)")

    # 2 + 5 + 7. walk-forward selection on excess returns, bootstrap vs buy-and-hold
    oos, picks = walk_forward_selection(rets, train_len, test_len, bench=bh)
    bh_oos = bh.reindex(oos.index)
    bs = block_bootstrap_sharpe(oos.to_numpy(), bench=bh_oos.to_numpy())
    print(f"\n== Walk-forward selection on excess return (train {train_len}d, test {test_len}d, "
          f"{len(picks)} windows)")
    print(f"OOS annualised Sharpe {sharpe(oos) * np.sqrt(ANN):.2f}  95% CI {bs['sr_ci'].round(2)}  "
          f"P(SR<=0) {bs['p_sr_le_0']:.2f}")
    print(f"buy-and-hold same days {sharpe(bh_oos) * np.sqrt(ANN):.2f}; difference 95% CI "
          f"{bs['diff_ci'].round(2)}  P(diff<=0) {bs['p_diff_le_0']:.2f}")
    print(f"distinct rules picked: {len(set(picks))} of {len(picks)} windows")

    if n_null > 0:
        print("\n== Drift check: exposure x drift vs. timing (MA rules)")
        drift_check(rets, bh, train_len, test_len)
        obs_ar = np.mean(window_rank_corr(rets.loc[:, ~rets.columns.str.startswith("rand_")], bh,
                                          train_len, test_len, "ar"))
        null_ar = persistence_null(close, cost_bps, train_len, test_len, n_sim=n_null, block=1, metric="ar")
        nm = null_ar.mean(axis=1)
        print(f"exposure-neutral (appraisal ratio) persistence: observed {obs_ar:+.2f}, no-edge baseline "
              f"(block 1, {n_null} paths) mean {nm.mean():+.2f}, 5-95% {np.percentile(nm, 5):+.2f} .. "
              f"{np.percentile(nm, 95):+.2f}, P(base<=obs) {np.mean(nm <= obs_ar):.2f}")

    if n_null > 0 and not onsets:
        # no onsets given: every window is in one group ("no onset" = all windows)
        print("\n== Persistence of rules across walk-forward windows (all windows)")
        tab = onset_persistence(rets, bh, train_len, test_len, [])
        print("\n== MA-rule persistence vs. a no-edge baseline")
        for b in (1, 20):
            compare_with_null(tab, persistence_null(close, cost_bps, train_len, test_len,
                                                    n_sim=n_null, block=b), f", block {b}")

    if onsets:
        print(f"\n== Persistence around structural-break onsets ({len(onsets)} onsets, ex-post dates)")
        print("note: onsets only classify the walk-forward windows after the fact; rule selection, positions,")
        print("      DSR, walk-forward Sharpe and the figures above are identical with or without --onsets")
        print("onsets mapped to daily closes:", ", ".join(str(o.date()) for o in onsets))
        tab = onset_persistence(rets, bh, train_len, test_len, onsets)
        if n_null > 0:
            print("\n== MA-rule persistence vs. a no-edge baseline")
            # block 1: no serial dependence at all; block 20: keeps volatility clustering within
            # 20 days. Both remove any persistent edge; showing both checks the sensitivity.
            for b in (1, 20):
                compare_with_null(tab, persistence_null(close, cost_bps, train_len, test_len,
                                                        n_sim=n_null, block=b), f", block {b}")

    if episodes:
        flag = episode_flag(oos.index, episodes).to_numpy()
        ci = stratified_block_bootstrap_diff(oos.to_numpy(), bh_oos.to_numpy(), flag)
        print(f"difference 95% CI, bootstrap stratified by episode days: {ci.round(2)}")
        print("\n== Attribution: episode days vs. other days (ex-post dates, description only)")
        episode_attribution(bh, rets[best], oos, episodes)
        print("\n== Leave one episode out (full search repeated)")
        leave_one_episode_out(rets, bh, episodes)
        print("\n== Persistence: in-sample vs next-window rank correlation of excess Sharpe")
        persistence_test(rets, bh, train_len, test_len, episodes)

    # figures
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(sr_ex * np.sqrt(ANN), bins=40, color="#9aa7b8")
    ax[0].axvline(sr_ex[best] * np.sqrt(ANN), color="#c0392b", lw=2, label=f"best ({best})")
    _, sr0 = dsr(excess[best].to_numpy(), sr_ex.to_numpy(), n_raw)
    ax[0].axvline(sr0 * np.sqrt(ANN), color="#333", ls="--", label="expected max under null")
    ax[0].axvline(0, color="#999", lw=0.8)
    ax[0].set_xlabel("annualised Sharpe of excess return over buy-and-hold (in-sample)")
    tag = f"{asset}: " if asset else ""
    ax[0].set_title(f"{tag}{n_raw} rules tried")
    ax[0].legend(fontsize=8)
    for ser, lab, col in [(rets[best], f"best vs buy-and-hold: {best}", "#c0392b"),
                          (oos, "walk-forward OOS", "#2f6fd1"), (bh, "buy-and-hold", "#555")]:
        ax[1].plot((1 + ser).cumprod(), label=lab, color=col, lw=1.3)
    ax[1].set_yscale("log")
    ax[1].set_title(f"{tag}cumulative growth")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, f"yardstick_{asset}.png" if asset else "yardstick.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def selftest(reps: int = 300, n_rules: int = 100, t: int = 1000, seed: int = 0) -> None:
    """Calibrate the yardstick on simulated data where the truth is known."""
    rng = np.random.default_rng(seed)
    psr_hit = dsr_hit = power = 0
    for _ in range(reps):
        x = rng.standard_t(5, size=(t, n_rules)) * 0.01
        srs = np.array([sharpe(x[:, j]) for j in range(n_rules)])
        best = x[:, srs.argmax()]
        psr_hit += psr(best) > 0.95
        dsr_hit += dsr(best, srs, n_rules)[0] > 0.95
        x[:, 0] += x[:, 1:].std() * 2 / np.sqrt(ANN)               # one rule with true annual SR = 2
        srs = np.array([sharpe(x[:, j]) for j in range(n_rules)])
        power += dsr(x[:, srs.argmax()], srs, n_rules)[0] > 0.95
    print(f"no edge, best of {n_rules} rules, {t} days:")
    print(f"  PSR > 0.95 in {psr_hit / reps:.1%} of runs   (naive test: almost always 'significant')")
    print(f"  DSR > 0.95 in {dsr_hit / reps:.1%} of runs   (nominal 5%; below 5% means conservative)")
    print(f"one rule with true annual SR 2 among {n_rules}: DSR > 0.95 in {power / reps:.1%} of runs (power)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="calibrate PSR/DSR on simulated data")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--product", default="BTC-USD", help="Coinbase product, e.g. ETH-USD")
    ap.add_argument("--end", default=None)
    ap.add_argument("--price-csv", default=None)
    ap.add_argument("--family", choices=["ma", "random", "both"], default="both")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit of turnover, in basis points")
    ap.add_argument("--train", type=int, default=365)
    ap.add_argument("--test", type=int, default=90)
    ap.add_argument("--trials-csv", default="data/trials.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--episodes", default=None, help="CSV start,end of structural-break episodes (description only)")
    ap.add_argument("--legacy-random", action="store_true",
                    help="old random-rule generator (to reproduce earlier published numbers)")
    ap.add_argument("--persistence-null", type=int, default=0, metavar="N",
                    help="with --onsets: N block-shuffled paths for the MA rank-correlation baseline")
    ap.add_argument("--onsets", default=None, help="CSV 'start' of structural-break onsets, end ignored (description only)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    close = make_demo() if args.demo else load_daily_close(args.start, args.end, args.price_csv, args.product)
    asset = "demo" if args.demo else args.product
    print(f"asset: {asset}")
    episodes = load_episodes(args.episodes) if args.episodes else None
    onsets = load_onsets(args.onsets) if args.onsets else None
    run(close, args.family, args.cost_bps, args.train, args.test, args.trials_csv, args.outdir, episodes, onsets,
        args.legacy_random, args.persistence_null, asset)


if __name__ == "__main__":
    main()
