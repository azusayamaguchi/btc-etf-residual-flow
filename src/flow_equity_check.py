"""
Are the recent BTC ETF outflows larger than the equity sell-off can explain?

Flow model (estimated on an expanding window, only days before t):
    base   : F_t ~ BTC return (t, t-1..t-3) + flows (t-1..t-3)                (as in article 2)
    equity : base + same-day S&P 500 return, SOX return, change in VIX       (all at the 16:00 ET close)
The recent residual  F_t - Fhat_t  is the part of the flow that the model (BTC move, equity
move, past flows) did not predict. Its sum over the recent window is compared with every past window of the
same length.

Data: flows from Farside (farside_to_csv.py), BTC at 16:00 ET from Coinbase 1h candles,
S&P 500 (^GSPC), Philadelphia Semiconductor index (^SOX) and VIX (^VIX) daily closes from Yahoo.

    python src/flow_equity_check.py --flows data/farside_btc_flows.csv --recent 20
    python src/flow_equity_check.py --demo
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from btc_etf_flow_analysis import _ols_hac, build_panel, load_flows, load_hourly, load_price_at_close

TICKERS = {"spx": "^GSPC", "sox": "^SOX", "vix": "^VIX"}


# ------------------------------------------------------------ data
def load_equity(start: str, end: str | None) -> pd.DataFrame:
    """Daily closes from Yahoo -> same-day log returns of S&P 500 and SOX, change in VIX level."""
    import yfinance as yf
    px = yf.download(list(TICKERS.values()), start=start, end=end, auto_adjust=False,
                     progress=False)["Close"].rename(columns={v: k for k, v in TICKERS.items()})
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    return pd.DataFrame({"spx_ret": np.log(px["spx"]).diff(), "sox_ret": np.log(px["sox"]).diff(),
                         "vix_chg": px["vix"].diff()}).dropna()


def make_demo(n: int = 700, seed: int = 0):
    """Synthetic: flows chase BTC and equities; the last 20 days add an extra outflow."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-10", periods=n)
    spx = rng.normal(0, 0.01, n)
    sox = 1.6 * spx + rng.normal(0, 0.012, n)
    vix = -120 * spx + rng.normal(0, 0.8, n)
    btc = 0.8 * spx + rng.normal(0, 0.03, n)
    flow = np.zeros(n)
    for t in range(1, n):
        flow[t] = 4000 * btc[t] + 3000 * btc[t - 1] + 6000 * spx[t] + 0.3 * flow[t - 1] + rng.normal(0, 200)
    flow[-20:] -= 150                                    # unexplained selling in the last 20 days
    price = pd.Series(60000 * np.exp(np.cumsum(btc)), index=days, name="price")
    eq = pd.DataFrame({"spx_ret": spx, "sox_ret": sox, "vix_chg": vix}, index=days)
    return pd.DataFrame({"Total": flow}, index=days), price, eq


# ------------------------------------------------------------ model
def design(df: pd.DataFrame, equity: bool, p: int = 3) -> pd.DataFrame:
    cols = {f"ret_l{k}": df["ret"].shift(k) for k in range(0, p + 1)} | \
           {f"flow_l{k}": df["flow"].shift(k) for k in range(1, p + 1)}
    if equity:
        cols |= {"spx_ret": df["spx_ret"], "sox_ret": df["sox_ret"], "vix_chg": df["vix_chg"]}
    X = pd.concat(cols, axis=1)
    X.insert(0, "const", 1.0)
    return X


def oos_predict(df: pd.DataFrame, equity: bool, min_obs: int = 120) -> pd.Series:
    """Fhat_t from coefficients estimated on days < t (expanding window)."""
    X, y = design(df, equity), df["flow"]
    ok = X.notna().all(axis=1) & y.notna()
    Xv, yv, idx = X[ok].to_numpy(), y[ok].to_numpy(), X[ok].index
    pred = pd.Series(np.nan, index=df.index)
    for i in range(min_obs, len(yv)):
        beta, *_ = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)
        pred.loc[idx[i]] = Xv[i] @ beta
    return pred


def oos_r2(y: pd.Series, pred: pd.Series) -> float:
    m = pred.notna()
    return 1 - ((y[m] - pred[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()


# ------------------------------------------------------------ report
def run(df: pd.DataFrame, recent: int, outdir: str) -> None:
    print(f"period {df.index.min().date()} to {df.index.max().date()} ({len(df)} trading days)")
    p_base, p_eq = oos_predict(df, False), oos_predict(df, True)
    print(f"\n== Flow model, out-of-sample R2:  base {oos_r2(df['flow'], p_base):.1%}   "
          f"+ equity {oos_r2(df['flow'], p_eq):.1%}")
    res = _ols_hac(df["flow"], design(df, True).drop(columns="const"))
    print("equity terms in the full-sample fit ($m per unit; HAC t):")
    for c, unit in [("spx_ret", "1% S&P 500"), ("sox_ret", "1% SOX"), ("vix_chg", "1 pt VIX")]:
        scale = 0.01 if c != "vix_chg" else 1.0
        print(f"  {unit:11s}: {res.params[c] * scale:+8.1f}  (t={res.tvalues[c]:+.2f})")

    eps = (df["flow"] - p_eq).dropna()
    rec = eps.iloc[-recent:]
    hist = eps.iloc[:-recent]
    sums = hist.rolling(recent).sum().dropna()               # every past window of the same length
    pct = (sums <= rec.sum()).mean()
    print(f"\n== Last {recent} trading days ({rec.index.min().date()} to {rec.index.max().date()})")
    tab = pd.DataFrame({"flow": df["flow"].loc[rec.index], "predicted": p_eq.loc[rec.index],
                        "residual": rec})
    print(tab.round(0).to_string())
    print(f"\nsum: flow {tab.flow.sum():+.0f}  model prediction (BTC + equities) {tab.predicted.sum():+.0f}  "
          f"residual {tab.residual.sum():+.0f}  ($m)")
    print(f"the residual sum is lower than or equal to {pct:.0%} of the {len(sums)} past "
          f"{recent}-day windows   (small = unusually large outflow beyond the model prediction)")

    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(tab))
    ax[0].bar(x - 0.2, tab.flow, 0.4, label="actual flow", color="#555")
    ax[0].bar(x + 0.2, tab.predicted, 0.4, label="model prediction (BTC + equities)", color="#9aa7b8")
    ax[0].set_xticks(x[::4], [d.strftime("%m-%d") for d in tab.index[::4]])
    ax[0].axhline(0, color="#999", lw=0.8)
    ax[0].set_title(f"Last {recent} trading days ($m)")
    ax[0].legend(fontsize=8)
    ax[1].hist(sums, bins=40, color="#9aa7b8", label=f"past {recent}-day windows")
    ax[1].axvline(rec.sum(), color="#c0392b", lw=2, label=f"last {recent} days")
    ax[1].set_title(f"Residual flow (actual - prediction) summed over {recent} days")
    ax[1].set_xlabel("$m")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "flow_equity_check.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flows", help="Farside BTC flow CSV")
    ap.add_argument("--start", default="2024-01-11")
    ap.add_argument("--end", default=None)
    ap.add_argument("--recent", type=int, default=20, help="trading days in the recent window")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        flows, price, eq = make_demo()
    else:
        if not args.flows:
            raise SystemExit("--flows is required (python src/farside_to_csv.py --asset BTC)")
        flows = load_flows(args.flows)
        price = load_price_at_close(load_hourly(args.start, args.end, product="BTC-USD"))
        eq = load_equity(args.start, args.end)
    df = build_panel(flows, price).join(eq, how="inner")
    run(df, args.recent, args.outdir)


if __name__ == "__main__":
    main()
