"""
Raw flows don't predict, residuals do: US spot Bitcoin ETF flows vs. BTC returns.

Usage
-----
    # Offline demo with synthetic data (no network)
    python src/btc_etf_flow_analysis.py --demo

    # Real data: flows from a Farside CSV (see farside_to_csv.py), prices from Coinbase
    python src/btc_etf_flow_analysis.py --flows data/farside_btc_flows.csv --start 2024-01-11

    # Real data, scraping the recent Farside table directly
    python src/btc_etf_flow_analysis.py --start 2026-01-01

Requirements: pandas numpy scipy statsmodels matplotlib requests beautifulsoup4
"""
from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ET = "America/New_York"
MISSING = {"-", "", "—", "–"}


# ============================================================ data loading
def _parse_num(x) -> float:
    """Farside notation -> float: '(484.9)' -> -484.9, '1,234.5' -> 1234.5, '-' -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in MISSING:
        return np.nan
    neg = bool(re.fullmatch(r"\(.*\)", s))
    v = float(s.strip("()"))
    return -v if neg else v


def load_flows(path: str) -> pd.DataFrame:
    """Daily flows from CSV with columns Date, <ETF tickers...>, Total (US$m)."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("Date").sort_index()
    df = df.apply(lambda c: c.map(_parse_num))
    if "Total" not in df.columns:
        df["Total"] = df.sum(axis=1)
    return df.dropna(subset=["Total"])


def load_farside_flows(start: str, end: str | None) -> pd.DataFrame:
    """Scrape total daily net flows (US$m) from the recent Farside table (https://farside.co.uk/btc/)."""
    import requests
    from bs4 import BeautifulSoup

    r = requests.get("https://farside.co.uk/btc/", timeout=30,
                     headers={"User-Agent": "Mozilla/5.0 (compatible; data-analysis/1.0)"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    date_re = re.compile(r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$")
    records = []
    for row in soup.select("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", " ").strip()
                 for c in row.find_all(["td", "th"], recursive=False)]
        if len(cells) < 3 or not date_re.fullmatch(cells[0]):
            continue
        # Holidays / unreported days: every ETF cell is '-' but Total shows 0.0 -> skip.
        if not [v for v in cells[1:-1] if v not in MISSING] or cells[-1] in MISSING:
            continue
        total = _parse_num(cells[-1])
        if pd.notna(total):
            records.append((pd.to_datetime(cells[0], format="%d %b %Y"), total))
    if not records:
        raise ValueError("Could not parse Farside flows; the page layout may have changed.")
    flows = (pd.DataFrame(records, columns=["Date", "Total"])
             .drop_duplicates("Date", keep="last").set_index("Date").sort_index())
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end else pd.Timestamp.now(tz=ET).tz_localize(None).normalize() + pd.Timedelta(days=1)
    flows = flows[(flows.index >= lo) & (flows.index < hi)]
    if flows.empty:
        raise ValueError("No Farside flows in the requested period.")
    return flows


def load_hourly(start: str, end: str | None) -> pd.Series:
    """Coinbase BTC-USD hourly closes, indexed by candle END time (ET). Completed candles only."""
    import requests

    start_utc = pd.Timestamp(start, tz=ET).tz_convert("UTC")
    end_utc = (pd.Timestamp(end, tz=ET) if end else pd.Timestamp.now(tz=ET)).tz_convert("UTC")
    if end_utc <= start_utc:
        raise ValueError("end must be later than start")

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    rows = []
    with requests.Session() as session:
        t = start_utc
        while t < end_utc:
            stop = min(t + pd.Timedelta(hours=299), end_utc)   # API returns at most 300 candles
            r = session.get(url, timeout=30, params={
                "start": t.isoformat(), "end": stop.isoformat(), "granularity": 3600})
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Coinbase API error: {data}")
            rows.extend(data)
            t = stop
    if not rows:
        raise ValueError("Coinbase returned no candles in the requested range.")

    c = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    c["time"] = pd.to_datetime(c["time"], unit="s", utc=True)
    c = c.drop_duplicates("time").set_index("time").sort_index()
    c = c[(c.index >= start_utc) & (c.index < end_utc)]
    c.index = c.index.tz_convert(ET) + pd.Timedelta(hours=1)        # start time -> end time
    c = c[c.index <= pd.Timestamp.now(tz=ET)]                         # drop the unfinished candle
    return c["close"].astype(float).rename("hourly")


def load_price_at_close(hourly: pd.Series) -> pd.Series:
    """Daily BTC price at 16:00 ET (close of the 15:00-16:00 candle), the ETF NAV strike time."""
    at16 = hourly[(hourly.index.hour == 16) & (hourly.index.minute == 0)]
    price = at16.copy()
    price.index = at16.index.normalize().tz_localize(None)
    return price.rename("price")


def make_demo(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic data: flows chase yesterday's return, and flows weakly lead returns."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-03-03", periods=n)
    ret, flow = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        ret[t] = 0.0002 + 0.00002 * flow[t - 1] + rng.normal(0, 0.025)
        flow[t] = 3000 * ret[t - 1] + 0.3 * flow[t - 1] + rng.normal(0, 150)
    price = pd.Series(60000 * np.exp(np.cumsum(ret)), index=days, name="price")
    return pd.DataFrame({"Total": flow}, index=days), price


# ============================================================ panel and basic statistics
def build_panel(flows: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    df = flows[["Total"]].rename(columns={"Total": "flow"}).join(price, how="inner")
    df["ret"] = np.log(df["price"]).diff()            # previous trading day 16:00 -> today 16:00
    df["flow_5d"] = df["flow"].rolling(5).sum()
    df["ret_5d"] = df["ret"].rolling(5).sum()
    return df.dropna()


def lead_lag(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """Correlation of today's flow with the return k days later (k<0: earlier returns)."""
    rows = []
    for k in range(-max_lag, max_lag + 1):
        x, y = df["flow"], df["ret"].shift(-k)
        m = x.notna() & y.notna()
        rho, p = stats.spearmanr(x[m], y[m])
        rows.append({"lag": k, "pearson": x[m].corr(y[m]), "spearman": rho, "p_spearman": p})
    return pd.DataFrame(rows).set_index("lag")


def _ols_hac(y: pd.Series, X: pd.DataFrame, maxlags: int = 5):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    return sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})


def regressions(df: pd.DataFrame) -> None:
    """Same-day, leading and chasing regressions with HAC (Newey-West) standard errors."""
    z = df.assign(flow_100m=df["flow"] / 100)       # coefficients per US$100m
    specs = {
        "same day  ret_t   ~ flow_t ": (z["ret"], z[["flow_100m"]]),
        "leading   ret_t+1 ~ flow_t ": (z["ret"].shift(-1), z[["flow_100m"]]),
        "chasing   flow_t  ~ ret_t-1": (z["flow"], z[["ret"]].shift(1)),
    }
    for name, (y, X) in specs.items():
        res = _ols_hac(y, X)
        print(f"{name}  beta={res.params.iloc[1]: .5f}  t={res.tvalues.iloc[1]: .2f}  "
              f"R2={res.rsquared:.3f}  n={int(res.nobs)}")


# ============================================================ residual flow
def flow_residual(df: pd.DataFrame, p: int = 3, min_obs: int = 120) -> pd.Series:
    """
    Out-of-sample residual flow  eps_t = F_t - Fhat_t.

    Fhat_t = a + sum_{k=0..p} b_k r_{t-k} + sum_{k=1..p} c_k F_{t-k}
    Coefficients for day t are estimated on data up to t-1 only (expanding window).
    Note: r_t (today's return) is a regressor, so Fhat_t is conditional on today's
    16:00 price, not a forecast made at the start of the day.
    """
    X = pd.concat({f"ret_l{k}": df["ret"].shift(k) for k in range(0, p + 1)} |
                  {f"flow_l{k}": df["flow"].shift(k) for k in range(1, p + 1)}, axis=1)
    X.insert(0, "const", 1.0)
    y = df["flow"]
    ok = X.notna().all(axis=1) & y.notna()
    Xv, yv, idx = X[ok].to_numpy(), y[ok].to_numpy(), X[ok].index
    resid = pd.Series(np.nan, index=df.index, name="residual_flow")
    for i in range(min_obs, len(yv)):
        beta, *_ = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)   # fit on days < t
        resid.loc[idx[i]] = yv[i] - Xv[i] @ beta
    return resid


def r2_decomposition(df: pd.DataFrame, p: int = 3) -> None:
    """In-sample R^2 of the flow model as regressor groups are added (order matters)."""
    lag_ret = {f"ret_l{k}": df["ret"].shift(k) for k in range(1, p + 1)}
    same = {"ret_l0": df["ret"]}
    lag_flow = {f"flow_l{k}": df["flow"].shift(k) for k in range(1, p + 1)}
    steps = [("past returns (t-1..t-3)", lag_ret),
             ("+ same-day return (t)", lag_ret | same),
             ("+ past flows (t-1..t-3)", lag_ret | same | lag_flow)]
    prev = 0.0
    for name, cols in steps:
        d = pd.concat([df["flow"].rename("y"), pd.DataFrame(cols)], axis=1).dropna()
        r2 = sm.OLS(d["y"], sm.add_constant(d.drop(columns="y"))).fit().rsquared
        print(f"{name:26s} R2={r2:.3f}  (increment {r2 - prev:+.3f})")
        prev = r2


def residual_test(df: pd.DataFrame) -> None:
    """Does raw flow vs. residual flow predict next-day returns?"""
    z = df.assign(residual=flow_residual(df), ret_next=df["ret"].shift(-1))
    z = z.dropna(subset=["residual", "ret_next"])
    oos_r2 = 1 - z["residual"].var() / z["flow"].var()
    print(f"Out-of-sample R2 of the flow model, 1 - Var(eps)/Var(F): {oos_r2:.1%}")
    for col, label in [("flow", "raw flow     "), ("residual", "residual flow")]:
        res = _ols_hac(z["ret_next"], z[[col]] / 100)
        rho, _ = stats.spearmanr(z[col], z["ret_next"])
        print(f"{label} -> next-day return  beta={res.params.iloc[1] * 100: .3f}% per $100m  "
              f"t={res.tvalues.iloc[1]: .2f}  spearman={rho: .3f}  n={int(res.nobs)}")


# ============================================================ "divergence": outflow but price up
def divergences(df: pd.DataFrame) -> pd.DataFrame:
    """Days with negative 5-day cumulative flow but positive 5-day return."""
    return df[(df["flow_5d"] < 0) & (df["ret_5d"] > 0)][["price", "flow_5d", "ret_5d"]]


def divergence_test(df: pd.DataFrame, horizon: int = 5, n_perm: int = 5000, seed: int = 0) -> None:
    """
    Forward `horizon`-day return after divergence days vs. other days.
    Divergence days cluster and forward windows overlap, so a plain t-test overstates
    precision. Instead, circularly shift the divergence-day pattern along the calendar
    (preserving its clustering) and compare the observed difference with the shifted ones.
    """
    fwd = df["ret"].rolling(horizon).sum().shift(-horizon)
    flag = (df["flow_5d"] < 0) & (df["ret_5d"] > 0)
    m = fwd.notna()
    fwd, flag = fwd[m].to_numpy(), flag[m].to_numpy()

    diff = fwd[flag].mean() - fwd[~flag].mean()
    episodes = int((np.diff(np.r_[0, flag.astype(int)]) == 1).sum())
    rng = np.random.default_rng(seed)
    shifts = rng.integers(horizon * 2, len(fwd) - horizon * 2, n_perm)
    null = np.array([fwd[np.roll(flag, s)].mean() - fwd[~np.roll(flag, s)].mean() for s in shifts])
    pval = (np.abs(null) >= abs(diff)).mean()

    print(f"divergence days: {int(flag.sum())} ({episodes} episodes)")
    print(f"next {horizon}-day mean return: after divergence {fwd[flag].mean():+.2%} / "
          f"other days {fwd[~flag].mean():+.2%}  (difference {diff:+.2%})")
    print(f"circular-shift permutation p-value: {pval:.3f} ({n_perm} shifts)")


# ============================================================ intraday timing of the effect
def intraday_decomposition(df: pd.DataFrame, hourly: pd.Series, open_hour: int = 9,
                           max_h: int = 24) -> pd.DataFrame:
    """
    Split the next-day return into
      overnight: day-t 16:00 -> next trading day open_hour:00 ET (weekends included on Fridays)
      daytime:   next trading day open_hour:00 -> 16:00 ET
    and regress each on residual and raw flow. Also returns the coefficient of the
    cumulative return from day-t 16:00 to h clock hours later, h = 1..max_h.
    """
    resid = flow_residual(df)
    days = df.index
    nxt = pd.Series(days[1:].append(pd.DatetimeIndex([pd.NaT])), index=days)
    d = pd.Series(days, index=days)

    def px(ts_naive: pd.Series, hour: int) -> pd.Series:
        ts = (ts_naive + pd.Timedelta(hours=hour)).dt.tz_localize(ET, nonexistent="NaT", ambiguous="NaT")
        return pd.Series(hourly.reindex(ts).to_numpy(), index=ts_naive.index)

    p0, p_open, p_close = px(d, 16), px(nxt, open_hour), px(nxt, 16)
    parts = {
        f"overnight (16:00 -> {open_hour}:00)": np.log(p_open / p0),
        f"daytime ({open_hour}:00 -> 16:00)  ": np.log(p_close / p_open),
        "total (to next 16:00)       ": np.log(p_close / p0),
    }
    print(f"{'segment':30s}{'residual':>10s}{'t':>7s}{'raw':>10s}{'t':>7s}   (% per $100m)")
    betas = {}
    for name, r in parts.items():
        out = []
        for x in (resid, df["flow"]):
            res = _ols_hac(r, (x / 100).to_frame("x"))
            out += [res.params.iloc[1] * 100, res.tvalues.iloc[1]]
        betas[name] = out[0]
        print(f"{name:30s}{out[0]:>10.3f}{out[1]:>7.2f}{out[2]:>10.3f}{out[3]:>7.2f}")
    names = list(parts)
    print(f"share of the residual-flow effect realised overnight: {betas[names[0]] / betas[names[2]]:.0%}")

    t0 = (d + pd.Timedelta(hours=16)).dt.tz_localize(ET)
    base = hourly.reindex(t0).to_numpy()
    prof = []
    for h in range(1, max_h + 1):
        r = pd.Series(np.log(hourly.reindex(t0 + pd.Timedelta(hours=h)).to_numpy() / base), index=days)
        res = _ols_hac(r, (resid / 100).to_frame("x"))
        prof.append({"h": h, "beta": res.params.iloc[1] * 100, "se": res.bse.iloc[1] * 100,
                     "t": res.tvalues.iloc[1]})
    return pd.DataFrame(prof).set_index("h")


# ============================================================ figures
def plot_overview(df: pd.DataFrame, ll: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={"height_ratios": [2, 1.3, 1]})

    a = ax[0]
    a.bar(df.index, df["flow"], color=np.where(df["flow"] >= 0, "#5aa469", "#d9534f"), width=1.0)
    a.set_ylabel("Daily net flow (US$m)")
    a2 = a.twinx()
    a2.plot(df.index, df["price"], color="#333", lw=1.2)
    a2.set_ylabel("BTC price at 16:00 ET (US$)")
    a.set_title("BTC price and US spot ETF daily net flows")

    b = ax[1]
    b.scatter(df["flow_5d"], df["ret_5d"] * 100, s=8, alpha=0.5)
    b.axhline(0, color="grey", lw=0.8)
    b.axvline(0, color="grey", lw=0.8)
    b.set_xlabel("5-day cumulative ETF flow (US$m)   <- outflow | inflow ->")
    b.set_ylabel("5-day BTC return (%)")
    b.set_title("Do ETF flows and price move in the same direction?")
    kw = dict(transform=b.transAxes, fontsize=9, color="#555")
    b.text(0.02, 0.97, "outflow, price up", va="top", ha="left", **kw)
    b.text(0.98, 0.97, "inflow, price up", va="top", ha="right", **kw)
    b.text(0.02, 0.03, "outflow, price down", va="bottom", ha="left", **kw)
    b.text(0.98, 0.03, "inflow, price down", va="bottom", ha="right", **kw)

    c = ax[2]
    c.bar(ll.index, ll["spearman"], color="#4a7ab5")
    c.axhline(0, color="grey", lw=0.8)
    c.set_xticks(ll.index)
    c.set_xticklabels(["3 days\nbefore", "2 days\nbefore", "day\nbefore", "same\nday",
                       "next\nday", "2 days\nafter", "3 days\nafter"])
    c.set_xlabel("BTC return day, relative to the day of the ETF flow")
    c.set_ylabel("Spearman correlation")
    c.set_title("Do ETF flows chase price, or lead it?")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def plot_profile(prof: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(prof.index, prof["beta"], color="#4a7ab5", marker="o", ms=3)
    ax.fill_between(prof.index, prof["beta"] - 2 * prof["se"], prof["beta"] + 2 * prof["se"],
                    color="#4a7ab5", alpha=0.15, label="±2 standard errors")
    ax.axhline(0, color="grey", lw=0.8)
    ticks = [1, 4, 8, 12, 17, 20, 24]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{(16 + h) % 24}:00" for h in ticks])
    ax.set_xlabel("Clock time (ET) after the 16:00 ET flow cut-off")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Cumulative return per US$100m of residual flow (24 hours from 16:00 ET)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


# ============================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flows", help="flow CSV (Date, ..., Total in US$m); default: scrape Farside")
    ap.add_argument("--start", default="2024-01-11")
    ap.add_argument("--end", default=None)
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    hourly = None
    if args.demo:
        flows, price = make_demo()
    else:
        flows = load_flows(args.flows) if args.flows else load_farside_flows(args.start, args.end)
        hourly = load_hourly(args.start, args.end)
        price = load_price_at_close(hourly)

    df = build_panel(flows, price)
    if len(df) < 130:
        raise ValueError(f"only {len(df)} usable days; at least ~130 are needed (120 for the expanding window)")
    print(f"period: {df.index.min().date()} to {df.index.max().date()}  ({len(df)} trading days)\n")

    ll = lead_lag(df)
    print("== Lead-lag correlation (flow on day t vs. return on day t+k)\n", ll.round(3), "\n")
    print("== Regressions (HAC s.e., flow in US$100m)")
    regressions(df)
    print("\n== R2 decomposition of the flow model (in-sample)")
    r2_decomposition(df)
    print("\n== Raw vs. residual flow -> next-day return")
    residual_test(df)
    print("\n== Divergence days (5-day outflow, 5-day price up)")
    divergence_test(df)
    plot_overview(df, ll, os.path.join(args.outdir, "btc_etf_flow.png"))

    if hourly is not None:
        print("\n== When does the residual-flow effect show up?")
        prof = intraday_decomposition(df, hourly)
        print("\ncumulative-return coefficient h hours after 16:00 ET (% per $100m)")
        print(prof.loc[[1, 4, 8, 12, 17, 20, 24]].round(3))
        plot_profile(prof, os.path.join(args.outdir, "btc_etf_flow_intraday.png"))


if __name__ == "__main__":
    main()
