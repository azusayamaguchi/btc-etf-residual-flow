"""
Stablecoin supply changes vs. next-day BTC returns, compared with ETF residual flow.

Pre-specified hypotheses (decide BEFORE looking at the results):
  H1 (primary)   : the residual stablecoin flow (USDT + USDC supply change that past/same-day
                   BTC returns and past supply changes cannot explain) is positively related
                   to the next-day BTC return.                         -> tested at 5%
  H2 (secondary) : the relation survives controlling for the BTC ETF residual flow, and
  H3 (secondary) : the ETF residual flow survives controlling for the stablecoin residual.
                                                       -> Bonferroni, 0.05 / 2 = 0.025

Usage:
    python src/stablecoin_flow.py --etf-flows data/farside_btc_flows.csv --start 2025-01-01
    python src/stablecoin_flow.py --demo          # offline, synthetic data

Data: DefiLlama stablecoins API (free, no key): https://stablecoins.llama.fi
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from btc_etf_flow_analysis import (ET, _ols_hac, build_panel, flow_residual, lead_lag,
                                   load_flows, load_hourly, load_price_at_close,
                                   r2_decomposition, residual_test)

BASE = "https://stablecoins.llama.fi"


# ============================================================ data
def resolve_ids(symbols: list[str]) -> dict[str, str]:
    """Map symbols (USDT, USDC, ...) to DefiLlama stablecoin ids."""
    import requests
    r = requests.get(f"{BASE}/stablecoins", params={"includePrices": "false"}, timeout=60)
    r.raise_for_status()
    assets = r.json()["peggedAssets"]
    out = {}
    for sym in symbols:
        # several assets can share a symbol; take the one with the largest supply
        cands = [a for a in assets if a.get("symbol", "").upper() == sym.upper()]
        if not cands:
            raise ValueError(f"stablecoin {sym} not found on DefiLlama")
        best = max(cands, key=lambda a: (a.get("circulating") or {}).get("peggedUSD", 0) or 0)
        out[sym] = str(best["id"])
    return out


def fetch_supply(symbols: list[str]) -> pd.DataFrame:
    """Daily circulating supply (US$m) per stablecoin, indexed by the UTC snapshot date."""
    import requests
    cols = {}
    for sym, sid in resolve_ids(symbols).items():
        r = requests.get(f"{BASE}/stablecoincharts/all", params={"stablecoin": sid}, timeout=60)
        r.raise_for_status()
        rows = []
        for p in r.json():
            circ = p.get("totalCirculatingUSD") or p.get("totalCirculating") or {}
            v = circ.get("peggedUSD")
            if v is not None:
                rows.append((pd.to_datetime(int(p["date"]), unit="s"), float(v) / 1e6))
        s = pd.Series(dict(rows)).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        cols[sym] = s
        print(f"{sym} (id {sid}): {len(s)} daily snapshots, {s.index.min().date()} to {s.index.max().date()}")
    return pd.DataFrame(cols)


def supply_to_trading_day_flow(supply: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.Series:
    """
    Align daily UTC snapshots to the 16:00 ET trading-day panel WITHOUT look-ahead.

    The snapshot dated D (00:00 UTC) is taken at 19:00/20:00 ET on day D-1, i.e. before
    16:00 ET on day D. The flow assigned to trading day t is therefore
        S(snapshot dated t) - S(snapshot dated previous trading day),
    which is fully known before the day-t 16:00 ET close and does not overlap the
    next-day return window (16:00 t -> 16:00 t+1). Weekend changes accrue to Monday.
    """
    total = supply.sum(axis=1, min_count=supply.shape[1])     # require every coin present
    s = total.reindex(trading_days)
    return s.diff().rename("Total")


# ============================================================ horse race
def horse_race(btc: pd.DataFrame, eps_stable: pd.Series, eps_etf: pd.Series) -> None:
    """Next-day BTC return on stablecoin residual, ETF residual, and both."""
    z = pd.concat({"ret_next": btc["ret"].shift(-1),
                   "stable": eps_stable / 100, "etf": eps_etf / 100},
                  axis=1, join="inner").dropna()
    print(f"common days: {len(z)}   corr(eps_stable, eps_etf) = {z['stable'].corr(z['etf']):.3f}")
    for cols in (["stable"], ["etf"], ["stable", "etf"]):
        res = _ols_hac(z["ret_next"], z[cols])
        terms = ", ".join(f"{c}: {res.params[c] * 100:+.3f}%/$100m (t={res.tvalues[c]:.2f}, "
                          f"p={res.pvalues[c]:.3f})" for c in cols)
        print(f"  next-day BTC ~ {' + '.join(cols):13s} {terms}")


def make_demo(n: int = 450, seed: int = 11):
    """Synthetic: ETF residual carries information, stablecoin flow only chases price."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n)
    ret, etf, stab = np.zeros(n), np.zeros(n), np.zeros(n)
    shock = rng.normal(0, 200, n)
    for t in range(1, n):
        ret[t] = 0.00002 * shock[t - 1] + rng.normal(0, 0.025)
        etf[t] = 3000 * ret[t - 1] + shock[t]
        stab[t] = 4000 * ret[t - 1] + 0.4 * stab[t - 1] + rng.normal(0, 150)
    price = pd.Series(80000 * np.exp(np.cumsum(ret)), index=days, name="price")
    return (pd.DataFrame({"Total": etf}, index=days), pd.DataFrame({"Total": stab}, index=days), price)


# ============================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etf-flows", help="Farside BTC flow CSV")
    ap.add_argument("--coins", default="USDT,USDC")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--save-supply", default="data/stablecoin_supply.csv")
    args = ap.parse_args()

    if args.demo:
        etf_flows, stable_flows, price = make_demo()
    else:
        if not args.etf_flows:
            raise SystemExit("--etf-flows is required (run src/farside_to_csv.py --asset BTC)")
        price = load_price_at_close(load_hourly(args.start, args.end, product="BTC-USD"))
        etf_flows = load_flows(args.etf_flows)
        supply = fetch_supply([c.strip() for c in args.coins.split(",")])
        os.makedirs(os.path.dirname(args.save_supply) or ".", exist_ok=True)
        supply.to_csv(args.save_supply)
        print(f"saved {args.save_supply}")
        # BTC trades every day, so price.index contains weekends too. Use US trading days
        # (days with an ETF flow record) so both panels share the same day definition.
        trading_days = price.index.intersection(etf_flows.index)
        stable_flows = supply_to_trading_day_flow(supply, trading_days).to_frame()

    btc = build_panel(etf_flows, price)                 # ETF flow panel (BTC returns)
    stab = build_panel(stable_flows, price)             # stablecoin flow panel (same returns)
    print(f"\nperiod: {stab.index.min().date()} to {stab.index.max().date()}  ({len(stab)} trading days)")

    print("\n== Stablecoin supply change: lead-lag with BTC returns\n", lead_lag(stab).round(3))
    print("\n== R2 decomposition of the stablecoin flow model (in-sample)")
    r2_decomposition(stab)
    print("\n== H1: raw vs. residual stablecoin flow -> next-day BTC return")
    residual_test(stab)
    print("\n== H2/H3: stablecoin residual vs. ETF residual (horse race)")
    horse_race(btc, flow_residual(stab), flow_residual(btc))


if __name__ == "__main__":
    main()
