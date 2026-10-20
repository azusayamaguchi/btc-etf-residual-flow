"""
Download a full Farside "ETF Flow - All Data" table (BTC or ETH) into a daily CSV.

Usage:
    python src/farside_to_csv.py --asset BTC --out data/farside_btc_flows.csv
    python src/farside_to_csv.py --asset ETH --out data/farside_eth_flows.csv

    # If the request is blocked, save the page from your browser ("HTML only") and parse it:
    python src/farside_to_csv.py --asset ETH --html saved_page.html --out data/farside_eth_flows.csv

Output columns: Date, <ETF tickers...>, Total   (US$ million; outflows are negative)
Requirements: pandas requests beautifulsoup4
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

URLS = {
    "BTC": "https://farside.co.uk/bitcoin-etf-flow-all-data/",
    "ETH": "https://farside.co.uk/ethereum-etf-flow-all-data/",
}
DATE_RE = re.compile(r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$")
MISSING = {"-", "—", "–", ""}


def to_num(s: str) -> float:
    """'(484.9)' -> -484.9, '1,234.5' -> 1234.5, '-' -> NaN (not reported)."""
    s = s.replace(",", "").replace("\xa0", "").strip()
    if s in MISSING:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    v = float(s.strip("()"))
    return -v if neg else v


def fetch_html(path: str | None, url: str) -> str:
    if path:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    import requests
    r = requests.get(url, timeout=60, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
    r.raise_for_status()
    return r.text


def parse(html: str) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    best = None   # the Date ... Total table with the most dated rows
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                for tr in table.find_all("tr")]
        # Header = the ticker row ending in "Total". Its first cell is "Date" on some
        # pages and blank on others (e.g. the ETH table), so do not rely on it.
        header = next((r for r in rows if len(r) > 2 and r[-1].strip().lower() == "total"
                       and r[0].strip().lower() in ("", "date")), None)
        data = [r for r in rows if r and DATE_RE.fullmatch(r[0].strip())]
        if header and data and (best is None or len(data) > len(best[1])):
            best = (header, data)
    if best is None:
        raise ValueError("No Date...Total table found; the page layout may have changed.")

    header, data = best
    cols = ["Date"] + [h.strip() for h in header[1:]]   # first header cell may be blank
    records = [[r[0].strip()] + [to_num(v) for v in r[1:]] for r in data if len(r) == len(cols)]
    df = pd.DataFrame(records, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df = df.drop_duplicates("Date", keep="last").set_index("Date").sort_index()

    # Holidays / unreported days: every ETF is '-' while Total shows 0.0 -> drop them.
    etf_cols = [c for c in df.columns if c != "Total"]
    unreported = df[etf_cols].isna().all(axis=1)
    df = df[~unreported]

    # Partially reported days: an ETF that was already trading shows '-' (its number is not
    # in yet, so Total is incomplete). The last rows of the table are often like this.
    listed = df[etf_cols].notna().cummax()                 # True from an ETF's first reported day
    partial = (df[etf_cols].isna() & listed).any(axis=1)
    trailing = partial[::-1].cummin()[::-1]                 # partial days at the end of the table
    inner = partial & ~trailing
    if trailing.any():
        print(f"dropped {int(trailing.sum())} incomplete trailing day(s): "
              f"{', '.join(str(d.date()) for d in df.index[trailing])}")
    if inner.any():
        print(f"note: {int(inner.sum())} earlier day(s) have a missing ETF value "
              f"(e.g. {df.index[inner][0].date()}); Total is kept as published")
    df = df[~trailing]

    gap = (df[etf_cols].sum(axis=1) - df["Total"]).abs()
    bad = gap[gap > 0.15]
    print(f"parsed {len(df)} trading days ({df.index.min().date()} to {df.index.max().date()}), "
          f"{len(etf_cols)} ETFs, dropped {int(unreported.sum())} unreported days")
    if len(bad):
        print(f"warning: ETF columns do not sum to Total on {len(bad)} days (e.g. {bad.index[0].date()})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(URLS), default="BTC")
    ap.add_argument("--html", help="saved HTML page (default: download)")
    ap.add_argument("--out", default=None, help="default: data/farside_<asset>_flows.csv")
    args = ap.parse_args()
    args.out = args.out or f"data/farside_{args.asset.lower()}_flows.csv"
    df = parse(fetch_html(args.html, URLS[args.asset]))
    df.to_csv(args.out, date_format="%Y-%m-%d", float_format="%.1f")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
