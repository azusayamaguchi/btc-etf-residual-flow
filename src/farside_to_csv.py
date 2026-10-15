"""
Download the full Farside "Bitcoin ETF Flow - All Data" table into a daily CSV.

Usage:
    python src/farside_to_csv.py --out data/farside_btc_flows.csv

    # If the request is blocked, save the page from your browser ("HTML only") and parse it:
    python src/farside_to_csv.py --html saved_page.html --out data/farside_btc_flows.csv

Output columns: Date, <ETF tickers...>, Total   (US$ million; outflows are negative)
Requirements: pandas requests beautifulsoup4
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
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


def fetch_html(path: str | None) -> str:
    if path:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    import requests
    r = requests.get(URL, timeout=60, headers={
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
        header = next((r for r in rows if r and r[0].strip().lower() == "date"
                       and r[-1].strip().lower() == "total"), None)
        data = [r for r in rows if r and DATE_RE.fullmatch(r[0].strip())]
        if header and data and (best is None or len(data) > len(best[1])):
            best = (header, data)
    if best is None:
        raise ValueError("No Date...Total table found; the page layout may have changed.")

    header, data = best
    cols = [h.strip() for h in header]
    records = [[r[0].strip()] + [to_num(v) for v in r[1:]] for r in data if len(r) == len(cols)]
    df = pd.DataFrame(records, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df = df.drop_duplicates("Date", keep="last").set_index("Date").sort_index()

    # Holidays / unreported days: every ETF is '-' while Total shows 0.0 -> drop them.
    etf_cols = [c for c in df.columns if c != "Total"]
    unreported = df[etf_cols].isna().all(axis=1)
    df = df[~unreported]

    gap = (df[etf_cols].sum(axis=1) - df["Total"]).abs()
    bad = gap[gap > 0.15]
    print(f"parsed {len(df)} trading days ({df.index.min().date()} to {df.index.max().date()}), "
          f"{len(etf_cols)} ETFs, dropped {int(unreported.sum())} unreported days")
    if len(bad):
        print(f"warning: ETF columns do not sum to Total on {len(bad)} days (e.g. {bad.index[0].date()})")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="saved HTML page (default: download)")
    ap.add_argument("--out", default="data/farside_btc_flows.csv")
    args = ap.parse_args()
    df = parse(fetch_html(args.html))
    df.to_csv(args.out, date_format="%Y-%m-%d", float_format="%.1f")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
