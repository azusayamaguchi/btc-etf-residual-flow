# Raw flows don't predict, residuals do: 686 days of US spot Bitcoin ETF flows

Do daily flows into US spot Bitcoin ETFs tell us anything about where BTC goes next? Using 686 trading days of data (January 2024 – October 2026), this repository finds two things:

1. **About half of the variation in daily flows (46.5%) can be explained by same-day and recent BTC returns and by past flows.** Raw flows have almost no power to predict the next day's return.
2. **The part that cannot be explained — the *residual flow* — does predict the next day's return**: +0.17% per US$100m of residual flow (t = 3.43), roughly half of it overnight and half during the next US session.

> This is a research note, not investment advice. See [Limitations](#limitations).

---

## Data and method

| Item | Source | Details |
| --- | --- | --- |
| ETF flows | [Farside Investors](https://farside.co.uk/bitcoin-etf-flow-all-data/) | Total daily net flow of all US spot Bitcoin ETFs, US$m |
| BTC price | Coinbase Exchange API | BTC-USD, 1-hour candles |
| Sample | | 2024-01-19 to 2026-10-13, 686 trading days |

BTC trades 24/7, but ETF flows are aggregated per US trading day and every ETF strikes its NAV at **16:00 ET**. Prices are therefore aligned to that cut-off: the daily "close" is the BTC price at 16:00 ET, and the daily return $r_t$ runs from 16:00 ET on the previous trading day to 16:00 ET on day $t$. This keeps the price window and the flow-reporting window as close as possible.

All regressions use HAC (Newey–West, 5 lags) standard errors. Flow coefficients are reported per US$100m.

---

## 1. Flows chase price

![BTC price and ETF flows, 5-day flows vs. 5-day returns, lead-lag correlation](figures/btc_etf_flow.png)

Daily flows are most strongly correlated with the **previous day's and the same day's** BTC return, and hardly at all with later returns.

| BTC return day relative to the flow | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spearman correlation | 0.119 | 0.151 | 0.417 | 0.427 | 0.073 | −0.018 | 0.005 |

- A 1% BTC rise on day $t-1$ is followed by about **+US$45m** of flow on day $t$ (t = 8.24).
- Raw flow explains only **0.9%** of the variance of the next day's return ($R^2 = 0.009$).
- The same-day correlation (0.43) cannot be given a causal direction: flows may move the price, or investors may buy after seeing the price rise.

Headlines of the form *"ETF inflows are a tailwind for BTC"* treat flows as a leading indicator. On average, the data say they are mostly a lagging one.

---

## 2. Residual flow: the part price moves cannot explain

### Model

Let $F_t$ be the daily net flow and $r_t$ the daily BTC return. The expected flow is a linear model of the same-day and three past returns and three past flows:

$$
\hat{F}_t = a + \sum_{k=0}^{3} b_k\, r_{t-k} + \sum_{k=1}^{3} c_k\, F_{t-k}
$$

and the residual flow is

$$
\varepsilon_t = F_t - \hat{F}_t .
$$

### Estimation

The coefficients $a, b_k, c_k$ used for day $t$ are estimated on data up to day $t-1$ only (expanding window) and then applied to day $t$. $\hat{F}_t$ is therefore an **out-of-sample prediction** and $\varepsilon_t$ an **out-of-sample residual**. The first 120 days are used only for estimation, leaving 562 days for evaluation.

Note that the same-day return $r_t$ is a regressor, so $\hat{F}_t$ is conditional on the 16:00 ET price of day $t$; it is not a forecast made at the start of the day. $r_t$ is included deliberately: without it, the residual would still contain the reaction to today's price move, and any link with tomorrow's return could simply be the continuation of today's move rather than information specific to flows.

### How much of the flow is explained?

The out-of-sample $R^2 = 1 - \mathrm{Var}(\varepsilon)/\mathrm{Var}(F)$ is **46.5%**. Adding the regressor groups one at a time (in-sample, full period):

| Regressors added | Increment in $R^2$ | Interpretation |
| --- | --- | --- |
| Returns, $t-1$ to $t-3$ | 20.6% | Clear trend-chasing |
| Same-day return, $t$ | 18.7% | Direction of causality unclear |
| Flows, $t-1$ to $t-3$ | 9.9% | Persistence of flows |
| **Total** | **49.1%** | |

Increments depend on the order in which groups are added. Unambiguous trend-chasing accounts for about a fifth of the variation; about half is explainable in total.

### Result

| Predictor of next-day return | per US$100m | t | Spearman |
| --- | --- | --- | --- |
| Raw flow | +0.068% | 2.10 | 0.051 |
| Residual flow | +0.173% | 3.43 | 0.149 |

Removing the explainable part raises the coefficient about 2.5× and the rank correlation about 3×. The explainable half of raw flows dilutes whatever information flows carry about the next day.

### Robustness

Early 2024 was dominated by large GBTC redemptions, so the test was repeated from January 2025: **+0.176% per US$100m (t = 3.05, 317 days)**, essentially unchanged. This subsample is part of the full sample, so it is not an independent replication.

---

## 3. When does the effect show up?

The next-day return is split at 09:00 ET into an overnight and a daytime segment:

| Segment | Residual flow | t | Raw flow | t |
| --- | --- | --- | --- | --- |
| Overnight (16:00 → next 09:00 ET) | +0.090% | 2.30 | +0.059% | 2.26 |
| Daytime (09:00 → 16:00 ET) | +0.083% | 2.96 | +0.025% | 1.23 |
| Total | +0.173% | 3.43 | +0.084% | 2.68 |

*Per US$100m.*

The question is whether the effect is only a **reporting-timing artefact**: when an ETF receives inflows, authorised participants buy spot BTC, and part of that buying can happen after the 16:00 ET cut-off. If that were the whole story, the effect would be concentrated overnight.

It is not: **52%** of the residual-flow effect appears overnight and the rest during the next US session. The daytime effect is insignificant for raw flow (t = 1.23) but significant for residual flow (t = 2.96).

![Cumulative return per US$100m of residual flow, 24 hours from 16:00 ET](figures/btc_etf_flow_intraday.png)

Hour by hour, the effect builds up slowly overnight and rises again from early afternoon into 16:00 ET the next day. The afternoon rise may reflect ETF-related trading into the next NAV strike, but that remains a hypothesis. The x-axis is clock time, so for Friday flows the profile runs to Saturday 16:00; the 24-hour coefficient (+0.130%, t = 3.21) therefore differs from the next-trading-day total in the table (+0.173%).

---

## 4. "Outflow but price up" divergences

A *divergence day* has a negative 5-day cumulative flow but a positive 5-day return. On average, returns over the following 5 days were weaker after divergences, but the difference is well within what chance produces.

| Mean return over the next 5 days | Full sample | Since 2025 |
| --- | --- | --- |
| After divergence days | −1.14% | −1.10% |
| Other days | +0.68% | −0.08% |
| Difference | −1.82% | −1.01% |
| p-value | 0.23 | 0.44 |

There were 54 divergence days, but they cluster into only 34 episodes, and the forward 5-day windows of neighbouring days overlap. The effective sample is much smaller than 54, and a plain t-test would overstate precision. Instead, the calendar pattern of divergence days was shifted circularly at random 5,000 times, keeping its clustering. In 23% of the shifted, "fake" patterns the difference was at least as large as observed (in either direction); since 2025, 44%. Divergences do not provide a reliable signal that a rebound will fail.

---

## Conclusion

Raw spot-ETF flows barely predict the next day's BTC return, because about half of their variation is explained by recent price moves and past flows. The residual — flow beyond what price moves explain — is associated with +0.17% per US$100m the next day, realised roughly half overnight and half in the next US session. Daily flows are better read as *"how much above (or below) what recent price action would explain"* than as a headline dollar amount.

## Limitations

- **Small explanatory power.** Most of BTC's daily movement is unrelated to ETF flows.
- **Not necessarily tradable.** Depending on when Farside data become complete, part of the next-day effect may occur before the data are public.
- **Model choice.** The expected-flow model (three lags, linear) is one choice among many; other lag lengths have not yet been tested.
- **Order dependence.** The $R^2$ decomposition depends on the order in which regressor groups are added.
- **Divergences.** No statistically reliable conclusion about what follows "outflow but price up" episodes.

Planned next steps: robustness to lag length and sample period, and differences across individual ETFs (IBIT, FBTC, …).

---

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download the full daily flow table (or save the page from a browser and use --html)
python src/farside_to_csv.py --out data/farside_btc_flows.csv

# 2. Run the analysis (fetches Coinbase hourly candles; writes figures/)
python src/btc_etf_flow_analysis.py --flows data/farside_btc_flows.csv --start 2024-01-11

# Offline demo with synthetic data
python src/btc_etf_flow_analysis.py --demo
```

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── farside_to_csv.py          # Farside "All Data" table -> daily CSV
│   └── btc_etf_flow_analysis.py   # all statistics and figures in this note
├── data/                          # flow CSV goes here (not committed)
└── figures/                       # generated figures
```

### Core function: residual flow

```python
def flow_residual(df, p=3, min_obs=120):
    """Out-of-sample residual flow eps_t = F_t - Fhat_t (coefficients fit on days < t)."""
    X = pd.concat({f"ret_l{k}": df["ret"].shift(k) for k in range(0, p + 1)} |
                  {f"flow_l{k}": df["flow"].shift(k) for k in range(1, p + 1)}, axis=1)
    X.insert(0, "const", 1.0)
    y = df["flow"]
    ok = X.notna().all(axis=1) & y.notna()
    Xv, yv, idx = X[ok].to_numpy(), y[ok].to_numpy(), X[ok].index
    resid = pd.Series(np.nan, index=df.index)
    for i in range(min_obs, len(yv)):
        beta, *_ = np.linalg.lstsq(Xv[:i], yv[:i], rcond=None)   # fit on days < t
        resid.loc[idx[i]] = yv[i] - Xv[i] @ beta
    return resid
```

`df` holds the daily flow (`flow`, US$m) and the 16:00 ET log return (`ret`).

---

## Data sources and disclaimer

- ETF flows: [Farside Investors](https://farside.co.uk/bitcoin-etf-flow-all-data/). Farside notes that its data contain estimates and may contain errors. The raw data are not redistributed in this repository.
- BTC prices: Coinbase Exchange public API (BTC-USD, 1-hour candles).

This repository is for research and educational purposes only and does not constitute investment advice.
