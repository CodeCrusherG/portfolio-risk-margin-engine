"""Portfolio Risk and Margin Engine.

VaR / ES (historical, parametric, Monte Carlo), Kupiec + Basel traffic-light
back-testing, VaR-based initial margin, and automated daily commentary.
"""
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------- data layer
def synthetic_prices(n_days=1500, seed=42):
    """Correlated GBM prices with a volatility shock; used when no internet."""
    rng = np.random.default_rng(seed)
    names = ["EQ_A", "EQ_B", "INDEX", "FX_USDINR", "BOND_ETF"]
    mu = np.array([0.12, 0.10, 0.09, 0.02, 0.05]) / TRADING_DAYS
    vol = np.array([0.28, 0.25, 0.18, 0.06, 0.07]) / np.sqrt(TRADING_DAYS)
    corr = np.array([
        [1.0, 0.6, 0.8, 0.1, -0.1],
        [0.6, 1.0, 0.7, 0.1, -0.1],
        [0.8, 0.7, 1.0, 0.0, -0.2],
        [0.1, 0.1, 0.0, 1.0, 0.1],
        [-0.1, -0.1, -0.2, 0.1, 1.0],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, len(names))) @ L.T
    scale = np.ones(n_days)
    scale[900:1000] = 2.5  # stress regime so back-test shows exceptions
    rets = mu + z * vol * scale[:, None]
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=names)

def load_prices(tickers=None, start="2015-01-01"):
    """Real data via yfinance if tickers given, else synthetic."""
    if not tickers:
        return synthetic_prices()
    import yfinance as yf
    cols = {}
    for t in tickers:
        for attempt in range(3):
            d = yf.download(t, start=start, auto_adjust=True,
                            progress=False, threads=False)
            if not d.empty:
                s = d["Close"]
                cols[t] = s.squeeze() if hasattr(s, "squeeze") else s
                break
        else:
            raise RuntimeError(f"Could not download {t} after 3 attempts")
    df = pd.DataFrame(cols).ffill()
    cal = next((t for t in ("^NSEI", "NIFTYBEES.NS") if t in cols), tickers[0])
    print(f"Using {cal} as the trading calendar")
    df = df.loc[cols[cal].dropna().index].dropna()
    print(f"Loaded {df.shape[0]} rows, {df.index[0].date()} to "
          f"{df.index[-1].date()}, columns: {list(df.columns)}")
    return df


def save_to_sql(prices, db_path="market_data.db"):
    with sqlite3.connect(db_path) as con:
        prices.reset_index(names="date").to_sql(
            "prices", con, if_exists="replace", index=False)


def query_prices(db_path="market_data.db"):
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("SELECT * FROM prices ORDER BY date", con,
                         parse_dates=["date"])
    return df.set_index("date")


def portfolio_returns(prices, weights):
    rets = np.log(prices / prices.shift(1)).dropna()
    w = pd.Series(weights).reindex(rets.columns).values
    return pd.Series(rets.values @ w, index=rets.index, name="port_ret")


# --------------------------------------------------------------- risk models
def hist_var(r, alpha=0.99):
    return -np.quantile(r, 1 - alpha)


def hist_es(r, alpha=0.99):
    cutoff = np.quantile(r, 1 - alpha)
    return -r[r <= cutoff].mean()


def param_var(r, alpha=0.99):
    return -(r.mean() + stats.norm.ppf(1 - alpha) * r.std(ddof=1))


def param_es(r, alpha=0.99):
    z = stats.norm.ppf(1 - alpha)
    return -(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / (1 - alpha))


def mc_var_es(rets_df, weights, alpha=0.99, n_sims=100_000, seed=0):
    """Monte Carlo VaR/ES from a multivariate normal fit to asset returns."""
    rng = np.random.default_rng(seed)
    w = pd.Series(weights).reindex(rets_df.columns).values
    sims = rng.multivariate_normal(rets_df.mean().values,
                                   rets_df.cov().values, n_sims) @ w
    return hist_var(sims, alpha), hist_es(sims, alpha)


def stress_test(prices, weights, shocks):
    """Apply instantaneous % shocks per asset; return portfolio P&L (fraction)."""
    unknown = set(shocks) - set(prices.columns)
    if unknown:
        raise ValueError(f"Shock keys not in price data: {unknown}")
    w = pd.Series(weights).reindex(prices.columns)
    s = pd.Series(shocks).reindex(prices.columns).fillna(0.0)
    return float((w * s).sum())


# ------------------------------------------------------------- back-testing
def rolling_hist_var(r, window=250, alpha=0.99):
    """One-day-ahead VaR using only data available before each day."""
    var = r.rolling(window).apply(lambda x: hist_var(x.values, alpha),
                                  raw=False).shift(1)
    return var.dropna()


def kupiec_pof(n_exc, n_obs, alpha=0.99):
    """Kupiec proportion-of-failures likelihood-ratio test."""
    p = 1 - alpha
    x, n = n_exc, n_obs
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        ph = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def traffic_light(exceptions_250d):
    if exceptions_250d <= 4:
        return "GREEN"
    return "YELLOW" if exceptions_250d <= 9 else "RED"


def backtest(r, window=250, alpha=0.99):
    var = rolling_hist_var(r, window, alpha)
    aligned = r.loc[var.index]
    exc = aligned < -var
    last250 = int(exc.tail(250).sum())
    lr, pval = kupiec_pof(int(exc.sum()), len(exc), alpha)
    return {
        "var_series": var,
        "exceptions": exc,
        "n_obs": len(exc),
        "n_exceptions": int(exc.sum()),
        "expected": round(len(exc) * (1 - alpha), 1),
        "last250_exceptions": last250,
        "zone": traffic_light(last250),
        "kupiec_lr": lr,
        "kupiec_p": pval,
        "kupiec_reject": pval < 0.05,
    }


# ---------------------------------------------------- margin and commentary
def initial_margin(r, notional, alpha=0.99, horizon=10, window=500):
    """VaR-based IM: 99% hist VaR, sqrt-time scaled to a 10-day MPOR."""
    return hist_var(r.tail(window).values, alpha) * np.sqrt(horizon) * notional


def margin_history(r, notional, start_offset=30, **kw):
    idx = r.index[-start_offset:]
    return pd.Series([initial_margin(r.loc[:d], notional, **kw) for d in idx],
                     index=idx, name="initial_margin")


def commentary(r, im_hist, notional):
    """Auto-written day-on-day move note (the CEM daily commentary)."""
    d0, d1 = im_hist.index[-2], im_hist.index[-1]
    im0, im1 = im_hist.iloc[-2], im_hist.iloc[-1]
    chg = im1 - im0
    pct = chg / im0 * 100
    pnl = r.iloc[-1] * notional
    if abs(pct) < 0.005:
        direction = "was unchanged"
    else:
        direction = "increased" if chg > 0 else "decreased"
    vol20 = r.tail(20).std() * np.sqrt(TRADING_DAYS) * 100
    vol_long = r.tail(250).std() * np.sqrt(TRADING_DAYS) * 100
    regime = "above" if vol20 > vol_long else "below"
    if direction == "was unchanged":
        driver = ("the 99% tail quantile staying fixed, as the new "
                  "observation fell inside the tail threshold")
    elif r.iloc[-1] < r.tail(500).quantile(0.02):
        driver = "a large adverse return entering the lookback window"
    else:
        driver = ("the rolling window shifting, with older stress days "
                  "dropping out or being replaced")
    move = ("" if direction == "was unchanged"
            else f" by {abs(chg):,.0f} ({pct:+.2f}%)")
    parts = [
        f"{d1.date()}: Initial margin {direction}{move} vs {d0.date()} "
        f"at {im1:,.0f}.",
        f"Day P&L was {pnl:+,.0f} ({r.iloc[-1] * 100:+.2f}%).",
        f"20-day realised vol is {vol20:.1f}%, {regime} the 1-year "
        f"average of {vol_long:.1f}%.",
        f"Movement driven by {driver}.",
    ]
    return " ".join(parts)