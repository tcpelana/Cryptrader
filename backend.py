# backend.py
import os, json, time, tempfile
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# optional heavy libs (import lazily to avoid startup cost)
def import_ccxt():
    try:
        import ccxt
        return ccxt
    except Exception:
        return None

import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# -------------------------
# FETCHERS
# -------------------------
def fetch_ohlcv_ccxt(exchange_id, symbol, timeframe, since=None, limit=1000):
    ccxt = import_ccxt()
    if not ccxt:
        return None
    ex = getattr(ccxt, exchange_id.lower())() if hasattr(ccxt, exchange_id.lower()) else ccxt.binance()
    # some exchanges require symbol like 'BTC/USDT'
    s = symbol if "/" in symbol else symbol.replace("BINANCE:","").replace(":","/")
    all_rows = []
    since_ms = None
    if since:
        since_ms = int(pd.Timestamp(since).timestamp()*1000)
    while True:
        batch = ex.fetch_ohlcv(s, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows += batch
        if len(batch) < limit:
            break
        since_ms = batch[-1][0] + 1
        time.sleep(0.2)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","vol"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("date")
    return df[["open","high","low","close","vol"]]

def fetch_ohlcv_coingecko(symbol, vs_currency="usd", days=2000):
    # expects 'bitcoin' or uses market id; fallback method
    # Try translate symbol like BTC -> bitcoin via coin list
    symbol_id = symbol.lower()
    if ":" in symbol_id:
        symbol_id = symbol_id.split(":")[-1]
    if "/" in symbol_id:
        symbol_id = symbol_id.split("/")[0]
    # fetch market_chart
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        prices = data.get("prices", [])
        df = pd.DataFrame(prices, columns=["ts","close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date")
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["vol"] = 0
        return df[["open","high","low","close","vol"]]
    except Exception:
        return None

def fetch_ohlcv_auto(symbol, timeframe, since):
    # try CCXT (Binance) first, fallback to CoinGecko
    df = None
    try:
        df = fetch_ohlcv_ccxt("binance", symbol, timeframe, since=since)
    except Exception:
        df = None
    if df is None:
        df = fetch_ohlcv_coingecko(symbol)
    return df

# -------------------------
# NEWS / SENTIMENT (simple)
# -------------------------
def fetch_news_features(query, from_dt, to_dt, api_key=None):
    # returns a DataFrame indexed by date with columns ['news_count','avg_sent']
    if not api_key:
        # return zeros
        idx = pd.date_range(start=from_dt, end=to_dt, freq='D')
        return pd.DataFrame(index=idx, data={"news_count":0,"avg_sent":0})
    # NewsAPI.org simple fetch (paging limited)
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "from": from_dt, "to": to_dt, "language": "en", "pageSize": 100, "apiKey": api_key}
    r = requests.get(url, params=params, timeout=30)
    articles = r.json().get("articles", [])
    # simple sentiment via VADER (if available) else naive polarity
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        rows = []
        for a in articles:
            title = a.get("title","")
            desc = a.get("description","") or ""
            txt = (title + ". " + desc)[:1000]
            s = analyzer.polarity_scores(txt)["compound"]
            dt = pd.to_datetime(a.get("publishedAt")).tz_convert(None).date()
            rows.append((dt, s))
        if not rows:
            idx = pd.date_range(start=from_dt, end=to_dt, freq='D')
            return pd.DataFrame(index=idx, data={"news_count":0,"avg_sent":0})
        df = pd.DataFrame(rows, columns=["date","sent"])
        agg = df.groupby("date").agg(news_count=("sent","count"), avg_sent=("sent","mean"))
        agg.index = pd.to_datetime(agg.index)
        full_idx = pd.date_range(start=from_dt, end=to_dt, freq='D')
        agg = agg.reindex(full_idx, fill_value=0)
        return agg
    except Exception:
        # fallback: only count articles per day
        rows = []
        for a in articles:
            dt = pd.to_datetime(a.get("publishedAt")).date()
            rows.append(dt)
        if not rows:
            idx = pd.date_range(start=from_dt, end=to_dt, freq='D')
            return pd.DataFrame(index=idx, data={"news_count":0,"avg_sent":0})
        df = pd.Series(1, index=pd.to_datetime(rows))
        agg = df.groupby(df.index.date).sum()
        agg = pd.DataFrame({"news_count": agg})
        agg["avg_sent"] = 0
        full_idx = pd.date_range(start=from_dt, end=to_dt, freq='D')
        agg.index = pd.to_datetime(agg.index)
        agg = agg.reindex(full_idx, fill_value=0)
        return agg

# -------------------------
# STRATEGIES / BACKTEST
# -------------------------
def simulate_strategy(df, strategy="ma_cross", capital=1000, risk_pct=1.0, leverage=1, aggressive=False):
    """
    Simple backtester. Returns list of trades (dicts) and equity Series.
    Strategies: buy_hold, ma_cross, rsi_meanrev, ml_prob
    """
    df = df.copy().dropna()
    trades = []
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = capital
    position = 0.0
    entry_price = None

    if strategy == "buy_hold":
        # buy at first close, hold
        entry_price = df["close"].iloc[0]
        position = capital / entry_price * leverage
        equity = df["close"] / entry_price * capital * leverage
        return [], equity

    if strategy == "ma_cross":
        df["ma_fast"] = df["close"].rolling(10).mean()
        df["ma_slow"] = df["close"].rolling(50).mean()
        signal = (df["ma_fast"] > df["ma_slow"]).astype(int)
    elif strategy == "rsi_meanrev":
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = up / (down.replace(0, np.nan))
        rsi = 100 - 100 / (1 + rs)
        signal = (rsi < 30).astype(int) - (rsi > 70).astype(int)
    elif strategy == "ml_prob":
        # placeholder: use simple momentum prob with past returns
        df["ret1"] = df["close"].pct_change().fillna(0)
        signal = (df["ret1"] > 0).astype(int)
    else:
        signal = pd.Series(0, index=df.index)

    cash = capital
    pos_units = 0.0
    for i in range(1, len(df)):
        date = df.index[i]
        price = df["close"].iloc[i]
        sig = signal.iloc[i]
        prev_sig = signal.iloc[i-1]
        # enter long on rising signal (simple)
        if sig == 1 and prev_sig == 0:
            # buy with risk_pct of equity
            risk_amount = cash * (risk_pct/100.0)
            units = (risk_amount * leverage) / price
            pos_units += units
            cash -= units * price / leverage  # rough
            trades.append({"date": date, "side":"buy","price":price,"size":units,"pnl":None})
        # exit on falling signal
        if sig == 0 and prev_sig == 1 and pos_units>0:
            pnl = pos_units * (price - trades[-1]["price"])
            cash += pos_units * price
            trades[-1]["pnl"] = pnl
            pos_units = 0
        equity.iloc[i] = cash + pos_units * price
    equity = equity.fillna(method="ffill").fillna(capital)
    return trades, equity

# -------------------------
# METRICS & MC & BOOTSTRAP
# -------------------------
def metrics_from_equity(equity_series, periods_per_year=252):
    equity = equity_series.dropna()
    if len(equity) < 2:
        return {}
    returns = equity.pct_change().dropna()
    total_periods = len(equity)
    ann_return = (equity.iloc[-1]/equity.iloc[0])**(periods_per_year/total_periods) - 1
    ann_vol = returns.std() * (periods_per_year**0.5)
    sharpe = ann_return / ann_vol if ann_vol>0 else float("nan")
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(equity.iloc[-1])
    }

def monte_carlo_final_equity(close_series, mc_runs=1000):
    rets = close_series.pct_change().dropna().values
    if len(rets)==0:
        return []
    mc = []
    for _ in range(mc_runs):
        sample = np.random.choice(rets, size=len(rets), replace=True)
        price = close_series.iloc[0]
        for r in sample:
            price = price * (1 + r)
        mc.append(price)
    return mc

def bootstrap_pvalue(strategy_equity, baseline_equity, n=2000):
    s_rets = strategy_equity.pct_change().dropna().values
    b_rets = baseline_equity.pct_change().dropna().values
    if len(s_rets)==0 or len(b_rets)==0:
        return 1.0
    diff_obs = np.mean(s_rets) - np.mean(b_rets)
    pooled = np.concatenate([s_rets, b_rets])
    count = 0
    for _ in range(n):
        perm = np.random.choice(pooled, size=len(pooled), replace=True)
        a = perm[:len(s_rets)]
        b = perm[len(s_rets):len(s_rets)+len(b_rets)]
        if np.mean(a)-np.mean(b) >= diff_obs:
            count += 1
    return count / n

# -------------------------
# PLOT HELPERS
# -------------------------
def make_price_fig(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name=f"{symbol} - Precio", line=dict(color="#7db0ff")))
    fig.update_layout(title=f"{symbol} - Precio", xaxis_title="date", yaxis_title="USD", template="plotly_dark")
    return fig

def make_equity_fig(equity_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values, name="equity", line=dict(color="#7bffb0")))
    fig.update_layout(title="Equity", xaxis_title="date", yaxis_title="equity", template="plotly_dark")
    return fig

def make_mc_fig(mc_results):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc_results, nbinsx=50))
    fig.update_layout(title="Distribución MC (precio final simulado)", template="plotly_dark")
    return fig

# -------------------------
# RUN ANALYSIS (entry)
# -------------------------
def run_analysis(symbol, start, end, timeframe, capital, mc_sims, strategy, risk, leverage, aggressive):
    start = start or (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    end = end or datetime.utcnow().strftime("%Y-%m-%d")
    # fetch price
    df = fetch_ohlcv_auto(symbol, timeframe, start)
    if df is None or len(df) < 10:
        raise ValueError("No hay datos OHLCV para el símbolo solicitado.")
    # news
    news_api = os.environ.get("NEWSAPI_KEY")
    news_df = fetch_news_features(symbol.split(":")[-1], start, end, api_key=news_api)
    # align daily news to df index (resample)
    news_daily = news_df.reindex(pd.date_range(start=df.index.min().date(), end=df.index.max().date(), freq='D'), fill_value=0)
    news_daily.index = pd.to_datetime(news_daily.index)
    # merge: forward-fill news to intraday indexes by date
    df = df.copy()
    df["date_only"] = df.index.floor("D")
    df = df.join(news_daily, on="date_only")
    df = df.fillna(0)
    # run strategy/backtest
    trades, equity = simulate_strategy(df, strategy, capital, risk, leverage, aggressive)
    metrics = metrics_from_equity(equity)
    # baseline: buy&hold
    _, bh_equity = simulate_strategy(df, "buy_hold", capital, risk, leverage, False)
    # monte carlo using close series
    mc = monte_carlo_final_equity(df["close"], mc_runs=mc_sims if mc_sims else 500)
    mc_fig = make_mc_fig(mc)
    # p-value bootstrap
    pval = bootstrap_pvalue(equity, bh_equity, n=1000)
    # prepare outputs
    summary = f"Periodo: {df.index.min().date()} — {df.index.max().date()}\nSimulaciones MC: {mc_sims}\nP-valor bootstrap (vs buy&hold): {pval:.3f}"
    metrics_text = json.dumps(metrics, indent=2)
    price_fig = make_price_fig(df, symbol)
    equity_fig = make_equity_fig(equity)
    trades_df = pd.DataFrame(trades)
    # top news preview
    news_preview = []
    try:
        top = news_df.sort_values("news_count", ascending=False).head(10)
        for idx,row in top.iterrows():
            news_preview.append(f"{idx.date()}: count={row['news_count']}, avg_sent={row['avg_sent']:.3f}")
    except Exception:
        news_preview = []
    return {
        "summary": summary,
        "metrics_text": metrics_text,
        "price_fig": price_fig,
        "equity_fig": equity_fig,
        "mc_fig": mc_fig,
        "trades_df": trades_df,
        "news_list": news_preview,
        "raw_df": df,
        "metrics": metrics,
        "pval": pval
    }

def export_results_file(res):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    out = {
        "summary": res.get("summary"),
        "metrics": res.get("metrics"),
        "pval": res.get("pval"),
        "raw_head": res.get("raw_df").head(10).to_dict(orient="list")
    }
    tmp.write(json.dumps(out, default=str, indent=2).encode())
    tmp.close()
    return tmp.name
# EOF