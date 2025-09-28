# backend.py
"""
Backend integrado para CrypTrader
- Fetchers: CCXT (Binance) y CoinGecko fallback
- News fetcher: NewsAPI (opcional) + VADER fallback
- Alerts: lectura segura de alerts.json
- Strategy simulator: fees, slippage, initial alert handling
- Metrics: CAGR, vol, sharpe, max drawdown
- Monte Carlo + bootstrap p-value
- run_analysis(): función principal que devuelve dict usado por app.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# optional imports lazily to avoid startup failures if not installed
def try_import(module_name):
    try:
        return __import__(module_name)
    except Exception:
        return None

ccxt = try_import("ccxt")
requests = try_import("requests")
vader_mod = try_import("vaderSentiment.vaderSentiment")

# utils (atomic write, seed, etc.)
try:
    from utils import atomic_write, set_global_seed, hash_payload  # utils.py must exist
except Exception:
    # minimal fallbacks
    def atomic_write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                f.write(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                f.write(str(data))

    def set_global_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)

    def hash_payload(payload):
        import hashlib
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

# Configs / ENV
ALERTS_FILE = Path(os.environ.get("ALERTS_FILE", "alerts.json"))
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", None)
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
START_CAPITAL = float(os.environ.get("START_CAPITAL", "10000"))
FEE_PCT = float(os.environ.get("FEE_PCT", "0.00075"))    # 0.075% default
SLIPPAGE_PCT = float(os.environ.get("SLIPPAGE_PCT", "0.0005"))  # 0.05% default
MC_DEFAULT_RUNS = int(os.environ.get("MC_DEFAULT_RUNS", "500"))
DEDUPE_TTL = int(os.environ.get("TV_DEDUPE_TTL", "30"))

set_global_seed(RANDOM_SEED)


# ---------------------------
# UTIL: alerts reading
# ---------------------------
def read_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Lee alerts.json de forma segura y devuelve las más recientes (limit)."""
    if not ALERTS_FILE.exists():
        return []
    try:
        with ALERTS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data[:limit]
    except Exception:
        return []


# ---------------------------
# FETCHERS (prices)
# ---------------------------
def fetch_ohlcv_ccxt_binance(symbol: str, timeframe: str = "1d", since: Optional[str] = None, limit: int = 1000) -> Optional[pd.DataFrame]:
    """
    Intenta obtener OHLCV desde Binance via CCXT.
    symbol: 'BTC/USDT' or 'BTCUSDT' or 'BINANCE:BTCUSDT'
    """
    if not ccxt:
        return None
    try:
        exchange = getattr(ccxt, "binance")()
        s = symbol.upper().replace("BINANCE:", "").replace("/", "")
        # CCXT symbol format 'BTC/USDT'
        if "/" not in s:
            s2 = s[:-4] + "/USDT" if s.endswith("USDT") else s
            # try simple split
            if len(s2) < 3:
                s2 = symbol
        else:
            s2 = symbol
        s2 = s2.replace("BINANCE:", "").replace("USDT", "/USDT") if "USDT" in s else s2
        # since in ms
        since_ms = None
        if since:
            since_ms = int(pd.Timestamp(since).timestamp() * 1000)
        all_rows = []
        while True:
            batch = exchange.fetch_ohlcv(s2, timeframe=timeframe, since=since_ms, limit=limit)
            if not batch:
                break
            all_rows += batch
            if len(batch) < limit:
                break
            since_ms = batch[-1][0] + 1
            time.sleep(0.2)
        if not all_rows:
            return None
        df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "vol"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "vol"]]
    except Exception:
        return None


def fetch_ohlcv_coingecko(symbol: str, vs_currency: str = "usd", days: int = 2000) -> Optional[pd.DataFrame]:
    """
    Fallback a CoinGecko: nota que CG usa ids (bitcoin) no símbolos.
    Intentamos convertir 'BTC'->'bitcoin' de forma simple (no exhaustiva).
    """
    if not requests:
        return None
    try:
        # derive id naive
        s = symbol.lower().replace("binance:", "").replace("/", "")
        cg_id = s
        # common mappings
        mapping = {"btc": "bitcoin", "eth": "ethereum", "bnb": "binancecoin"}
        if cg_id in mapping:
            cg_id = mapping[cg_id]
        url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["ts", "close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date")
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["vol"] = 0
        return df[["open", "high", "low", "close", "vol"]]
    except Exception:
        return None


def fetch_ohlcv(symbol: str, timeframe: str = "1d", since: Optional[str] = None) -> pd.DataFrame:
    """
    Wrapper: intenta CCXT Binance primero, luego CoinGecko fallback.
    Devuelve DataFrame o lanza ValueError si no hay datos.
    """
    df = fetch_ohlcv_ccxt_binance(symbol, timeframe, since=since)
    if df is not None and len(df) > 0:
        return df
    df = fetch_ohlcv_coingecko(symbol, days=2000)
    if df is not None and len(df) > 0:
        return df
    raise ValueError(f"No OHLCV data available for symbol {symbol}")


# ---------------------------
# NEWS fetcher + sentiment
# ---------------------------
def fetch_news_agg(query: str, start: str, end: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Retorna DataFrame diario indexado por fecha con columnas ['news_count','avg_sent']
    Si NEWSAPI_KEY no está presente devuelve zeros.
    """
    idx = pd.date_range(start=start, end=end, freq="D")
    zeros = pd.DataFrame(index=idx, data={"news_count": 0, "avg_sent": 0.0})
    if not api_key or not requests:
        return zeros
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "from": start, "to": end, "language": "en", "pageSize": 100, "apiKey": api_key}
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        articles = data.get("articles", [])
        if not articles:
            return zeros
        # sentiment: VADER if available
        scores = []
        rows = []
        analyzer = None
        if vader_mod:
            analyzer = vader_mod.SentimentIntensityAnalyzer()
        for a in articles:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            txt = (title + ". " + desc)[:2000]
            dt_str = a.get("publishedAt") or a.get("published")
            try:
                dt = pd.to_datetime(dt_str).tz_convert(None).date() if dt_str else None
            except Exception:
                try:
                    dt = pd.to_datetime(dt_str).date()
                except Exception:
                    dt = None
            if dt is None:
                continue
            if analyzer:
                s = analyzer.polarity_scores(txt)["compound"]
            else:
                # fallback naive polarity: presence of words
                s = 0.0
                if any(w in txt.lower() for w in ["gain", "surge", "rise", "bull"]):
                    s += 0.5
                if any(w in txt.lower() for w in ["drop", "crash", "dump", "bear"]):
                    s -= 0.5
            rows.append((pd.to_datetime(dt), s))
        if not rows:
            return zeros
        ndf = pd.DataFrame(rows, columns=["date", "sent"])
        agg = ndf.groupby("date").agg(news_count=("sent", "count"), avg_sent=("sent", "mean"))
        full_idx = pd.date_range(start=start, end=end, freq="D")
        agg = agg.reindex(full_idx, fill_value=0)
        agg.index = pd.to_datetime(agg.index)
        return agg
    except Exception:
        return zeros


# ---------------------------
# STRATEGIES / BACKTEST
# ---------------------------
def ensure_no_future_features(df: pd.DataFrame) -> bool:
    """
    Placeholder check: in practice, ensure features were built with shifts and no future leakage.
    Returns True if check passed (lightweight).
    """
    # Implement any custom checks you want; for now return True as default.
    return True


def simulate_strategy(df: pd.DataFrame,
                      capital: float = START_CAPITAL,
                      fee_pct: float = FEE_PCT,
                      slippage_pct: float = SLIPPAGE_PCT,
                      initial_signal: Optional[str] = None) -> Tuple[List[Dict[str, Any]], pd.Series]:
    """
    Simulador de estrategia sencillo (long only).
    - df must contain 'close' column.
    - initial_signal: "LONG" or "SHORT" or None
    Returns: trades list y equity series indexado como df.index
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    df = df.copy().reset_index()
    trades = []
    cash = float(capital)
    pos_units = 0.0

    for i, row in df.iterrows():
        price = float(row["close"])
        sig = row.get("signal", None)  # signal column optional (BUY/SELL)
        # initial signal at first bar
        if i == 0 and initial_signal == "LONG" and pos_units == 0:
            # buy as much as possible
            # account for slippage: effective price higher
            eff_price = price * (1 + slippage_pct)
            units = cash / eff_price
            fee = cash * fee_pct
            pos_units = units
            cash = cash - units * eff_price - fee
            trades.append({"date": row["date"].isoformat(), "side": "buy", "price": eff_price, "size": units, "fee": fee})
        # reactive signals
        if sig == "BUY" and pos_units == 0:
            eff_price = price * (1 + slippage_pct)
            units = cash / eff_price
            fee = cash * fee_pct
            pos_units = units
            cash = cash - units * eff_price - fee
            trades.append({"date": row["date"].isoformat(), "side": "buy", "price": eff_price, "size": units, "fee": fee})
        elif sig == "SELL" and pos_units > 0:
            eff_price = price * (1 - slippage_pct)
            proceeds = pos_units * eff_price
            fee = proceeds * fee_pct
            cash = cash + proceeds - fee
            trades.append({"date": row["date"].isoformat(), "side": "sell", "price": eff_price, "size": pos_units, "fee": fee})
            pos_units = 0.0
        # record equity
        equity = cash + pos_units * price
        df.at[i, "equity"] = equity

    # forward-fill equity and fill NaNs
    if "equity" not in df.columns:
        df["equity"] = capital
    df["equity"] = df["equity"].ffill().fillna(capital)
    eq_series = pd.Series(df["equity"].values, index=pd.DatetimeIndex(df["date"]))
    return trades, eq_series


# ---------------------------
# Compatibilidad: simulate_strategy_with_costs
# ---------------------------
# Esta función es autocontenida y sirve para que app.py pueda importarla.
# Asume que el DataFrame de entrada tiene columna 'close' y un index de fechas.
def simulate_strategy_with_costs(df, capital=None, fee_pct=None, slippage_pct=None, initial_signal=None):
    """
    Simulador compatible con app.py:
    - df: pd.DataFrame con columna 'close' (index datetime o columna 'date')
    - capital: float (si None usa START_CAPITAL)
    - fee_pct: comisiones (si None usa FEE_PCT)
    - slippage_pct: slippage (si None usa SLIPPAGE_PCT)
    - initial_signal: "LONG" | "SHORT" | None
    Retorna: (trades_list, equity_series) donde equity_series es pd.Series indexado por fechas.
    """
    try:
        import pandas as _pd
        import numpy as _np
    except Exception:
        raise RuntimeError("pandas/numpy required for simulate_strategy_with_costs")

    # usar valores por defecto del módulo si no se pasan
    if capital is None:
        try:
            capital = float(globals().get("START_CAPITAL", 10000))
        except Exception:
            capital = 10000.0
    if fee_pct is None:
        try:
            fee_pct = float(globals().get("FEE_PCT", 0.00075))
        except Exception:
            fee_pct = 0.00075
    if slippage_pct is None:
        try:
            slippage_pct = float(globals().get("SLIPPAGE_PCT", 0.0005))
        except Exception:
            slippage_pct = 0.0005

    # normalizar df
    df_local = df.copy()
    # si 'date' es columna, usarla; si no, index debe ser datetime
    if "date" in df_local.columns:
        try:
            df_local = df_local.set_index(pd.to_datetime(df_local["date"]))
        except Exception:
            df_local = df_local.set_index("date")
    # asegurar columna 'close'
    if "close" not in df_local.columns and "Close" in df_local.columns:
        df_local["close"] = df_local["Close"]
    if "close" not in df_local.columns:
        raise ValueError("DataFrame must contain 'close' column for simulate_strategy_with_costs")

    trades = []
    cash = float(capital)
    pos_units = 0.0
    equity_list = []
    dates = []

    for idx, row in df_local.iterrows():
        price = float(row["close"])
        # obtener señal si existe
        sig = None
        # soportar columna signal o señal en row
        if "signal" in df_local.columns:
            try:
                sig = str(row["signal"])
            except Exception:
                sig = None

        # manejar initial signal en la primera fila
        if len(equity_list) == 0 and initial_signal is not None and pos_units == 0:
            if str(initial_signal).upper() == "LONG":
                eff_price = price * (1 + slippage_pct)
                units = cash / eff_price if eff_price > 0 else 0.0
                fee = cash * fee_pct
                pos_units = units
                cash = cash - units * eff_price - fee
                trades.append({"date": str(idx), "side": "buy", "price": eff_price, "size": units, "fee": fee})
            elif str(initial_signal).upper() == "SHORT":
                # short handling simple: no impl. por defecto lo ignoramos (long-only)
                pass

        # señales en data
        if sig is not None:
            sig_upper = str(sig).upper()
            if "BUY" in sig_upper and pos_units == 0:
                eff_price = price * (1 + slippage_pct)
                units = cash / eff_price if eff_price > 0 else 0.0
                fee = cash * fee_pct
                pos_units = units
                cash = cash - units * eff_price - fee
                trades.append({"date": str(idx), "side": "buy", "price": eff_price, "size": units, "fee": fee})
            elif "SELL" in sig_upper and pos_units > 0:
                eff_price = price * (1 - slippage_pct)
                proceeds = pos_units * eff_price
                fee = proceeds * fee_pct
                cash = cash + proceeds - fee
                trades.append({"date": str(idx), "side": "sell", "price": eff_price, "size": pos_units, "fee": fee})
                pos_units = 0.0

        equity = cash + pos_units * price
        equity_list.append(equity)
        dates.append(idx)

    # construir series de equity
    if len(dates) == 0:
        # DataFrame vacío -> devolver capital inicial
        eq_series = pd.Series([capital], index=pd.DatetimeIndex([pd.Timestamp.now()]))
    else:
        eq_series = pd.Series(equity_list, index=pd.DatetimeIndex(dates))
    return trades, eq_series


# ---------------------------
# METRICS, MC, BOOTSTRAP
# ---------------------------
def metrics_from_equity(equity: pd.Series, periods_per_year: int = 252) -> Dict[str, Any]:
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    rets = eq.pct_change().dropna()
    total_periods = len(eq)
    ann_return = (eq.iloc[-1] / eq.iloc[0]) ** (periods_per_year / total_periods) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = drawdown.min()
    return {"ann_return": float(ann_return), "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_drawdown": float(max_dd), "final_equity": float(eq.iloc[-1])}


def monte_carlo_final_prices(close_series: pd.Series, mc_runs: int = MC_DEFAULT_RUNS) -> List[float]:
    rets = close_series.pct_change().dropna().values
    if len(rets) == 0:
        return []
    mc = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(mc_runs):
        sample = rng.choice(rets, size=len(rets), replace=True)
        price = float(close_series.iloc[0])
        for r in sample:
            price = price * (1 + float(r))
        mc.append(price)
    return mc


def bootstrap_pvalue(strategy_eq: pd.Series, baseline_eq: pd.Series, n: int = 1000) -> float:
    s = strategy_eq.pct_change().dropna().values
    b = baseline_eq.pct_change().dropna().values
    if len(s) == 0 or len(b) == 0:
        return 1.0
    diff_obs = np.mean(s) - np.mean(b)
    pooled = np.concatenate([s, b])
    rng = np.random.default_rng(RANDOM_SEED)
    count = 0
    for _ in range(n):
        perm = rng.choice(pooled, size=len(pooled), replace=True)
        a = perm[:len(s)]
        b_ = perm[len(s):len(s) + len(b)]
        if np.mean(a) - np.mean(b_) >= diff_obs:
            count += 1
    return float(count) / float(n)


# ---------------------------
# PLOTTING HELPERS (Plotly)
# ---------------------------
def make_price_fig(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name=f"{symbol} - close"))
    fig.update_layout(title=f"{symbol} - Price", xaxis_title="date", yaxis_title="USD", template="plotly_dark")
    return fig


def make_equity_fig(eq: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="equity"))
    fig.update_layout(title="Equity", xaxis_title="date", yaxis_title="equity", template="plotly_dark")
    return fig


def make_mc_fig(mc_results: List[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mc_results, nbinsx=50))
    fig.update_layout(title="Monte Carlo: Distribución precio final", template="plotly_dark")
    return fig


# ---------------------------
# MAIN: run_analysis
# ---------------------------
def run_analysis(symbol: str = "BINANCE:BTCUSDT",
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 timeframe: str = "1d",
                 capital: Optional[float] = None,
                 mc_sims: Optional[int] = None,
                 strategy: str = "ma_cross",
                 risk_pct: float = 1.0,
                 leverage: float = 1.0,
                 aggressive: bool = False) -> Dict[str, Any]:
    """
    Pipeline principal:
    - fetch OHLCV
    - fetch news (optional)
    - load alerts
    - merge features
    - apply simple strategy/backtest
    - compute metrics, MC, p-value
    Retorna dict con keys:
      summary, metrics_text, price_fig, equity_fig, mc_fig, trades_df, news_list, raw_df, metrics, pval
    """
    # defaults
    mc_sims = int(mc_sims) if mc_sims is not None else MC_DEFAULT_RUNS
    capital = float(capital) if capital is not None else START_CAPITAL

    # Dates default
    if not end:
        end = datetime.utcnow().strftime("%Y-%m-%d")
    if not start:
        # default 365 days back
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Fetch prices
    try:
        df = fetch_ohlcv(symbol, timeframe=timeframe, since=start)
        # normalize index name
        df = df.rename_axis("date")
        # ensure close col exists
        if "close" not in df.columns and "Close" in df.columns:
            df["close"] = df["Close"]
        # ensure lower columns
        df = df.rename(columns={c: c.lower() for c in df.columns})
    except Exception as e:
        raise RuntimeError(f"Error fetching OHLCV for {symbol}: {e}")

    # Fetch news features
    try:
        news_df = fetch_news_agg(symbol.split(":")[-1], start, end, api_key=NEWSAPI_KEY)
    except Exception:
        news_df = pd.DataFrame(index=pd.date_range(start=start, end=end, freq='D'), data={"news_count": 0, "avg_sent": 0.0})

    # Align news to price index (forward fill daily)
    try:
        news_daily = news_df.reindex(pd.date_range(start=df.index.min().date(), end=df.index.max().date(), freq='D'), fill_value=0)
        news_daily.index = pd.to_datetime(news_daily.index)
        df["date_only"] = df.index.floor("D")
        df = df.join(news_daily, on="date_only")
        df = df.fillna(0)
    except Exception:
        df["news_count"] = 0
        df["avg_sent"] = 0.0

    # Load alerts and interpret last alert for this symbol
    alerts = read_alerts(50)
    last_alert = alerts[0] if alerts else None
    alert_signal = None
    if last_alert:
        s = (str(last_alert.get("signal") or last_alert.get("type") or last_alert.get("action") or "")).upper()
        if "LONG" in s or "BUY" in s:
            alert_signal = "LONG"
        elif "SHORT" in s or "SELL" in s:
            alert_signal = "SHORT"

    # Optionally validate alerts belong to symbol (prevent cross-signal pollution)
    # We do a best effort: compare normalized symbol strings
    try:
        def _norm(s): return str(s).upper().replace("BINANCE:", "").replace("/", "")
        if last_alert and last_alert.get("symbol"):
            if _norm(last_alert.get("symbol")) not in _norm(symbol):
                # ignore alert for different symbol
                alert_signal = None
    except Exception:
        pass

    # Build a very small example signal column (for demo strategies).
    # In production replace with real strategy feature engineering.
    df = df.copy()
    # simple signal: momentum sign of 1-period return
    df["ret1"] = df["close"].pct_change().fillna(0)
    df["signal"] = df["ret1"].apply(lambda x: "BUY" if x > 0 else "SELL")

    # Ensure no future leakage (placeholder check)
    if not ensure_no_future_features(df):
        raise RuntimeError("Detected potential future leakage in features")

    # Simulate strategy (use the alert_signal to open initial position if present)
    trades, equity = simulate_strategy(df, capital=capital, fee_pct=FEE_PCT, slippage_pct=SLIPPAGE_PCT, initial_signal=alert_signal)

    # Baseline buy & hold
    try:
        bh_trades, bh_equity = simulate_strategy(df.assign(signal="BUY"), capital=capital, fee_pct=FEE_PCT, slippage_pct=SLIPPAGE_PCT, initial_signal=None)
    except Exception:
        bh_trades, bh_equity = [], equity

    # Metrics
    metrics = metrics_from_equity(equity)
    metrics_bh = metrics_from_equity(bh_equity) if bh_equity is not None else {}

    # Monte Carlo
    mc_results = monte_carlo_final_prices(df["close"], mc_runs=mc_sims)
    mc_fig = make_mc_fig(mc_results) if len(mc_results) > 0 else None

    # bootstrap p-value versus buy & hold
    pval = bootstrap_pvalue(equity, bh_equity, n=1000)

    # Prepare plots
    price_fig = make_price_fig(df, symbol)
    equity_fig = make_equity_fig(equity)

    # Prepare news preview & alert preview
    news_preview = []
    try:
        topnews = news_daily.sort_values("news_count", ascending=False).head(10)
        for idx, row in topnews.iterrows():
            news_preview.append(f"{idx.date()}: count={int(row['news_count'])}, avg_sent={row['avg_sent']:.3f}")
    except Exception:
        news_preview = []

    alert_preview = []
    for a in alerts[:10]:
        ts = a.get("_received_at", a.get("time", "unknown"))
        sym = a.get("symbol") or a.get("ticker") or ""
        sig = a.get("signal") or a.get("type") or a.get("action") or a.get("message") or ""
        alert_preview.append(f"{ts} | {sym} | {sig}")

    summary = f"Periodo datos: {df.index.min().date()} — {df.index.max().date()}\nSimulaciones MC: {mc_sims}\nBootstrap p-value vs BH: {pval:.3f}"

    # trades -> DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "side", "price", "size", "fee"])

    out = {
        "summary": summary,
        "metrics_text": json.dumps(metrics, indent=2),
        "price_fig": price_fig,
        "equity_fig": equity_fig,
        "mc_fig": mc_fig,
        "trades_df": trades_df,
        "news_list": news_preview,
        "alerts_preview": alert_preview,
        "raw_df": df,
        "metrics": metrics,
        "metrics_bh": metrics_bh,
        "pval": pval,
        "mc_results": mc_results
    }
    return out


# ---------------------------
# Small convenience: export results to json/csv
# ---------------------------
def export_results_file(res: Dict[str, Any], prefix: str = "cryptrader_results") -> str:
    """Export minimal summary + metrics + sample raw_df to JSON file and return path."""
    out = {
        "summary": res.get("summary"),
        "metrics": res.get("metrics"),
        "pval": res.get("pval"),
        "alerts": res.get("alerts_preview", []),
    }
    # attach a small head of raw df
    try:
        raw = res.get("raw_df")
        if isinstance(raw, pd.DataFrame):
            out["raw_head"] = raw.head(20).to_dict(orient="list")
    except Exception:
        out["raw_head"] = None
    fname = f"{prefix}_{int(time.time())}.json"
    atomic_write(fname, out)
    return str(Path(fname).resolve())
