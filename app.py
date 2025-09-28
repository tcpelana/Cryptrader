# app.py
"""
CrypTrader - Interfaz Gradio mejorada (versión con integración de alertas)
Coloca este archivo en la raíz del repo. Requiere:
- data_loader.fetch_ohlcv
- news_sentiment.fetch_news_agg (opcional)
- predictor.build_features, predictor.train_and_predict
- backend.simulate_strategy_with_costs (si quieres simulador con costos)
- tvhook.py escribe alerts.json (para procesar alertas)

Cómo usar:
1) Activa tu .venv
2) python app.py
3) Abre http://localhost:7860
"""

import os
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# imports del proyecto
from data_loader import fetch_ohlcv
from news_sentiment import fetch_news_agg
from predictor import build_features, train_and_predict

# optional backend simulator
try:
    from backend import simulate_strategy_with_costs
except Exception:
    simulate_strategy_with_costs = None

# alert storage
ALERTS_FILE = Path("alerts.json")
LATEST_FILE = Path("latest_alerts.json")

# gradio & plotly
try:
    import gradio as gr
    import plotly.graph_objects as go
except Exception:
    gr = None
    go = None

# -----------------------
# Helpers
# -----------------------
def safe_read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return []

def read_alerts(limit: int = 200):
    """Lee alerts.json o latest_alerts.json; devuelve lista de dicts (más recientes primero)."""
    arr = safe_read_json(ALERTS_FILE)
    if not arr:
        arr = safe_read_json(LATEST_FILE)
    try:
        arr_sorted = sorted(arr, key=lambda x: x.get("time", ""), reverse=True)
    except Exception:
        arr_sorted = arr
    return arr_sorted[:limit]

def format_alerts_for_table(alerts):
    """Convierte lista de alerts a tabla simple para gradio dataframe [time, symbol, signal]"""
    rows = []
    for a in alerts:
        t = a.get("time", "") or ""
        sym = a.get("symbol", "") or ""
        sig = a.get("signal", "") or ""
        rows.append([t, sym, sig])
    return rows

def price_fig(df: pd.DataFrame, symbol: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="candles"))
    fig.update_layout(title=f"{symbol} — Precio", xaxis_title="Fecha", yaxis_title="USD", template="plotly_dark", height=420)
    return fig

def equity_fig(eq_series: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_series.index, y=eq_series.values, name="Equity", mode="lines"))
    fig.update_layout(title="Equity", xaxis_title="Fecha", yaxis_title="Capital", template="plotly_dark", height=320)
    return fig

# -----------------------
# Core pipeline por símbolo
# -----------------------
def run_pipeline_for_symbol(symbol: str, start: str = None, end: str = None, include_news: bool = False, capital: float = 10000.0):
    """
    Ejecuta flujo: descarga precios, noticias (opcional), genera features, predice y simula.
    Retorna summary dict, figures y tabla de predicciones.
    """
    try:
        symbol = symbol.strip().upper()
        if not end:
            end = datetime.utcnow().strftime("%Y-%m-%d")
        if not start:
            start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

        df = fetch_ohlcv(symbol, timeframe="1d", start=start, end=end)
        if df is None or len(df) == 0:
            return {"error": f"No price data for {symbol}"}, None, None, None

        news_df = None
        if include_news:
            news_df = fetch_news_agg(symbol.split(":")[-1], start, end)

        feat = build_features(df, news_df=news_df)
        pred = train_and_predict(feat)
        pred = pred.sort_index()
        pred["signal"] = pred["pred_label"].apply(lambda x: "BUY" if int(x) == 1 else "SELL")

        # simulate
        trades = []
        equity_s = pd.Series(dtype=float)
        if simulate_strategy_with_costs is not None:
            trades, equity_s = simulate_strategy_with_costs(pred, capital=capital, fee_pct=float(os.environ.get("FEE_PCT", 0.00075)), slippage_pct=float(os.environ.get("SLIPPAGE_PCT", 0.0005)))
        else:
            equity_s = pd.Series(data=(capital * (pred["close"] / pred["close"].iloc[0])).values, index=pred.index)

        pfig = price_fig(df, symbol) if go else None
        efig = equity_fig(equity_s) if go else None
        table = pred[["close", "ret1", "pred_proba_up", "pred_label"]].tail(50).reset_index().rename(columns={"index":"date"})

        summary = {
            "symbol": symbol,
            "period": f"{pred.index.min().date()} to {pred.index.max().date()}",
            "final_equity": float(equity_s.iloc[-1]) if len(equity_s) else float(capital),
            "n_rows": int(len(pred)),
            "notes": "Baseline: RandomForest if sklearn available; fallback momentum rule."
        }
        return summary, pfig, efig, table, trades
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": str(e), "trace": tb}, None, None, None, []

# -----------------------
# Process alerts (batch)
# -----------------------
def process_alerts_and_simulate(alerts_list, capital: float = 10000.0, include_news=False):
    res = []
    for a in alerts_list:
        try:
            sym = a.get("symbol", "BTC-USD")
            # normalize symbol fallback
            s = sym
            summary, pfig, efig, table, trades = run_pipeline_for_symbol(s, include_news=include_news, capital=capital)
            # keep only lightweight pieces for returning in batch
            res.append({
                "alert": a,
                "summary": summary,
                "last_preds": table.tail(3).to_dict(orient="records") if isinstance(table, pd.DataFrame) else None,
                "n_trades": len(trades),
                "final_equity": summary.get("final_equity") if isinstance(summary, dict) else None
            })
        except Exception as e:
            res.append({"alert": a, "error": str(e)})
    return res

# -----------------------
# Gradio UI
# -----------------------
def build_ui():
    if gr is None:
        print("Por favor instala gradio: pip install gradio")
        return None

    with gr.Blocks(title="CrypTrader") as demo:
        gr.Markdown("# 🔥 CrypTrader\n**Predictor de criptomonedas** — histórico + noticias + alertas TradingView\n---")
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Parámetros")
                input_symbol = gr.Textbox(label="Símbolo (ej. BINANCE:BTCUSDT o BTC-USD)", value="BINANCE:BTCUSDT")
                input_start = gr.Textbox(label="Fecha inicio (YYYY-MM-DD)", value=(datetime.utcnow()-timedelta(days=365)).strftime("%Y-%m-%d"))
                input_end = gr.Textbox(label="Fecha fin (YYYY-MM-DD)", value=datetime.utcnow().strftime("%Y-%m-%d"))
                include_news = gr.Checkbox(label="Incluir noticias (NewsAPI)", value=False)
                capital = gr.Number(label="Capital inicial (USD)", value=10000)
                run_btn = gr.Button("Ejecutar pipeline")
                gr.Markdown("**Alertas**")
                load_alerts_btn = gr.Button("Cargar últimas alertas")
                process_alerts_btn = gr.Button("Procesar alertas y simular (batch)")
                include_news_alerts = gr.Checkbox(label="Incluir noticias al procesar alertas", value=False)
                gr.Markdown("**Ajustes**")
                fee = gr.Number(label="Comisión (fractional, ej. 0.00075)", value=float(os.environ.get("FEE_PCT", 0.00075)))
                slippage = gr.Number(label="Slippage (fractional, ej. 0.0005)", value=float(os.environ.get("SLIPPAGE_PCT", 0.0005)))
                # small help text
                gr.Markdown("Tips: usa `ngrok` para exponer tu `tvhook.py` durante desarrollo; configura `TV_HOOK_TOKEN` como variable de entorno y usa esa misma en TradingView.")
            with gr.Column(scale=3):
                out_summary = gr.JSON(label="Resumen")
                out_price = gr.Plot(label="Precio")
                out_equity = gr.Plot(label="Equity")
                out_table = gr.Dataframe(headers=["date", "close", "ret1", "pred_proba_up", "pred_label"], label="Predicciones recientes")
                gr.Markdown("### Últimas alertas")
                alerts_table = gr.Dataframe(headers=["time","symbol","signal"], label="Últimas alertas")
                alerts_results = gr.JSON(label="Resultados de procesamiento de alertas (batch)")

        # Main button callbacks
        def _run_pipeline(symbol, start, end, include_news_flag, capital_val, fee_val, slippage_val):
            # write env overrides for simulator
            os.environ["FEE_PCT"] = str(fee_val)
            os.environ["SLIPPAGE_PCT"] = str(slippage_val)
            out = run_pipeline_for_symbol(symbol, start, end, include_news_flag, capital_val)
            # run_pipeline_for_symbol returns summary, pfig, efig, table, trades
            if isinstance(out, tuple) and len(out) >= 5:
                summary, pfig, efig, table, trades = out
                return summary, pfig, efig, table
            else:
                return out, None, None, None

        run_btn.click(fn=_run_pipeline, inputs=[input_symbol, input_start, input_end, include_news, capital, fee, slippage], outputs=[out_summary, out_price, out_equity, out_table])

        # alerts callbacks
        def _load_alerts():
            arr = read_alerts(limit=200)
            return format_alerts_for_table(arr)

        def _process_alerts(include_news_flag, capital_val):
            arr = read_alerts(limit=20)
            results = process_alerts_and_simulate(arr, capital=float(capital_val), include_news=include_news_flag)
            return results

        load_alerts_btn.click(fn=_load_alerts, inputs=None, outputs=[alerts_table])
        process_alerts_btn.click(fn=_process_alerts, inputs=[include_news_alerts, capital], outputs=[alerts_results])

        # footer / credits
        gr.Markdown("---\n*CrypTrader — versión experimental. No es asesoramiento financiero. Usa con precaución.*\n")

    return demo

if __name__ == "__main__":
    demo = build_ui()
    if demo is not None:
        demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_PORT", 7860)), share=False)
