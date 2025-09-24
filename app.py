# cryptrader_app.py - CrypTrader (TradingView + CCXT + fallback + debug)
# Requisitos: gradio, pandas, numpy, requests, plotly, ccxt
# Pega este archivo como `app.py` en tu Hugging Face Space (SDK: Gradio)

import gradio as gr
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import plotly.express as px
import os
import json
import tempfile
import ccxt

CUSTOM_CSS = """
html, body { background: #0b0b0d; color: #ececec; font-family: Inter, Arial, sans-serif; }
.gradio-container { padding: 12px !important; }
input, textarea, select, .gr-input { border: 1px solid #2b2b2f !important; box-shadow: none !important; border-radius: 6px !important; padding: 6px !important; background:#0f1113; color:#ececec; }
.gr-button, button { background: #0b5fff !important; color: white !important; border: none !important; padding: 8px 12px !important; border-radius: 6px !important; }
.gradio-textbox[readonly] { background: #0f1113 !important; border: 1px solid #232326 !important; color:#ececec; }
.plotly-graph-div, .gradio-plot { max-height: 520px !important; overflow: auto !important; }
.gr-block { max-width: 1200px; margin: 6px auto; }
"""


def tradingview_widget_html(exchange_symbol="BINANCE:BTCUSDT", width="100%", height=420):
    secret_html = os.environ.get("TRADINGVIEW_WIDGET_HTML", "").strip()
    if secret_html:
        return secret_html.replace("{SYMBOL}", exchange_symbol)
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_{exchange_symbol.replace(':','_')}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
      "width": "{width}",
      "height": {height},
      "symbol": "{exchange_symbol}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "es",
      "container_id": "tradingview_{exchange_symbol.replace(':','_')}"
      }});
      </script>
    </div>
    """
    return html


# --- fetch_market_data con DEBUG y fallback ---

def fetch_market_data(symbol_input, timeframe="1d", start_dt=None, end_dt=None):
    debug_msgs = []
    if start_dt is None:
        start_dt = datetime.utcnow() - pd.Timedelta(days=365*2)
    if end_dt is None:
        end_dt = datetime.utcnow()

    # 1) TradingView API attempt
    tv_api_url = os.environ.get("TRADINGVIEW_API_URL", "").strip()
    tv_api_key = os.environ.get("TRADINGVIEW_API_KEY", "").strip()
    if tv_api_url:
        try:
            params = {"symbol": symbol_input, "from": int(time.mktime(start_dt.timetuple())), "to": int(time.mktime(end_dt.timetuple())), "interval": timeframe}
            headers = {}
            if tv_api_key:
                headers["Authorization"] = f"Bearer {tv_api_key}"
            debug_msgs.append(f"Intentando TradingView API: {tv_api_url} params={params} headers_set={bool(headers)}")
            r = requests.get(tv_api_url, params=params, headers=headers, timeout=30)
            debug_msgs.append(f"TradingView status_code={r.status_code}")
            r.raise_for_status()
            body = r.json()
            if "prices" in body and body["prices"]:
                debug_msgs.append("TradingView returned prices")
                return {"prices": body["prices"], "debug": debug_msgs}
            if "candles" in body:
                prices = []
                for c in body["candles"]:
                    ts = int(c.get("t", 0))
                    price = c.get("c", None)
                    if ts and price is not None:
                        prices.append([ts, float(price)])
                if prices:
                    debug_msgs.append("TradingView returned candles -> mapped to prices")
                    return {"prices": prices, "debug": debug_msgs}
            debug_msgs.append("TradingView returned no usable data")
        except Exception as e:
            debug_msgs.append(f"TradingView error: {repr(e)}")
            # continue to next source

    # 2) CCXT attempt
    try:
        debug_msgs.append("Intentando fuente CCXT")
        if ":" in symbol_input:
            exch, pair = symbol_input.split(":",1)
            exchange_id = exch.lower()
            pair_raw = pair.upper()
            if "/" not in pair_raw:
                if len(pair_raw) > 6:
                    base = pair_raw[:-4]
                    quote = pair_raw[-4:]
                else:
                    base = pair_raw[:-3]
                    quote = pair_raw[-3:]
                symbol_ccxt = f"{base}/{quote}"
            else:
                symbol_ccxt = pair_raw
            debug_msgs.append(f"CCXT: exchange={exchange_id} symbol={symbol_ccxt}")
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            api_key = os.environ.get(f"{exchange_id.upper()}_API_KEY", "").strip()
            api_secret = os.environ.get(f"{exchange_id.upper()}_API_SECRET", "").strip()
            if api_key and api_secret:
                exchange.apiKey = api_key
                exchange.secret = api_secret
            since_ms = int(start_dt.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, since=since_ms)
            prices = [[int(row[0]), float(row[4])] for row in ohlcv]
            if prices:
                debug_msgs.append(f"CCXT returned {len(prices)} rows")
                return {"prices": prices, "debug": debug_msgs}
        else:
            pair = symbol_input.replace("-", "/").upper()
            debug_msgs.append(f"CCXT default BINANCE symbol={pair}")
            exchange = ccxt.binance({"enableRateLimit": True})
            since_ms = int(start_dt.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms)
            prices = [[int(row[0]), float(row[4])] for row in ohlcv]
            if prices:
                debug_msgs.append(f"CCXT Binance returned {len(prices)} rows")
                return {"prices": prices, "debug": debug_msgs}
    except Exception as e:
        debug_msgs.append(f"CCXT error: {repr(e)}")

    # 3) CoinCap fallback
    try:
        debug_msgs.append("Intentando fallback CoinCap")
        cc_id = symbol_input.lower().split(":")[-1]
        if "/" in cc_id:
            cc_id = cc_id.split("/")[0].lower()
        start_ms = int(start_dt.timestamp()) * 1000
        end_ms = int(end_dt.timestamp()) * 1000
        cc_url = f"https://api.coincap.io/v2/assets/{cc_id}/history"
        cc_params = {"interval":"d1", "start": start_ms, "end": end_ms}
        rr = requests.get(cc_url, params=cc_params, timeout=30)
        debug_msgs.append(f"CoinCap status={rr.status_code}")
        rr.raise_for_status()
        data = rr.json().get("data", [])
        prices = []
        for item in data:
            if "priceUsd" in item and item["priceUsd"] is not None:
                ts = int(pd.to_datetime(item["date"]).timestamp() * 1000)
                prices.append([ts, float(item["priceUsd"])])
        if prices:
            debug_msgs.append(f"CoinCap returned {len(prices)} rows")
            return {"prices": prices, "debug": debug_msgs}
    except Exception as e:
        debug_msgs.append(f"CoinCap error: {repr(e)}")

    # No source succeeded
    raise RuntimeError("No se pudo obtener datos. Debug:\n" + "\n".join(debug_msgs))


# --- run_predict con manejo de debug ---

def run_predict(symbol_input,
                start_date_str="2022-01-01",
                end_date_str=None,
                capital=2000.0,
                num_sim=1000,
                timeframe="1d"):
    try:
        start_dt = pd.to_datetime(start_date_str)
    except Exception as e:
        return f"ERROR: Fecha inicio inválida ({e})", "", None, None, None

    if end_date_str is None or str(end_date_str).strip()=="":
        end_dt = pd.to_datetime(datetime.utcnow())
    else:
        try:
            end_dt = pd.to_datetime(end_date_str)
        except Exception as e:
            return f"ERROR: Fecha fin inválida ({e})", "", None, None, None

    symbol = str(symbol_input).strip()
    if symbol == "":
        return "ERROR: Introduce un símbolo (ej: BINANCE:BTCUSDT o BTC/USDT)", "", None, None, None

    try:
        fetched = fetch_market_data(symbol, timeframe=timeframe, start_dt=start_dt.to_pydatetime(), end_dt=end_dt.to_pydatetime())
        if isinstance(fetched, dict) and "debug" in fetched:
            debug_lines = "\n".join(fetched["debug"])
        else:
            debug_lines = ""
        data = fetched.get("prices", []) if isinstance(fetched, dict) else fetched
        prices = pd.DataFrame(data, columns=["ts","price"])
        if prices.empty:
            return f"ERROR: No hay datos para '{symbol}' en el rango indicado. Debug:\n{debug_lines}", "", None, None, None
        prices["date"] = pd.to_datetime(prices["ts"], unit="ms")
        prices = prices.set_index("date").resample("1D").last().ffill()
        prices["return"] = prices["price"].pct_change().fillna(0)
    except Exception as e:
        return f"ERROR al traer datos: {repr(e)}", "", None, None, None

    mu = prices["return"].mean()
    sigma = prices["return"].std()
    ann_return = ((1+mu)**252 - 1)
    ann_vol = sigma * np.sqrt(252)
    sharpe = (ann_return) / ann_vol if ann_vol>0 else float("nan")

    rets = prices["return"].values
    np.random.seed(42)
    n_sim = int(max(100, min(5000, num_sim)))
    sims = np.zeros(n_sim)
    for i in range(n_sim):
        sampled = np.random.choice(rets, size=len(rets), replace=True)
        sims[i] = np.prod(1+sampled)
    ci_low = np.percentile(sims, 2.5)
    ci_high = np.percentile(sims, 97.5)

    resumen = (
        f"Periodo: {prices.index.min().date()} — {prices.index.max().date()}\n"
        f"Return anual aprox: {ann_return:.2%}  | Volatilidad anual: {ann_vol:.2%}  | Sharpe aprox: {sharpe:.2f}\n"
        f"MC 95% intervalo multiplicador final: [{ci_low:.2f}, {ci_high:.2f}]"
    )

    metrics_text = (
        f"Return anual (aprox): {ann_return:.2%}\n"
        f"Volatilidad anual (aprox): {ann_vol:.2%}\n"
        f"Sharpe (rf≈0): {sharpe:.2f}\n"
        f"MC 95% intervalo: [{ci_low:.2f}, {ci_high:.2f}]"
    )

    fig_price = px.line(prices.reset_index(), x="date", y="price", labels={"price":"USD"}, title=f"{symbol} - Precio")
    fig_price.update_layout(margin=dict(l=30,r=20,t=40,b=30), height=360)
    equity = (1 + prices["return"]).cumprod() * float(capital)
    equity_df = prices.reset_index().copy()
    equity_df["equity"] = equity.values
    fig_eq = px.line(equity_df, x="date", y="equity", title="Equity (Buy & Hold)")
    fig_eq.update_layout(margin=dict(l=30,r=20,t=30,b=30), height=360)
    hist_df = pd.DataFrame({"final": sims})
    fig_hist = px.histogram(hist_df, x="final", nbins=50, title="Distribución final (MC bootstrap)")
    fig_hist.update_layout(margin=dict(l=30,r=20,t=30,b=30), height=320)

    return resumen, metrics_text, fig_price, fig_eq, fig_hist


# --- UI Gradio ---
with gr.Blocks(title="CrypTrader - TradingView + CCXT + Fallback", css=CUSTOM_CSS) as demo:
    gr.Markdown("## CrypTrader — Predicciones Crypto (TradingView + CCXT)")
    with gr.Row():
        with gr.Column(scale=1):
            inp_symbol = gr.Textbox(label="Símbolo (ej: BINANCE:BTCUSDT o BTC/USDT)", value="BINANCE:BTCUSDT")
            inp_start = gr.Textbox(label="Fecha inicio (YYYY-MM-DD)", value="2022-01-01")
            inp_end = gr.Textbox(label="Fecha fin (YYYY-MM-DD) — dejar vacío = hoy", value=str(datetime.utcnow().date()))
            inp_capital = gr.Number(label="Capital (USD)", value=2000.0)
            inp_sim = gr.Slider(label="Simulaciones Monte Carlo", minimum=100, maximum=5000, value=1000, step=100)
            inp_timeframe = gr.Dropdown(label="Timeframe", choices=["1d","4h","1h"], value="1d")
            chk_confirm = gr.Checkbox(label="Confirmo modo AGRESIVO (solo si aplica)", value=False)
            btn = gr.Button("Ejecutar")
            gr.Markdown("Si tienes snippet HTML de TradingView, pégalo en el secret TRADINGVIEW_WIDGET_HTML con {SYMBOL} como placeholder.")
        with gr.Column(scale=2):
            out_summary = gr.Textbox(label="Resumen", interactive=False)
            out_metrics = gr.Textbox(label="Métricas", interactive=False)
            tv_html = tradingview_widget_html("BINANCE:BTCUSDT", width="100%", height=420)
            out_tv = gr.HTML(value=tv_html)
            out_price = gr.Plot(label="Precio")
            out_equity = gr.Plot(label="Equity")
            out_hist = gr.Plot(label="Distribución MC")

    def on_execute(symbol, start_date, end_date, capital, num_sim, timeframe, confirm):
        summary, metrics, fig_price, fig_eq, fig_hist = run_predict(symbol, start_date, end_date, capital, num_sim, timeframe)
        tv_sym = symbol
        if "/" in symbol and ":" not in symbol:
            base = symbol.split("/")[0].upper()
            quote = symbol.split("/")[1].upper()
            tv_sym = f"BINANCE:{base}{quote}"
        new_tv = tradingview_widget_html(tv_sym, width="100%", height=420)
        return summary, metrics, new_tv, fig_price, fig_eq, fig_hist

    btn.click(fn=on_execute, inputs=[inp_symbol, inp_start, inp_end, inp_capital, inp_sim, inp_timeframe, chk_confirm],
              outputs=[out_summary, out_metrics, out_tv, out_price, out_equity, out_hist])

    demo.load(fn=lambda s,a,b,c,d,tf,conf: on_execute(s,a,b,c,d,tf,conf),
              inputs=[inp_symbol, inp_start, inp_end, inp_capital, inp_sim, inp_timeframe, chk_confirm],
              outputs=[out_summary, out_metrics, out_tv, out_price, out_equity, out_hist])


def export_results(symbol, start_date, end_date, capital, num_sim, timeframe):
    resumen, metrics_text, fig_price, fig_eq, fig_hist = run_predict(symbol, start_date, end_date, capital, num_sim, timeframe)
    out = {"meta": {"symbol": symbol, "start": start_date, "end": end_date, "timestamp": str(datetime.utcnow())},
           "summary": resumen, "metrics_text": metrics_text}
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(json.dumps(out, indent=2).encode("utf-8"))
    tmp.close()
    return tmp.name


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False)
