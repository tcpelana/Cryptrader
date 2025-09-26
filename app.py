# app.py
import os
import gradio as gr
from backend import read_alerts, simulate_strategy_with_costs, read_alerts, ensure_no_future_features
from backend import run_analysis  # tu función plena que devuelve resultados

def run_ui(symbol, start, end, timeframe, strategy):
    # call backend.run_analysis (assume implemented)
    res = run_analysis(symbol=symbol)
    # res must include summary, metrics_text, figs etc.
    summary = res.get("summary","")
    metrics = res.get("metrics_text", str(res.get("metrics", {})))
    price_fig = res.get("price_fig")
    equity_fig = res.get("equity_fig")
    trades_df = res.get("trades_df")
    alerts = read_alerts(5)
    alerts_text = "\n".join([f"{a.get('_received_at')} | {a.get('symbol')} | {a.get('signal')}" for a in alerts])
    return summary, metrics, price_fig, equity_fig, trades_df, alerts_text

with gr.Blocks() as demo:
    gr.Markdown("# CrypTrader — Panel")
    with gr.Row():
        with gr.Column(scale=1):
            symbol = gr.Textbox(label="Símbolo", value="BINANCE:BTCUSDT")
            start = gr.Textbox(label="Start (YYYY-MM-DD)", value="")
            end = gr.Textbox(label="End (YYYY-MM-DD)", value="")
            timeframe = gr.Dropdown(["1d","4h","1h","15m"], value="1d")
            strategy = gr.Dropdown(["ma_cross","rsi_meanrev","ml_prob"], value="ma_cross")
            run_btn = gr.Button("Ejecutar")
        with gr.Column(scale=2):
            out_summary = gr.Textbox(label="Resumen", lines=6)
            out_metrics = gr.Textbox(label="Métricas", lines=6)
            out_price = gr.Plot(label="Precio")
            out_equity = gr.Plot(label="Equity")
            out_trades = gr.Dataframe(headers=["date","side","price","size","fee","pnl"])
            out_alerts = gr.Textbox(label="Últimas alertas", lines=6)

    run_btn.click(fn=run_ui, inputs=[symbol,start,end,timeframe,strategy], outputs=[out_summary,out_metrics,out_price,out_equity,out_trades,out_alerts])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
