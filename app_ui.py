# app_ui.py
import gradio as gr
from backend import run_analysis, export_results_file
import pandas as pd

css = """
body { background-color: #0b0f14; color: #e6eef6; }
.gradio-container { max-width: 1400px; margin: auto; }
"""

def launch_ui():
    with gr.Blocks(css=css, title="CrypTrader — Predicciones Crypto (TradingView + CCXT)") as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                gr.Markdown("## Inputs")
                symbol = gr.Textbox(label="Símbolo (ej: BINANCE:BTCUSDT o BTC/USDT)", value="BINANCE:BTCUSDT")
                start = gr.Textbox(label="Fecha inicio (YYYY-MM-DD)", value="2022-01-01")
                end = gr.Textbox(label="Fecha fin (YYYY-MM-DD) — dejar vacío = hoy", value="")
                timeframe = gr.Dropdown(["1m","5m","15m","1h","4h","1d"], value="1d", label="Timeframe")
                capital = gr.Number(value=2000, label="Capital (USD)")
                mc_sims = gr.Slider(100, 5000, value=1000, step=100, label="Simulaciones Monte Carlo")
                strategy = gr.Dropdown(["buy_hold","ma_cross","rsi_meanrev","ml_prob"], value="ma_cross", label="Strategy preset")
                risk = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Risk % por trade")
                leverage = gr.Number(value=1, label="Leverage")
                aggressive = gr.Checkbox(label="Confirmo modo AGRESIVO (solo si aplica)")
                run_btn = gr.Button("Ejecutar", variant="primary")
                save_btn = gr.Button("Exportar JSON")
                gr.Markdown("**Secrets:** si tienes `TRADINGVIEW_WIDGET_HTML`, pégalo en HF Secrets para mostrar chart TV.")
            with gr.Column(scale=2):
                gr.Markdown("## Resultados")
                out_summary = gr.Textbox(label="Resumen", lines=6)
                out_metrics = gr.Textbox(label="Métricas", lines=6)
                out_price = gr.Plot(label="Precio (Plotly)")
                out_equity = gr.Plot(label="Equity (Plotly)")
                out_mc = gr.Plot(label="Distribución MC")
                out_trades = gr.Dataframe(headers=["date","side","price","size","pnl"], label="Trades")
                out_news = gr.Textbox(label="Noticias & Sentimiento (top)", lines=8)

        # Actions
        def on_run(symbol, start, end, timeframe, capital, mc_sims, strategy, risk, leverage, aggressive):
            res = run_analysis(symbol, start, end, timeframe, capital, int(mc_sims), strategy, float(risk), float(leverage), bool(aggressive))
            # res expected dict: summary,str metrics, figs, dataframe, news_list
            summary = res.get("summary","")
            metrics = res.get("metrics_text","")
            price_fig = res.get("price_fig", {})
            equity_fig = res.get("equity_fig", {})
            mc_fig = res.get("mc_fig", {})
            trades_df = res.get("trades_df", pd.DataFrame())
            news_text = "\n\n".join(res.get("news_list", []))
            return summary, metrics, price_fig, equity_fig, mc_fig, trades_df, news_text

        def on_save(symbol, start, end, timeframe, capital, mc_sims, strategy, risk, leverage, aggressive):
            res = run_analysis(symbol, start, end, timeframe, capital, int(mc_sims), strategy, float(risk), float(leverage), bool(aggressive))
            path = export_results_file(res)
            return path

        run_btn.click(on_run, inputs=[symbol,start,end,timeframe,capital,mc_sims,strategy,risk,leverage,aggressive],
                      outputs=[out_summary,out_metrics,out_price,out_equity,out_mc,out_trades,out_news])
        save_btn.click(on_save, inputs=[symbol,start,end,timeframe,capital,mc_sims,strategy,risk,leverage,aggressive],
                       outputs=gr.File(label="Descargar resultados JSON"))

    demo.launch(server_name="0.0.0.0", server_port=7860)
    
if __name__ == "__main__":
    launch_ui()
# To run: python app_ui.py