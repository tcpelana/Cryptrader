# app_ui.py
import gradio as gr
from backend import run_analysis, export_results_file, load_alerts

def on_run(symbol):
    summary, metrics, fig_price, fig_equity, df = run_analysis(symbol)
    alerts_list = load_alerts(5)
    alerts_text = "\n".join([f"{a.get('_received_at')} | {a.get('symbol')} | {a.get('signal')}" for a in alerts_list])
    return summary, str(metrics), fig_price, fig_equity, df.head().to_markdown(), alerts_text

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 CrypTrader")
    symbol_in = gr.Textbox(label="Símbolo", value="BTCUSDT")
    btn = gr.Button("Run Analysis")

    out_summary = gr.Textbox(label="Resumen")
    out_metrics = gr.Textbox(label="Métricas")
    out_price = gr.Plot(label="Precio")
    out_equity = gr.Plot(label="Equity")
    out_df = gr.Textbox(label="DataFrame preview")
    out_alerts = gr.Textbox(label="Últimas alertas")

    btn.click(on_run, inputs=symbol_in, outputs=[out_summary, out_metrics, out_price, out_equity, out_df, out_alerts])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
