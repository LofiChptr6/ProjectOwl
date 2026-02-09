"""
Orchestration Dashboard  (Dash / Plotly)
=========================================
A lightweight web UI that polls ``training_metrics`` in PostgreSQL and
renders live charts for:

1. **Data pipeline throughput** — rows pulled / sec, preprocess rate
2. **Training loss** — per-batch and per-epoch curves
3. **Validation metrics** — loss & accuracy after each epoch
4. **Bottleneck indicator** — bar chart of time-per-phase

Launch with::

    python -m owl.orchestration.dashboard

Then open http://localhost:8050 in a browser.
"""

from __future__ import annotations

import logging

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from pathlib import Path

from owl.config import (
    DASHBOARD_PORT,
    DASHBOARD_UPDATE_INTERVAL_MS,
    REPORT_DIR,
)
from owl.data.db import get_engine

logger = logging.getLogger(__name__)


def _query(sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, get_engine())
    except Exception as exc:
        logger.debug("Dashboard query failed: %s", exc)
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════
#  App layout
# ══════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="ProjectOwl — Dashboard",
)

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row(dbc.Col(html.H2("ProjectOwl Training Dashboard",
                             className="text-center my-3"))),

    # auto-refresh
    dcc.Interval(id="tick", interval=DASHBOARD_UPDATE_INTERVAL_MS),

    dbc.Tabs([
        # ── Tab 1: Training loss ──────────────────────────────────────────
        dbc.Tab(label="Training Loss", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="loss-graph"), md=8),
                dbc.Col(dcc.Graph(id="acc-graph"), md=4),
            ]),
        ]),
        # ── Tab 2: Worker load vs model ingestion ────────────────────────
        dbc.Tab(label="Worker vs Ingestion", children=[
            dbc.Row(dbc.Col(html.P(
                "Data wait = time waiting for DataLoader workers; Compute = GPU time. "
                "If data wait dominates, increase num_workers or cache.",
                className="text-muted small"))),
            dbc.Row([
                dbc.Col(dcc.Graph(id="worker-vs-ingestion-graph"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="throughput-graph"), md=6),
                dbc.Col(dcc.Graph(id="bottleneck-graph"), md=6),
            ]),
        ]),
        # ── Tab 3: Model architecture ────────────────────────────────────
        dbc.Tab(label="Model Architecture", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Model:", className="me-2"),
                    dcc.Dropdown(
                        id="model-arch-dropdown",
                        options=[{"label": "CNN", "value": "cnn"}, {"label": "Transformer", "value": "transformer"}],
                        value="cnn",
                        clearable=False,
                        style={"width": "120px"},
                    ),
                ], width=2),
            ], className="mb-2"),
            html.Div(id="model-arch-content", className="p-3"),
        ]),
        # ── Tab 4: Raw metrics table ─────────────────────────────────────
        dbc.Tab(label="Raw Metrics", children=[
            html.Div(id="metrics-table", className="p-3"),
        ]),
    ]),
])


# ══════════════════════════════════════════════════════════════════════════
#  Callbacks
# ══════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("loss-graph", "figure"),
    Output("acc-graph", "figure"),
    Input("tick", "n_intervals"),
)
def update_training_charts(_):
    # ── loss per batch ────────────────────────────────────────────────
    df = _query("""
        SELECT epoch, batch, metric_value
        FROM training_metrics
        WHERE metric_name = 'train_loss'
        ORDER BY id DESC LIMIT 5000
    """)
    if df.empty:
        loss_fig = go.Figure().update_layout(
            title="Train Loss (no data yet)",
            template="plotly_dark",
        )
    else:
        df = df.sort_values(["epoch", "batch"])
        df["step"] = range(len(df))
        loss_fig = px.line(df, x="step", y="metric_value",
                           title="Train Loss (per batch)",
                           template="plotly_dark")
        loss_fig.update_layout(yaxis_title="Loss", xaxis_title="Batch step")

    # ── validation accuracy per epoch ─────────────────────────────────
    df_acc = _query("""
        SELECT epoch, metric_value
        FROM training_metrics
        WHERE metric_name = 'val_acc'
        ORDER BY epoch
    """)
    if df_acc.empty:
        acc_fig = go.Figure().update_layout(
            title="Val Accuracy (no data yet)",
            template="plotly_dark",
        )
    else:
        acc_fig = px.line(df_acc, x="epoch", y="metric_value",
                          title="Validation Accuracy",
                          template="plotly_dark")
        acc_fig.update_layout(yaxis_title="Accuracy", xaxis_title="Epoch")

    return loss_fig, acc_fig


@app.callback(
    Output("worker-vs-ingestion-graph", "figure"),
    Input("tick", "n_intervals"),
)
def update_worker_vs_ingestion(_):
    """Real-time: time waiting for data workers vs GPU compute time per batch."""
    df = _query("""
        SELECT id, timestamp, epoch, batch, metric_name, metric_value
        FROM training_metrics
        WHERE metric_name IN ('data_wait_sec', 'compute_sec')
        ORDER BY id DESC LIMIT 1500
    """)
    if df.empty:
        fig = go.Figure().update_layout(
            title="Worker load vs model ingestion (no data yet)",
            template="plotly_dark",
        )
        return fig
    df = df.sort_values("id")
    # One line per metric
    fig = go.Figure(layout=dict(title="Worker load vs model ingestion", template="plotly_dark"))
    for name, label in [("data_wait_sec", "Data wait (workers)"), ("compute_sec", "GPU compute")]:
        sub = df[df["metric_name"] == name]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["id"], y=sub["metric_value"],
                name=label, mode="lines",
            ))
    fig.update_layout(
        xaxis_title="Batch (metric id)",
        yaxis_title="Seconds",
        legend=dict(orientation="h"),
    )
    return fig


@app.callback(
    Output("throughput-graph", "figure"),
    Output("bottleneck-graph", "figure"),
    Input("tick", "n_intervals"),
)
def update_throughput_charts(_):
    # ── throughput time-series ────────────────────────────────────────
    df = _query("""
        SELECT timestamp, phase, metric_value
        FROM training_metrics
        WHERE metric_name = 'throughput_rows_per_sec'
        ORDER BY timestamp DESC LIMIT 2000
    """)
    if df.empty:
        tp_fig = go.Figure().update_layout(
            title="Throughput (no data yet)", template="plotly_dark")
    else:
        df = df.sort_values("timestamp")
        tp_fig = px.line(df, x="timestamp", y="metric_value", color="phase",
                         title="Throughput (rows/sec by phase)",
                         template="plotly_dark")

    # ── bottleneck bar chart (avg time per phase) ─────────────────────
    df_bn = _query("""
        SELECT phase, AVG(metric_value) AS avg_throughput
        FROM training_metrics
        WHERE metric_name = 'throughput_rows_per_sec'
          AND metric_value > 0
        GROUP BY phase
    """)
    if df_bn.empty:
        bn_fig = go.Figure().update_layout(
            title="Bottleneck (no data yet)", template="plotly_dark")
    else:
        bn_fig = px.bar(df_bn, x="phase", y="avg_throughput",
                        title="Avg Throughput by Phase (higher = faster)",
                        template="plotly_dark", color="phase")

    return tp_fig, bn_fig


@app.callback(
    Output("model-arch-content", "children"),
    Input("tick", "n_intervals"),
    Input("model-arch-dropdown", "value"),
)
def update_model_arch(_, model_name):
    """Display torchinfo summary for the selected model."""
    if not model_name:
        return html.P("Select a model.", className="text-muted")
    path = Path(REPORT_DIR) / model_name / "model_arch.txt"
    if not path.exists():
        return html.P(
            f"No architecture for {model_name} — run training first.",
            className="text-muted",
        )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return html.Pre(
            text,
            style={
                "fontSize": "11px",
                "overflow": "auto",
                "maxHeight": "70vh",
                "backgroundColor": "#2c3e50",
                "padding": "12px",
                "borderRadius": "4px",
            },
        )
    except Exception as exc:
        logger.debug("Failed to load model arch: %s", exc)
        return html.P("Could not load architecture.", className="text-muted")


@app.callback(
    Output("metrics-table", "children"),
    Input("tick", "n_intervals"),
)
def update_raw_table(_):
    df = _query("""
        SELECT timestamp, model_name, epoch, batch, metric_name, metric_value, phase
        FROM training_metrics
        ORDER BY id DESC LIMIT 50
    """)
    if df.empty:
        return html.P("No metrics yet — start a training run.",
                       className="text-muted")
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_dashboard(port: int = DASHBOARD_PORT, debug: bool = False):
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logger.info("Starting dashboard on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard(debug=True)
