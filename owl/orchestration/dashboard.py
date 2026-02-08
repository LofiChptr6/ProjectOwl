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

from owl.config import DASHBOARD_PORT, DASHBOARD_UPDATE_INTERVAL_MS
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
        # ── Tab 2: Throughput & bottleneck ────────────────────────────────
        dbc.Tab(label="Pipeline Throughput", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="throughput-graph"), md=6),
                dbc.Col(dcc.Graph(id="bottleneck-graph"), md=6),
            ]),
        ]),
        # ── Tab 3: Raw metrics table ─────────────────────────────────────
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
    return dbc.Table.from_dataframe(df, striped=True, bordered=True,
                                     hover=True, dark=True, size="sm")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

def run_dashboard(port: int = DASHBOARD_PORT, debug: bool = False):
    logger.info("Starting dashboard on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dashboard(debug=True)
