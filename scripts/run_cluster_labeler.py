#!/usr/bin/env python
"""Streamlit GUI to view example curves from each cluster and name them.

Run after train_clusters.py::

    streamlit run scripts/run_cluster_labeler.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.config import CLUSTER_OUTPUT_DIR

CLUSTER_NAMES_FILE = CLUSTER_OUTPUT_DIR / "cluster_names.json"


def load_data():
    p = CLUSTER_OUTPUT_DIR / "cluster_examples.pt"
    if not p.exists():
        return None
    import torch
    return torch.load(p, weights_only=False)


def save_names(names: dict):
    CLUSTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLUSTER_NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)


def load_names() -> dict:
    if CLUSTER_NAMES_FILE.exists():
        with open(CLUSTER_NAMES_FILE) as f:
            return json.load(f)
    return {}


st.set_page_config(page_title="Cluster Labeler", layout="wide")
st.title("Path Cluster Labeler")

data = load_data()
if data is None:
    st.error(f"No cluster data found. Run: `python scripts/train_clusters.py`")
    st.stop()

curves = data["curves"]  # (K, examples_per, T, F)
feature_names = data.get("feature_names", [f"f{i}" for i in range(curves.shape[-1])])
num_clusters = data["num_clusters"]

saved = load_names()

# Feature to plot (dropdown)
feat_idx = st.sidebar.selectbox(
    "Feature to plot",
    options=list(range(len(feature_names))),
    format_func=lambda i: feature_names[i] if i < len(feature_names) else str(i),
    index=min(3, len(feature_names) - 1),
)

names = {}
for c in range(num_clusters):
    default = saved.get(str(c), f"cluster_{c}")
    with st.expander(f"**Cluster {c}** — {default}", expanded=True):
        name = st.text_input(
            f"Name for cluster {c}",
            value=default,
            key=f"name_{c}",
        )
        names[str(c)] = name

        # Plot example curves
        fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, sharey=True)
        axes = axes.flatten()
        for j, ax in enumerate(axes):
            if j < curves.shape[1]:
                ax.plot(curves[c, j, :, feat_idx], linewidth=0.8)
            ax.set_title(f"Example {j + 1}")
            ax.tick_params(labelsize=7)
        fig.suptitle(f"Cluster {c}: {name}", fontsize=12, y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

if st.button("Save labels"):
    save_names(names)
    st.success(f"Saved to {CLUSTER_NAMES_FILE}")
