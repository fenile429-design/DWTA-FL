"""
Visualization Utilities for DWTA-FL Experiments.

Generates plots for:
  - Accuracy / F1 curves across rounds
  - Client trust and smoothed drift evolution
  - Drift resilience comparisons (Table 12 / Table S4)
  - Threat State Vector (TSV) profiles and mitigation actions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_convergence_curve(
    history_records: List[Dict],
    title: str = "DWTA-FL Convergence Curve",
    save_path: Optional[str] = None,
):
    """Plot ||w_{t+1} - w_t||_2 across communication rounds."""
    rounds = [r["round"] for r in history_records]
    norm_diffs = [r["norm_diff"] for r in history_records]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, norm_diffs, marker="o", markersize=3, color="#1f77b4", label="||w_{t+1} - w_t||_2")
    plt.axhline(y=config.STABILITY_EPSILON, color="r", linestyle="--", label=f"Stability ε ({config.STABILITY_EPSILON})")
    plt.xlabel("Communication Round t")
    plt.ylabel("Global Update Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_trust_and_drift_evolution(
    history_records: List[Dict],
    client_ids: List[int],
    save_path: Optional[str] = None,
):
    """Plot trust T_i^t and smoothed drift D̂_i^t over rounds for selected clients."""
    rounds = [r["round"] for r in history_records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trust plot
    for cid in client_ids:
        trust_vals = [r["trust"].get(cid, 1.0) for r in history_records]
        axes[0].plot(rounds, trust_vals, label=f"Client {cid}")
    axes[0].set_xlabel("Communication Round t")
    axes[0].set_ylabel("Recursive Trust Score T_i^t")
    axes[0].set_title("Trust Score Evolution (Eq. 7)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Drift plot
    for cid in client_ids:
        drift_vals = [r["drifts"].get(cid, 0.0) for r in history_records]
        axes[1].plot(rounds, drift_vals, label=f"Client {cid}")
    axes[1].axhline(y=config.DRIFT_THRESHOLD_THETA_D, color="k", linestyle="--", label="Drift θ_D (0.25)")
    axes[1].set_xlabel("Communication Round t")
    axes[1].set_ylabel("Smoothed Drift D̂_i^t")
    axes[1].set_title("Smoothed Drift Evolution (Eq. 5)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
