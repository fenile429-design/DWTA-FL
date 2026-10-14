"""
Multi-Cloud Security Orchestration Experiment (Section 4.8, Table 19).

Evaluates the TSV-driven cost-aware mitigation layer across:
  - 3 simulated CSP domains: Cloud A, Cloud B, Cloud C
  - 5-second decision windows over 30-minute scenarios (360 windows)
  - Predefined interaction topology
  - Adaptive Response Efficiency (ARE, Eq. 30)
  - Threat state transitions (Normal, Localised, Propagating, Coordinated, Recovery)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dwta_fl.threat_vector import (
    compute_domain_alerts_and_severity,
    compute_propagation_risk,
    build_tsv,
)
from dwta_fl.mitigation import MitigationOrchestrator, ACTIONS
from utils.metrics import compute_are


def run_orchestration_simulation(
    num_domains: int = config.NUM_ORCHESTRATION_DOMAINS,
    total_windows: int = 360,  # 30 minutes @ 5s/window
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Simulate multi-cloud orchestration scenario with evolving threat profiles.
    """
    rng = np.random.default_rng(seed)

    # Topology: triangle connecting all 3 domains
    topology = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }

    orchestrators = {
        d_id: MitigationOrchestrator(domain_id=d_id)
        for d_id in range(num_domains)
    }

    # Tracking records
    history_actions = {d: [] for d in range(num_domains)}
    history_tsv = {d: [] for d in range(num_domains)}
    history_severities = []
    history_correctness = []
    history_costs = []

    # Scenario phases (in windows)
    # 0-60: Normal
    # 61-120: Localised threat on Cloud A
    # 121-200: Propagating threat spreading to Cloud B & C
    # 201-280: Coordinated threat across all clouds
    # 281-360: Recovery phase

    for w in range(total_windows):
        # 1. Generate synthetic flow alert probabilities reflecting the scenario
        alert_shares = {}
        severities = {}

        for d_id in range(num_domains):
            if w <= 60:
                # Normal traffic: low alerts (<0.1)
                prob_vals = rng.beta(1, 10, size=100)
                expected_action = "Monitor"
            elif 61 <= w <= 120:
                # Localised on domain 0 (Cloud A)
                if d_id == 0:
                    prob_vals = rng.beta(6, 2, size=100)  # High severity
                    expected_action = "Access Restriction"
                else:
                    prob_vals = rng.beta(1, 10, size=100)
                    expected_action = "Monitor"
            elif 121 <= w <= 200:
                # Propagating threat spreading
                prob_vals = rng.beta(5, 3, size=100)
                expected_action = "Isolation"
            elif 201 <= w <= 280:
                # Coordinated threat
                prob_vals = rng.beta(8, 2, size=100)
                expected_action = "Quarantine"
            else:
                # Recovery phase
                prob_vals = rng.beta(1, 8, size=100)
                expected_action = "Monitor"

            al_share, sev = compute_domain_alerts_and_severity(prob_vals)
            alert_shares[d_id] = al_share
            severities[d_id] = sev

        # 2. Build TSV and select mitigation for each domain
        for d_id in range(num_domains):
            pr = compute_propagation_risk(d_id, alert_shares, topology)
            # Simulated drift and trust
            drift_val = 0.05 if w <= 60 or w > 280 else 0.35
            trust_val = 0.95 if w <= 60 or w > 280 else 0.55

            tsv = build_tsv(
                alert_share=alert_shares[d_id],
                severity=severities[d_id],
                propagation_risk=pr,
                smoothed_drift=drift_val,
                trust=trust_val,
            )
            history_tsv[d_id].append(tsv)

            action, _ = orchestrators[d_id].select_action(
                severity=severities[d_id],
                propagation_risk=pr,
                trust=trust_val,
            )
            history_actions[d_id].append(action)

            # Update weights adaptively (Eq. 23)
            orchestrators[d_id].update_weights(
                severity=severities[d_id],
                propagation_risk=pr,
                trust=trust_val,
            )

            # Record event metrics for ARE
            cost = config.ACTION_COEFFICIENTS[action]["cost"]
            is_correct = (action == expected_action)

            history_severities.append(severities[d_id])
            history_correctness.append(is_correct)
            history_costs.append(cost)

    are = compute_are(history_severities, history_correctness, history_costs)

    return {
        "are": are,
        "history_actions": history_actions,
        "history_tsv": history_tsv,
        "total_windows": total_windows,
    }


def run_orchestration_benchmark(num_runs: int = 10) -> pd.DataFrame:
    """Run orchestration benchmark over 10 independent scenario runs."""
    are_scores = []
    for r in range(num_runs):
        res = run_orchestration_simulation(seed=100 + r)
        are_scores.append(res["are"])

    mean_are = np.mean(are_scores)
    std_are = np.std(are_scores, ddof=1)

    print(f"\n==================== Multi-Cloud Orchestration Benchmark (Table 19) ====================")
    print(f"Adaptive Response Efficiency (ARE): {mean_are:.2f} +/- {std_are:.2f}")

    df = pd.DataFrame([{
        "Metric": "Adaptive Response Efficiency (ARE)",
        "Mean +/- SD": f"{mean_are:.2f} +/- {std_are:.2f}",
    }])
    return df
