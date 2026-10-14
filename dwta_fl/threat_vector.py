"""
Threat State Vector (TSV) Computation for DWTA-FL (Section 3.8, Equations 18-20).

The security layer of each cloud domain constructs the TSV from the global model's
predictions on that domain's current monitoring window W_i^t of flows.

  - Eq. 18:
      P_al,i^t = (1 / |W_i^t|) * ∑_{x ∈ W_i^t} 1[p̂(x) ≥ 0.5]  (alert share)
      S_i^t    = (∑_{x ∈ W_i^t} p̂(x) * 1[p̂(x) ≥ 0.5]) / max(1, ∑ 1[p̂(x) ≥ 0.5])  (severity)
  - Eq. 19:
      PR_i^t   = (1 / |N(i)|) * ∑_{k ∈ N(i)} P_al,k^t  (propagation risk)
  - Eq. 20:
      TSV_i^t  = [P_al,i^t, S_i^t, PR_i^t, D̂_i^t, T_i^t]^T
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_domain_alerts_and_severity(
    malicious_probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute alert share P_al,i^t and severity S_i^t (Eq. 18).

    Args:
        malicious_probabilities: 1D array of p̂(x) for flows x in window W_i^t
        threshold: Classification threshold (0.5)

    Returns:
        (alert_share, severity)
    """
    total_flows = len(malicious_probabilities)
    if total_flows == 0:
        return 0.0, 0.0

    flagged = malicious_probabilities >= threshold
    flagged_count = int(np.sum(flagged))
    alert_share = float(flagged_count / total_flows)

    if flagged_count == 0:
        severity = 0.0
    else:
        flagged_probs = malicious_probabilities[flagged]
        severity = float(np.sum(flagged_probs) / flagged_count)

    return float(np.clip(alert_share, 0.0, 1.0)), float(np.clip(severity, 0.0, 1.0))


Tuple_Alerts = tuple  # Python type hint helper


def compute_propagation_risk(
    domain_id: int,
    alert_shares: Dict[int, float],
    topology: Dict[int, List[int]],
) -> float:
    """
    Compute propagation risk PR_i^t (Eq. 19).

    PR_i^t = (1 / |N(i)|) * ∑_{k ∈ N(i)} P_al,k^t

    Args:
        domain_id: Cloud domain index i
        alert_shares: Dict mapping domain k to P_al,k^t
        topology: Predefined interaction adjacency {i: [neighbors]}

    Returns:
        Propagation risk PR_i^t in [0, 1]
    """
    neighbors = topology.get(domain_id, [])
    if not neighbors:
        return 0.0

    neighbor_alerts = [alert_shares.get(k, 0.0) for k in neighbors]
    return float(np.clip(np.mean(neighbor_alerts), 0.0, 1.0))


def build_tsv(
    alert_share: float,
    severity: float,
    propagation_risk: float,
    smoothed_drift: float,
    trust: float,
) -> np.ndarray:
    """
    Construct Threat State Vector TSV_i^t (Eq. 20).

    TSV_i^t = [P_al,i^t, S_i^t, PR_i^t, D̂_i^t, T_i^t]^T

    Returns:
        numpy array of shape (5,)
    """
    return np.array([
        float(np.clip(alert_share, 0.0, 1.0)),
        float(np.clip(severity, 0.0, 1.0)),
        float(np.clip(propagation_risk, 0.0, 1.0)),
        float(np.clip(smoothed_drift, 0.0, 1.0)),
        float(np.clip(trust, 0.0, 1.0)),
    ], dtype=np.float64)
