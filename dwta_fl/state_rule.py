"""
Drift-versus-Attack State Rule for DWTA-FL (Section 3.6, Equations 12-14).

Separates legitimate concept drift from adversarial attacks:
  - Eq. 12: Γ_i^t = R_i^t / (1 + σ_i^t)  (risk-adjusted reliability)
  - Eq. 13: Four-state classification:
      * Safe:          D̂_i^t ≤ θ_D
      * Benign Drift:  D̂_i^t > θ_D, Γ_i^t ≥ θ_H
      * Suspicious:    D̂_i^t > θ_D, θ_L ≤ Γ_i^t < θ_H
      * Attacker:      D̂_i^t > θ_D, Γ_i^t < θ_L
  - Eq. 14: State gating factor q_i^t:
      * Safe / Benign Drift: 1.0
      * Suspicious:          κ_s (0.5)
      * Attacker:            κ_a (0.1)
"""

from typing import Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# State definitions
STATE_SAFE = "Safe"
STATE_BENIGN_DRIFT = "Benign Drift"
STATE_SUSPICIOUS = "Suspicious"
STATE_ATTACKER = "Attacker"


def compute_risk_adjusted_score(
    reliability: float,
    drift_uncertainty: float,
) -> float:
    """
    Compute risk-adjusted score Γ_i^t (Eq. 12).

    Γ_i^t = R_i^t / (1 + σ_i^t)

    Discounts reliability when drift uncertainty σ is large.
    """
    sigma = max(0.0, float(drift_uncertainty))
    return float(reliability / (1.0 + sigma))


def classify_client_state(
    smoothed_drift: float,
    gamma: float,
    theta_d: float = config.DRIFT_THRESHOLD_THETA_D,
    theta_h: float = config.HIGH_RELIABILITY_THETA_H,
    theta_l: float = config.LOW_RELIABILITY_THETA_L,
) -> str:
    """
    Assign client state Z_i^t according to Eq. 13.

    Args:
        smoothed_drift: D̂_i^t
        gamma: Γ_i^t
        theta_d: Drift threshold (0.25)
        theta_h: High reliability threshold (0.75)
        theta_l: Low reliability threshold (0.40)

    Returns:
        One of: 'Safe', 'Benign Drift', 'Suspicious', 'Attacker'
    """
    if smoothed_drift <= theta_d:
        return STATE_SAFE
    elif gamma >= theta_h:
        return STATE_BENIGN_DRIFT
    elif gamma >= theta_l:
        return STATE_SUSPICIOUS
    else:
        return STATE_ATTACKER


def get_state_gate(
    state: str,
    kappa_s: float = config.GATE_SUSPICIOUS_KAPPA_S,
    kappa_a: float = config.GATE_ATTACKER_KAPPA_A,
) -> float:
    """
    Get gating factor q_i^t according to Eq. 14.

    Args:
        state: Assigned state Z_i^t
        kappa_s: Suspicious gate factor (0.5)
        kappa_a: Attacker gate factor (0.1)

    Returns:
        Gate factor q_i^t in [0, 1]
    """
    if state in (STATE_SAFE, STATE_BENIGN_DRIFT):
        return 1.0
    elif state == STATE_SUSPICIOUS:
        return kappa_s
    elif state == STATE_ATTACKER:
        return kappa_a
    else:
        return 1.0


def evaluate_client_states(
    reliabilities: Dict[int, float],
    drift_uncertainties: Dict[int, float],
    smoothed_drifts: Dict[int, float],
    theta_d: float = config.DRIFT_THRESHOLD_THETA_D,
    theta_h: float = config.HIGH_RELIABILITY_THETA_H,
    theta_l: float = config.LOW_RELIABILITY_THETA_L,
    kappa_s: float = config.GATE_SUSPICIOUS_KAPPA_S,
    kappa_a: float = config.GATE_ATTACKER_KAPPA_A,
) -> Dict[int, Dict[str, object]]:
    """
    Evaluate states and gates for all clients in round.

    Returns:
        Dict mapping client_id to:
            {"gamma": Γ_i^t, "state": Z_i^t, "gate": q_i^t}
    """
    evaluations = {}
    for cid, r in reliabilities.items():
        sigma = drift_uncertainties.get(cid, 0.0)
        d_hat = smoothed_drifts.get(cid, 0.0)

        gamma = compute_risk_adjusted_score(r, sigma)
        state = classify_client_state(d_hat, gamma, theta_d, theta_h, theta_l)
        gate = get_state_gate(state, kappa_s, kappa_a)

        evaluations[cid] = {
            "gamma": gamma,
            "state": state,
            "gate": gate,
        }

    return evaluations
