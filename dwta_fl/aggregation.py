"""
Aggregation Weight Computation for DWTA-FL (Section 3.5, Equations 10-11).

- Eq. 10: R_i^t = (T_i^t)^μ * (C_i^t)^ν * exp(-γ * D̂_i^t) * exp(-τ * V_i^t)
  with μ = 1.2, ν = 1.0, γ = 0.5, τ = 0.4
- Eq. 11: α_i^t = (q_i^t * R_i^t) / ∑_{j ∈ U_t} (q_j^t * R_j^t)
"""

import numpy as np
from typing import Dict, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_reliability(
    trust: float,
    consistency: float,
    smoothed_drift: float,
    volatility: float,
    mu: float = config.TRUST_EXPONENT_MU,
    nu: float = config.CONSISTENCY_EXPONENT_NU,
    gamma: float = config.DRIFT_PENALTY_GAMMA,
    tau: float = config.VOLATILITY_PENALTY_TAU,
) -> float:
    """
    Compute client unnormalised reliability R_i^t (Eq. 10).

    R_i^t = (T_i^t)^μ * (C_i^t)^ν * exp(-γ * D̂_i^t) * exp(-τ * V_i^t)

    Args:
        trust: T_i^t in [0, 1]
        consistency: C_i^t in (0, 1]
        smoothed_drift: D̂_i^t in [0, 1]
        volatility: V_i^t ≥ 0
        mu: Trust exponent (1.2)
        nu: Consistency exponent (1.0)
        gamma: Drift penalty coefficient (0.5)
        tau: Volatility penalty coefficient (0.4)

    Returns:
        Reliability R_i^t > 0
    """
    # Guard against negative/zero bases in fractional powers
    t_val = max(1e-12, float(trust))
    c_val = max(1e-12, float(consistency))
    d_val = max(0.0, float(smoothed_drift))
    v_val = max(0.0, float(volatility))

    rel = (t_val ** mu) * (c_val ** nu) * np.exp(-gamma * d_val) * np.exp(-tau * v_val)
    return float(max(1e-12, rel))


def compute_aggregation_weights(
    reliabilities: Dict[int, float],
    gates: Dict[int, float],
) -> Tuple[Dict[int, float], bool]:
    """
    Compute normalised aggregation weights α_i^t (Eq. 11).

    α_i^t = (q_i^t * R_i^t) / ∑_{j ∈ U_t} (q_j^t * R_j^t)

    If denominator is zero, skip_update is True.

    Args:
        reliabilities: Dict mapping client_id to R_i^t
        gates: Dict mapping client_id to q_i^t

    Returns:
        Tuple of (weights_dict, skip_update_flag)
    """
    client_ids = list(reliabilities.keys())
    gated_weights = {}
    denom = 0.0

    for cid in client_ids:
        q = gates.get(cid, 1.0)
        r = reliabilities[cid]
        gw = q * r
        gated_weights[cid] = gw
        denom += gw

    if denom <= 1e-15:
        # All clients gated to zero or negligible: skip update
        return {cid: 0.0 for cid in client_ids}, True

    weights = {cid: gated_weights[cid] / denom for cid in client_ids}
    return weights, False
