"""
Consistency and Volatility Tracking for DWTA-FL (Section 3.5, Equations 8-9).

- Eq. 8: C_i^t = exp( - ||g_i^t - g_i^{t-1}||_2^2 / (σ_c^2 + ε) )
  where σ_c^2 is the mean of ||g_j^t - g_j^{t-1}||_2^2 over participating clients.
- Eq. 9: V_i^t = |C_i^t - C_i^{t-1}| + |D̂_i^t - D̂_i^{t-1}| ≥ 0
"""

import numpy as np
from typing import Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_consistency(
    current_direction: np.ndarray,
    prev_direction: np.ndarray,
    variance_c: float,
    epsilon: float = config.TRUST_EPSILON,
) -> float:
    """
    Compute temporal consistency C_i^t (Eq. 8).

    C_i^t = exp( - ||g_i^t - g_i^{t-1}||_2^2 / (σ_c^2 + ε) )

    Args:
        current_direction: g_i^t
        prev_direction: g_i^{t-1}
        variance_c: σ_c^2 (mean squared difference over participating clients)
        epsilon: Numerical stability constant (1e-8)

    Returns:
        Consistency score C_i^t in (0, 1]
    """
    diff = current_direction - prev_direction
    sq_dist = float(np.sum(diff ** 2))
    score = np.exp(-sq_dist / (variance_c + epsilon))
    return float(np.clip(score, 0.0, 1.0))


def compute_volatility(
    curr_consistency: float,
    prev_consistency: float,
    curr_smoothed_drift: float,
    prev_smoothed_drift: float,
) -> float:
    """
    Compute short-term volatility V_i^t (Eq. 9).

    V_i^t = |C_i^t - C_i^{t-1}| + |D̂_i^t - D̂_i^{t-1}| ≥ 0

    Args:
        curr_consistency: C_i^t
        prev_consistency: C_i^{t-1}
        curr_smoothed_drift: D̂_i^t
        prev_smoothed_drift: D̂_i^{t-1}

    Returns:
        Volatility V_i^t in [0, 2]
    """
    delta_c = abs(curr_consistency - prev_consistency)
    delta_d = abs(curr_smoothed_drift - prev_smoothed_drift)
    return float(delta_c + delta_d)


class ConsistencyVolatilityTracker:
    """
    Tracks temporal consistency and short-term volatility across rounds.
    """

    def __init__(
        self,
        num_clients: int = config.NUM_CLIENTS,
        epsilon: float = config.TRUST_EPSILON,
    ):
        self.num_clients = num_clients
        self.epsilon = epsilon

        # State storage
        self.prev_directions: Dict[int, np.ndarray] = {}
        self.consistencies: Dict[int, float] = {i: 1.0 for i in range(num_clients)}
        self.prev_consistencies: Dict[int, float] = {i: 1.0 for i in range(num_clients)}
        self.volatilities: Dict[int, float] = {i: 0.0 for i in range(num_clients)}

    def update_round(
        self,
        update_directions: Dict[int, np.ndarray],
        curr_smoothed_drifts: Dict[int, float],
        prev_smoothed_drifts: Dict[int, float],
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute consistency and volatility for participating clients.

        Args:
            update_directions: Dict mapping client_id to g_i^t
            curr_smoothed_drifts: Dict mapping client_id to D̂_i^t
            prev_smoothed_drifts: Dict mapping client_id to D̂_i^{t-1}

        Returns:
            Dict mapping client_id to {"consistency": C_i^t, "volatility": V_i^t}
        """
        client_ids = list(update_directions.keys())
        results = {}

        # Identify clients that have a previous update
        has_prev = [cid for cid in client_ids if cid in self.prev_directions]

        if has_prev:
            diffs = [
                update_directions[cid] - self.prev_directions[cid]
                for cid in has_prev
            ]
            sq_diffs = [float(np.sum(d ** 2)) for d in diffs]
            var_c = float(np.mean(sq_diffs))
        else:
            var_c = 1.0

        for cid in client_ids:
            curr_dir = update_directions[cid]
            if cid not in self.prev_directions:
                # First round for this client: perfect consistency, zero volatility
                c_curr = 1.0
                v_curr = 0.0
            else:
                prev_dir = self.prev_directions[cid]
                c_curr = compute_consistency(curr_dir, prev_dir, var_c, self.epsilon)
                c_prev = self.consistencies.get(cid, 1.0)
                d_curr = curr_smoothed_drifts.get(cid, 0.0)
                d_prev = prev_smoothed_drifts.get(cid, 0.0)
                v_curr = compute_volatility(c_curr, c_prev, d_curr, d_prev)

            # Update state
            self.prev_directions[cid] = curr_dir.copy()
            self.prev_consistencies[cid] = self.consistencies.get(cid, 1.0)
            self.consistencies[cid] = c_curr
            self.volatilities[cid] = v_curr

            results[cid] = {
                "consistency": c_curr,
                "volatility": v_curr,
            }

        return results

    def get_consistency(self, client_id: int) -> float:
        """Get current consistency C_i^t for client."""
        return self.consistencies.get(client_id, 1.0)

    def get_volatility(self, client_id: int) -> float:
        """Get current volatility V_i^t for client."""
        return self.volatilities.get(client_id, 0.0)
