"""
Recursive Trust Measurement for DWTA-FL (Section 3.4, Equations 6-7).

Computes per-round agreement score and recursive trust:
  - Eq. 6: Φ_i^t = exp( - ||g_i^t - ḡ^t||_2^2 / (σ_g^2 + ε) )
  - Eq. 7: T_i^t = β * T_i^{t-1} + (1 - β) * Φ_i^t, with T_i^0 = 1
"""

import numpy as np
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_agreement_score(
    update_direction: np.ndarray,
    mean_direction: np.ndarray,
    variance_g: float,
    epsilon: float = config.TRUST_EPSILON,
) -> float:
    """
    Compute per-round agreement score Φ_i^t (Eq. 6).

    Φ_i^t = exp( - ||g_i^t - ḡ^t||_2^2 / (σ_g^2 + ε) )

    Args:
        update_direction: Client update direction g_i^t
        mean_direction: Mean update direction ḡ^t across participating clients
        variance_g: Mean squared deviation σ_g^2 across clients
        epsilon: Numerical stability constant (1e-8)

    Returns:
        Agreement score Φ_i^t in (0, 1]
    """
    diff = update_direction - mean_direction
    sq_dist = float(np.sum(diff ** 2))
    score = np.exp(-sq_dist / (variance_g + epsilon))
    return float(np.clip(score, 0.0, 1.0))


def compute_recursive_trust(
    prev_trust: float,
    current_agreement: float,
    beta: float = config.TRUST_MEMORY_BETA,
) -> float:
    """
    Compute recursive trust score T_i^t (Eq. 7).

    T_i^t = β * T_i^{t-1} + (1 - β) * Φ_i^t

    Args:
        prev_trust: T_i^{t-1} (initial T_i^0 = 1)
        current_agreement: Φ_i^t
        beta: Trust memory parameter (β = 0.8)

    Returns:
        Updated trust score T_i^t in [0, 1]
    """
    updated_trust = beta * prev_trust + (1.0 - beta) * current_agreement
    return float(np.clip(updated_trust, 0.0, 1.0))


class TrustTracker:
    """
    Tracks recursive trust for all clients across communication rounds.
    """

    def __init__(
        self,
        num_clients: int = config.NUM_CLIENTS,
        beta: float = config.TRUST_MEMORY_BETA,
        epsilon: float = config.TRUST_EPSILON,
    ):
        self.num_clients = num_clients
        self.beta = beta
        self.epsilon = epsilon

        # State: T_i^0 = 1 for all clients
        self.trust_scores: Dict[int, float] = {i: 1.0 for i in range(num_clients)}
        self.agreement_scores: Dict[int, float] = {i: 1.0 for i in range(num_clients)}

    def update_round(
        self,
        update_directions: Dict[int, np.ndarray],
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute trust scores for all participating clients in current round.

        Args:
            update_directions: Dict mapping client_id to flattened g_i^t

        Returns:
            Dict mapping client_id to {"agreement": Φ_i^t, "trust": T_i^t}
        """
        client_ids = list(update_directions.keys())
        if not client_ids:
            return {}

        stacked = np.stack([update_directions[cid] for cid in client_ids], axis=0)
        mean_dir = np.mean(stacked, axis=0)

        # σ_g^2: mean squared deviation from mean_dir across participating clients
        sq_devs = np.sum((stacked - mean_dir) ** 2, axis=1)
        var_g = float(np.mean(sq_devs))

        results = {}
        for idx, cid in enumerate(client_ids):
            phi = compute_agreement_score(
                stacked[idx], mean_dir, var_g, self.epsilon
            )
            prev_t = self.trust_scores.get(cid, 1.0)
            t_curr = compute_recursive_trust(prev_t, phi, self.beta)

            self.agreement_scores[cid] = phi
            self.trust_scores[cid] = t_curr

            results[cid] = {
                "agreement": phi,
                "trust": t_curr,
            }

        return results

    def get_trust(self, client_id: int) -> float:
        """Get current trust score T_i^t for client."""
        return self.trust_scores.get(client_id, 1.0)
