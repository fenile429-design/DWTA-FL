"""
Krum Baseline (Blanchard et al., NeurIPS 2017).

Selects the client update that minimizes the sum of squared distances to its n - f - 2 closest neighbors:
  - Assumed Byzantine clients: f = 4 (out of 20)
"""

import numpy as np
from typing import Dict, List, Any, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mlp import (
    create_mlp,
    get_model_weights,
    set_model_weights,
    unflatten_weights,
)


def krum_select_index(
    updates: np.ndarray,
    f: int = 4,
) -> int:
    """
    Select the index of the update that minimizes neighbor distances.

    Args:
        updates: Array of shape (N, D)
        f: Number of assumed Byzantine workers

    Returns:
        Index of the selected client update
    """
    n = len(updates)
    # Number of nearest neighbors to sum: n - f - 2
    num_neighbors = max(1, n - f - 2)

    # Compute pairwise squared Euclidean distances
    sq_dists = np.sum((updates[:, None, :] - updates[None, :, :]) ** 2, axis=-1)

    scores = []
    for i in range(n):
        # Sort distances to all other clients (excluding self at distance 0)
        dists = np.sort(sq_dists[i])
        # Closest neighbors excluding self
        score = np.sum(dists[1:num_neighbors + 1])
        scores.append(score)

    return int(np.argmin(scores))


class KrumServer:
    """
    Server for Krum Byzantine-robust aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        f: int = 4,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.f = f
        self.learning_rate = learning_rate
        self.seed = seed

        self.global_model = create_mlp(
            input_dim=input_dim,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.weight_shapes = [w.shape for w in self.global_model.get_weights()]
        self.history: List[Dict[str, Any]] = []

    def get_global_weights(self) -> List[np.ndarray]:
        return get_model_weights(self.global_model)

    def set_global_weights(self, weights: List[np.ndarray]):
        set_model_weights(self.global_model, weights)

    def aggregate_round(
        self,
        round_num: int,
        client_updates: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate round using Krum selection.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        updates = np.stack([client_updates[cid]["delta_flat"] for cid in participating], axis=0)
        best_idx = krum_select_index(updates, f=self.f)
        selected_cid = participating[best_idx]
        selected_delta_flat = updates[best_idx]

        agg_delta = unflatten_weights(selected_delta_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(selected_delta_flat))
        record = {
            "round": round_num,
            "selected_client": selected_cid,
            "norm_diff": norm_diff,
        }
        self.history.append(record)
        return record

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        loss, acc = self.global_model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.global_model.predict(x, verbose=0)
