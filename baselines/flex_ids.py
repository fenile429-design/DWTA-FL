"""
FLEX-IDS Baseline (Table 8).

Explainable secure federated aggregation for edge/cloud IDS:
  - Pruning rate: 0.1 (magnitude-based parameter pruning of bottom 10%)
  - Agreement threshold: 0.6 (filters updates with cosine similarity < 0.6 to mean)
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


def prune_update(update_flat: np.ndarray, pruning_rate: float = 0.1) -> np.ndarray:
    """Prune bottom parameters by magnitude."""
    if pruning_rate <= 0.0:
        return update_flat
    abs_vals = np.abs(update_flat)
    threshold = np.percentile(abs_vals, pruning_rate * 100.0)
    pruned = update_flat.copy()
    pruned[abs_vals < threshold] = 0.0
    return pruned


class FLEXIDSServer:
    """
    Server for FLEX-IDS aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.6,
        pruning_rate: float = 0.1,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.threshold = threshold
        self.pruning_rate = pruning_rate
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
        Aggregate round using pruning and similarity thresholding.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        # 1. Prune each update
        pruned_updates = {}
        for cid in participating:
            pruned_updates[cid] = prune_update(
                client_updates[cid]["delta_flat"], self.pruning_rate
            )

        # 2. Compute reference direction
        mean_update = np.mean(list(pruned_updates.values()), axis=0)
        mean_norm = float(np.linalg.norm(mean_update)) + 1e-12

        # 3. Filter clients by similarity threshold
        accepted_clients = []
        for cid in participating:
            u = pruned_updates[cid]
            u_norm = float(np.linalg.norm(u)) + 1e-12
            sim = float(np.dot(u, mean_update) / (u_norm * mean_norm))
            if sim >= self.threshold:
                accepted_clients.append(cid)

        if not accepted_clients:
            accepted_clients = participating

        # Average accepted pruned updates
        agg_delta_flat = np.mean([pruned_updates[cid] for cid in accepted_clients], axis=0)
        agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(agg_delta_flat))
        record = {
            "round": round_num,
            "accepted_clients": accepted_clients,
            "norm_diff": norm_diff,
        }
        self.history.append(record)
        return record

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        loss, acc = self.global_model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.global_model.predict(x, verbose=0)
