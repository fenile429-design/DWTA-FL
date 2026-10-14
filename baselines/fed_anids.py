"""
Fed-ANIDS Baseline (Table 8).

Anomaly-based federated intrusion detection:
  - Anomaly threshold: 0.75
  - Aggregates updates while downweighting or excluding updates whose anomaly score > 0.75
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


class FedANIDSServer:
    """
    Server for Fed-ANIDS anomaly-score based aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        anomaly_threshold: float = 0.75,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.anomaly_threshold = anomaly_threshold
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
        Aggregate round filtering updates based on anomaly scores.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        updates_mat = np.stack([client_updates[cid]["delta_flat"] for cid in participating], axis=0)
        median_update = np.median(updates_mat, axis=0)

        # Compute normalized anomaly scores (distance to median)
        dists = np.linalg.norm(updates_mat - median_update, axis=1)
        max_dist = np.max(dists) if np.max(dists) > 0 else 1.0
        norm_scores = dists / max_dist

        # Retain clients with anomaly score <= threshold
        valid_indices = [i for i, s in enumerate(norm_scores) if s <= self.anomaly_threshold]
        if not valid_indices:
            valid_indices = list(range(len(participating)))

        accepted_cids = [participating[i] for i in valid_indices]
        agg_delta_flat = np.mean(updates_mat[valid_indices], axis=0)

        agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(agg_delta_flat))
        record = {
            "round": round_num,
            "accepted_clients": accepted_cids,
            "norm_diff": norm_diff,
        }
        self.history.append(record)
        return record

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        loss, acc = self.global_model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.global_model.predict(x, verbose=0)
