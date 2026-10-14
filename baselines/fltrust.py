"""
FLTrust Baseline (Cao et al., NDSS 2021).

Uses a clean root dataset (1000 samples) at the server to compute a baseline update g_0^t:
  - Eq. 16:
      s_i^t = max(0, <g_i^t, g_0^t> / (||g_i^t||_2 * ||g_0^t||_2))
      g_t^{FT} = ∑_{i ∈ U_t} s_i^t * (||g_0^t|| / ||g_i^t||) * g_i^t / ∑ s_i^t
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
    flatten_weights,
    compute_model_update,
)


class FLTrustServer:
    """
    Server for FLTrust Byzantine-robust aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        root_x: np.ndarray,
        root_y: np.ndarray,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.root_x = root_x
        self.root_y = root_y
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

    def compute_root_update(self) -> np.ndarray:
        """Compute server update direction g_0^t on the clean root dataset."""
        root_model = create_mlp(
            input_dim=self.input_dim,
            learning_rate=self.learning_rate,
            seed=self.seed,
        )
        current_w = self.get_global_weights()
        set_model_weights(root_model, current_w)

        # Train on root dataset for 1 epoch
        root_model.fit(
            self.root_x,
            self.root_y,
            epochs=1,
            batch_size=config.BATCH_SIZE,
            verbose=0,
            shuffle=True,
        )

        root_w = get_model_weights(root_model)
        delta_w = compute_model_update(current_w, root_w)
        return flatten_weights(delta_w)

    def aggregate_round(
        self,
        round_num: int,
        client_updates: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate client updates using FLTrust algorithm (Eq. 16).
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        # 1. Compute server update Δw_0^t
        root_delta = self.compute_root_update()
        root_norm = float(np.linalg.norm(root_delta)) + 1e-12

        # 2. Compute cosine similarity scores s_i^t with ReLU clipping
        scores = {}
        normalised_deltas = {}
        for cid in participating:
            delta_i = client_updates[cid]["delta_flat"]
            client_norm = float(np.linalg.norm(delta_i)) + 1e-12

            cos_sim = float(np.dot(delta_i, root_delta) / (client_norm * root_norm))
            s_i = max(0.0, cos_sim)
            scores[cid] = s_i

            # Rescaled update: (||g_0|| / ||g_i||) * g_i
            normalised_deltas[cid] = (root_norm / client_norm) * delta_i

        sum_scores = sum(scores.values())
        if sum_scores <= 1e-12:
            # Fall back to root update if no client is trusted
            agg_delta_flat = root_delta
            weights = {cid: 0.0 for cid in participating}
        else:
            weights = {cid: scores[cid] / sum_scores for cid in participating}
            agg_delta_flat = np.zeros_like(root_delta)
            for cid in participating:
                if weights[cid] > 0:
                    agg_delta_flat += weights[cid] * normalised_deltas[cid]

        agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(agg_delta_flat))
        record = {
            "round": round_num,
            "scores": scores,
            "weights": weights,
            "norm_diff": norm_diff,
        }
        self.history.append(record)
        return record

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        loss, acc = self.global_model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.global_model.predict(x, verbose=0)
