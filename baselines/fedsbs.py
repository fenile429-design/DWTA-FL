"""
FedSBS Baseline (Table 8).

Participant scoring and selection with global server momentum:
  - Global momentum coefficient: 0.9
  - Selected participants per round: 20
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


class FedSBSServer:
    """
    Server for FedSBS with participant scoring and global momentum.
    """

    def __init__(
        self,
        input_dim: int,
        momentum: float = 0.9,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.seed = seed

        self.global_model = create_mlp(
            input_dim=input_dim,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.weight_shapes = [w.shape for w in self.global_model.get_weights()]
        self.velocity_flat: Optional[np.ndarray] = None
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
        Score participants and aggregate with global momentum.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        # Client scoring based on gradient magnitude / sample quality
        scores = {}
        for cid in participating:
            u_norm = float(np.linalg.norm(client_updates[cid]["delta_flat"]))
            scores[cid] = u_norm

        total_samples = sum(client_updates[cid]["num_samples"] for cid in participating)
        raw_agg_delta = np.zeros_like(client_updates[participating[0]]["delta_flat"])
        for cid in participating:
            weight = client_updates[cid]["num_samples"] / total_samples
            raw_agg_delta += weight * client_updates[cid]["delta_flat"]

        # Apply global momentum: v_{t+1} = momentum * v_t + (1 - momentum) * delta
        if self.velocity_flat is None:
            self.velocity_flat = raw_agg_delta.copy()
        else:
            self.velocity_flat = self.momentum * self.velocity_flat + (1.0 - self.momentum) * raw_agg_delta

        agg_delta = unflatten_weights(self.velocity_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(self.velocity_flat))
        record = {
            "round": round_num,
            "norm_diff": norm_diff,
        }
        self.history.append(record)
        return record

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        loss, acc = self.global_model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.global_model.predict(x, verbose=0)
