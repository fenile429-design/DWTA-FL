"""
FL-IDS++ Baseline (Table 8).

Personalized federated intrusion detection framework:
  - Supports non-IID client distributions via personalized interpolation
  - Personalization weight: 0.5
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


class FLIDSPPServer:
    """
    Server for FL-IDS++ personalized aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        personalization_weight: float = 0.5,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.personalization_weight = personalization_weight
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
        Aggregate round with personalization weighting.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        # Average updates
        mean_delta_flat = np.mean([client_updates[cid]["delta_flat"] for cid in participating], axis=0)

        # Apply personalization scale factor
        effective_delta_flat = (1.0 - self.personalization_weight * 0.5) * mean_delta_flat

        agg_delta = unflatten_weights(effective_delta_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(effective_delta_flat))
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
