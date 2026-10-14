"""
FL-IDPP Baseline (Table 8).

Privacy-preserving federated intrusion detection system:
  - Norm clipping: clipping norm = 1.0
  - Differential privacy Gaussian perturbation: noise multiplier = 0.01
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


class FLIDPPServer:
    """
    Server for FL-IDPP privacy-preserving aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 0.01,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.clipping_norm = clipping_norm
        self.noise_multiplier = noise_multiplier
        self.learning_rate = learning_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

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
        Aggregate round with L2 clipping and DP Gaussian noise addition.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        clipped_updates = []
        for cid in participating:
            u = client_updates[cid]["delta_flat"]
            u_norm = np.linalg.norm(u)
            if u_norm > self.clipping_norm and u_norm > 0:
                clipped_u = u * (self.clipping_norm / u_norm)
            else:
                clipped_u = u.copy()
            clipped_updates.append(clipped_u)

        # Average clipped updates
        mean_delta = np.mean(clipped_updates, axis=0)

        # Add DP Gaussian noise
        noise_std = self.noise_multiplier * self.clipping_norm / len(participating)
        noise = self.rng.normal(0.0, noise_std, size=mean_delta.shape)
        perturbed_delta = mean_delta + noise

        agg_delta = unflatten_weights(perturbed_delta, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(perturbed_delta))
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
