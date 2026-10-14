"""
Bulyan Baseline (Mhamdi et al., ICML 2018).

Krum-based multi-candidate selection followed by coordinate-wise trimmed mean:
  - Assumed Byzantine clients: f = 4
  - Requires n ≥ 4f + 3 in theoretical proof; adapted for n=20 with f=4.
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
from baselines.krum import krum_select_index


def bulyan_aggregate(
    updates: np.ndarray,
    f: int = 4,
) -> np.ndarray:
    """
    Bulyan aggregation:
      Step 1: Iteratively select θ = n - 2f candidates using Krum.
      Step 2: Coordinate-wise trimmed mean discarding the f lowest and highest values.
    """
    n, d = updates.shape
    theta = max(1, n - 2 * f)

    # Iterative selection using Krum
    selected_indices = []
    remaining_indices = list(range(n))

    for _ in range(theta):
        if not remaining_indices:
            break
        sub_updates = updates[remaining_indices]
        # Run Krum on subset
        best_sub_idx = krum_select_index(sub_updates, f=min(f, max(0, len(remaining_indices) - 3)))
        original_idx = remaining_indices[best_sub_idx]
        selected_indices.append(original_idx)
        remaining_indices.pop(best_sub_idx)

    candidates = updates[selected_indices]  # Shape: (theta, d)
    num_cand = len(candidates)

    # Coordinate-wise trimmed mean
    # Discard f lowest and f highest values if possible
    trim = min(f, (num_cand - 1) // 2)

    # Sort each coordinate
    sorted_candidates = np.sort(candidates, axis=0)
    if trim > 0 and (num_cand - 2 * trim) > 0:
        trimmed = sorted_candidates[trim : num_cand - trim, :]
        agg = np.mean(trimmed, axis=0)
    else:
        agg = np.mean(candidates, axis=0)

    return agg


class BulyanServer:
    """
    Server for Bulyan robust aggregation.
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
        Aggregate round using Bulyan algorithm.
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        updates = np.stack([client_updates[cid]["delta_flat"] for cid in participating], axis=0)
        agg_delta_flat = bulyan_aggregate(updates, f=self.f)

        agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
        current_w = self.get_global_weights()
        new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
        self.set_global_weights(new_w)

        norm_diff = float(np.linalg.norm(agg_delta_flat))
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
