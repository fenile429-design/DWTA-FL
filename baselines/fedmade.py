"""
FedMADE Baseline (Section 3.7, Eq. 17, Table 8).

Group-based aggregation using class-probability matrices on an auxiliary dataset:
  - Auxiliary dataset: 2,000 samples
  - DBSCAN clustering on class-probability matrices: eps=0.5, min_samples=3
  - Group-weight quadratic optimization:
      min_{α_1,...,α_K ≥ 0} || ∑_{k=1}^K α_k P̄_{G_k} - I_C ||_2^2
  - Client weight: β_i ∝ α_k / |G_k| for i ∈ G_k
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import nnls
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


class FedMADEServer:
    """
    Server for FedMADE aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        aux_x: np.ndarray,
        aux_y: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 3,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.aux_x = aux_x
        self.aux_y = aux_y
        self.eps = eps
        self.min_samples = min_samples
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
        client_models: Optional[Dict[int, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate client updates using FedMADE algorithm (Eq. 17).
        """
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        # 1. Compute class-probability matrix for each client on auxiliary data
        current_w = self.get_global_weights()
        num_classes = config.MLP_OUTPUT_UNITS
        prob_matrices = []

        eval_model = create_mlp(self.input_dim, learning_rate=self.learning_rate, seed=self.seed)

        for cid in participating:
            c_delta = unflatten_weights(client_updates[cid]["delta_flat"], self.weight_shapes)
            c_weights = [gw + dw for gw, dw in zip(current_w, c_delta)]
            set_model_weights(eval_model, c_weights)

            preds = eval_model.predict(self.aux_x, verbose=0)  # shape (N, 2)
            # Compute empirical transition matrix P_i [C x C]
            mat = np.zeros((num_classes, num_classes), dtype=np.float64)
            for true_c in range(num_classes):
                idx = np.where(self.aux_y == true_c)[0]
                if len(idx) > 0:
                    mat[true_c] = np.mean(preds[idx], axis=0)
                else:
                    mat[true_c, true_c] = 1.0
            prob_matrices.append(mat.flatten())

        features = np.stack(prob_matrices, axis=0)

        # 2. DBSCAN clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(features)
        labels = db.labels_

        # Separate into clusters (noise -1 treated as individual clusters)
        clusters = {}
        for idx, lbl in enumerate(labels):
            key = lbl if lbl != -1 else f"noise_{idx}"
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(idx)

        group_keys = list(clusters.keys())
        k = len(group_keys)

        # Mean probability matrix per group
        p_groups = []
        for g_k in group_keys:
            indices = clusters[g_k]
            mean_mat = np.mean(features[indices], axis=0)  # shape (4,)
            p_groups.append(mean_mat)

        # 3. Solve NNLS: min || ∑ α_k P̄_{G_k} - I_C ||_2^2
        # Matrix A has shape (4, K), target b is vectorized identity (4,)
        mat_a = np.column_stack(p_groups)
        target_b = np.eye(num_classes).flatten()

        alpha_opt, _ = nnls(mat_a, target_b)
        if alpha_opt.sum() <= 1e-12:
            alpha_opt = np.ones(k) / k

        # 4. Compute client weights: β_i ∝ α_k / |G_k|
        client_weights = np.zeros(len(participating))
        for g_idx, g_k in enumerate(group_keys):
            indices = clusters[g_k]
            size = len(indices)
            weight_per_client = alpha_opt[g_idx] / size
            for i in indices:
                client_weights[i] = weight_per_client

        sum_w = client_weights.sum()
        if sum_w > 0:
            client_weights = client_weights / sum_w
        else:
            client_weights = np.ones(len(participating)) / len(participating)

        # 5. Aggregate updates
        agg_delta_flat = np.zeros_like(client_updates[participating[0]]["delta_flat"])
        for idx, cid in enumerate(participating):
            agg_delta_flat += client_weights[idx] * client_updates[cid]["delta_flat"]

        agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
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
