"""
DWTA-FL Server Implementation (Algorithm 1).

Orchestrates federated rounds:
  1. Broadcasts global model w_t
  2. Receives client updates Δw_i^t
  3. Computes drift, trust, consistency, volatility, reliability
  4. Applies drift-versus-attack state classification and gating
  5. Computes normalized weights α_i^t
  6. Aggregates update: w_{t+1} = w_t + ∑ α_i^t Δw_i^t
  7. Records stability monitoring rule ||w_{t+1} - w_t||_2 ≤ ε_stab
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mlp import (
    create_mlp,
    get_model_weights,
    set_model_weights,
    flatten_weights,
    unflatten_weights,
)
from dwta_fl.drift import DriftTracker
from dwta_fl.trust import TrustTracker
from dwta_fl.consistency import ConsistencyVolatilityTracker
from dwta_fl.aggregation import compute_reliability, compute_aggregation_weights
from dwta_fl.state_rule import evaluate_client_states


class DWTAServer:
    """
    Aggregation Server for DWTA-FL.
    """

    def __init__(
        self,
        input_dim: int,
        num_clients: int = config.NUM_CLIENTS,
        learning_rate: float = config.LEARNING_RATE,
        stability_epsilon: float = config.STABILITY_EPSILON,
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        self.stability_epsilon = stability_epsilon
        self.seed = seed

        # Global model
        self.global_model = create_mlp(
            input_dim=input_dim,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.weight_shapes = [w.shape for w in self.global_model.get_weights()]

        # Trackers
        self.drift_tracker = DriftTracker(num_clients=num_clients)
        self.trust_tracker = TrustTracker(num_clients=num_clients)
        self.consistency_tracker = ConsistencyVolatilityTracker(num_clients=num_clients)

        # Round history
        self.history: List[Dict[str, Any]] = []

    def get_global_weights(self) -> List[np.ndarray]:
        """Get current global weights w_t."""
        return get_model_weights(self.global_model)

    def set_global_weights(self, weights: List[np.ndarray]):
        """Set global weights."""
        set_model_weights(self.global_model, weights)

    def aggregate_round(
        self,
        round_num: int,
        client_updates: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute server-side aggregation for one communication round (Algorithm 1).

        Args:
            round_num: Current round t (0-indexed)
            client_updates: Dict mapping client_id to client training output:
                - delta_weights: List[np.ndarray]
                - delta_flat: np.ndarray
                - direction_flat: np.ndarray
                - num_samples: int

        Returns:
            Dict containing round metrics, weights, states, and stability info.
        """
        participating_clients = list(client_updates.keys())
        if not participating_clients:
            return {"skipped": True, "reason": "No participating clients"}

        # 1. Update Drift Tracker (Eqs. 3-5)
        prev_smoothed_drifts = {
            cid: self.drift_tracker.get_smoothed_drift(cid)
            for cid in participating_clients
        }
        drift_metrics = {}
        for cid in participating_clients:
            d_res = self.drift_tracker.update(
                cid, client_updates[cid]["delta_flat"]
            )
            drift_metrics[cid] = d_res

        curr_smoothed_drifts = {
            cid: self.drift_tracker.get_smoothed_drift(cid)
            for cid in participating_clients
        }
        drift_uncertainties = {
            cid: self.drift_tracker.get_drift_uncertainty(cid)
            for cid in participating_clients
        }

        # 2. Update Trust Tracker (Eqs. 6-7)
        directions = {
            cid: client_updates[cid]["direction_flat"]
            for cid in participating_clients
        }
        trust_metrics = self.trust_tracker.update_round(directions)

        # 3. Update Consistency & Volatility (Eqs. 8-9)
        consistency_metrics = self.consistency_tracker.update_round(
            directions,
            curr_smoothed_drifts,
            prev_smoothed_drifts,
        )

        # 4. Compute Reliability R_i^t (Eq. 10)
        reliabilities = {}
        for cid in participating_clients:
            t_score = trust_metrics[cid]["trust"]
            c_score = consistency_metrics[cid]["consistency"]
            d_hat = curr_smoothed_drifts[cid]
            v_score = consistency_metrics[cid]["volatility"]

            reliabilities[cid] = compute_reliability(
                trust=t_score,
                consistency=c_score,
                smoothed_drift=d_hat,
                volatility=v_score,
            )

        # 5. Drift-versus-Attack State Rule (Eqs. 12-14)
        state_evaluations = evaluate_client_states(
            reliabilities=reliabilities,
            drift_uncertainties=drift_uncertainties,
            smoothed_drifts=curr_smoothed_drifts,
        )

        gates = {cid: state_evaluations[cid]["gate"] for cid in participating_clients}

        # 6. Final Aggregation Weights α_i^t (Eq. 11)
        weights, skip_update = compute_aggregation_weights(reliabilities, gates)

        # 7. Model Update (Eq. 2)
        current_w = self.get_global_weights()
        if skip_update:
            norm_diff = 0.0
            is_stable = True
        else:
            # ∑ α_i^t * Δw_i^t
            agg_delta_flat = np.zeros_like(client_updates[participating_clients[0]]["delta_flat"])
            for cid in participating_clients:
                alpha = weights[cid]
                if alpha > 0.0:
                    agg_delta_flat += alpha * client_updates[cid]["delta_flat"]

            agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
            new_weights = [gw + dw for gw, dw in zip(current_w, agg_delta)]
            self.set_global_weights(new_weights)

            norm_diff = float(np.linalg.norm(agg_delta_flat))
            is_stable = bool(norm_diff <= self.stability_epsilon)

        round_record = {
            "round": round_num,
            "skip_update": skip_update,
            "norm_diff": norm_diff,
            "is_stable": is_stable,
            "weights": weights,
            "states": {cid: state_evaluations[cid]["state"] for cid in participating_clients},
            "gates": gates,
            "reliabilities": reliabilities,
            "trust": {cid: trust_metrics[cid]["trust"] for cid in participating_clients},
            "drifts": curr_smoothed_drifts,
            "consistencies": {cid: consistency_metrics[cid]["consistency"] for cid in participating_clients},
            "volatilities": {cid: consistency_metrics[cid]["volatility"] for cid in participating_clients},
        }

        self.history.append(round_record)
        return round_record

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate global model on test set."""
        loss, acc = self.global_model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get predicted class probabilities (N, 2)."""
        return self.global_model.predict(x, verbose=0)
