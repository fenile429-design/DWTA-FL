"""
Component Ablation Study for DWTA-FL (Section 4.11, Table 23).

Tests the contribution of each core component of DWTA-FL:
  1. Full DWTA-FL (all components active)
  2. w/o Drift Weighting (γ = 0)
  3. w/o Recursive Trust (μ = 0)
  4. w/o Temporal Consistency (ν = 0)
  5. w/o Volatility Penalty (τ = 0)
  6. w/o State Gating Rule (q_i^t = 1.0 for all clients)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.loader import load_dataset
from data.partition import partition_dataset_dirichlet
from dwta_fl.client import FLClient
from dwta_fl.server import DWTAServer
from dwta_fl.aggregation import compute_reliability, compute_aggregation_weights
from dwta_fl.state_rule import evaluate_client_states
from utils.metrics import compute_classification_metrics


class AblatedDWTAServer(DWTAServer):
    """Server subclass allowing selective disabling of DWTA-FL components."""

    def __init__(
        self,
        input_dim: int,
        variant: str,
        seed: Optional[int] = None,
        overrides: Optional[Dict[str, float]] = None,
    ):
        super().__init__(input_dim=input_dim, seed=seed)
        self.variant = variant
        self.overrides = overrides or {}
        if "beta" in self.overrides:
            self.trust_tracker.beta = self.overrides["beta"]

    def aggregate_round(
        self,
        round_num: int,
        client_updates: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        participating = list(client_updates.keys())
        if not participating:
            return {"skipped": True}

        # Drift tracking
        prev_smoothed = {cid: self.drift_tracker.get_smoothed_drift(cid) for cid in participating}
        for cid in participating:
            self.drift_tracker.update(cid, client_updates[cid]["delta_flat"])

        curr_smoothed = {cid: self.drift_tracker.get_smoothed_drift(cid) for cid in participating}
        uncertainties = {cid: self.drift_tracker.get_drift_uncertainty(cid) for cid in participating}

        # Trust tracking
        dirs = {cid: client_updates[cid]["direction_flat"] for cid in participating}
        trust_res = self.trust_tracker.update_round(dirs)

        # Consistency & Volatility tracking
        cons_res = self.consistency_tracker.update_round(dirs, curr_smoothed, prev_smoothed)

        # Apply ablation parameters
        mu = 0.0 if self.variant == "wo_trust" else self.overrides.get("mu", config.TRUST_EXPONENT_MU)
        nu = 0.0 if self.variant == "wo_consistency" else self.overrides.get("nu", config.CONSISTENCY_EXPONENT_NU)
        gamma = 0.0 if self.variant == "wo_drift" else self.overrides.get("gamma", config.DRIFT_PENALTY_GAMMA)
        tau = 0.0 if self.variant == "wo_volatility" else self.overrides.get("tau", config.VOLATILITY_PENALTY_TAU)

        reliabilities = {}
        for cid in participating:
            t_score = trust_res[cid]["trust"]
            c_score = cons_res[cid]["consistency"]
            d_hat = curr_smoothed[cid]
            v_score = cons_res[cid]["volatility"]

            reliabilities[cid] = compute_reliability(
                trust=t_score,
                consistency=c_score,
                smoothed_drift=d_hat,
                volatility=v_score,
                mu=mu,
                nu=nu,
                gamma=gamma,
                tau=tau,
            )

        if self.variant == "wo_state_rule":
            gates = {cid: 1.0 for cid in participating}
        else:
            state_evals = evaluate_client_states(
                reliabilities, uncertainties, curr_smoothed,
                theta_d=self.overrides.get("theta_d", config.DRIFT_THRESHOLD_THETA_D),
            )
            gates = {cid: state_evals[cid]["gate"] for cid in participating}

        weights, skip_update = compute_aggregation_weights(reliabilities, gates)

        # Model update
        current_w = self.get_global_weights()
        if not skip_update:
            agg_delta_flat = np.zeros_like(client_updates[participating[0]]["delta_flat"])
            for cid in participating:
                if weights[cid] > 0:
                    agg_delta_flat += weights[cid] * client_updates[cid]["delta_flat"]

            from models.mlp import unflatten_weights
            agg_delta = unflatten_weights(agg_delta_flat, self.weight_shapes)
            new_w = [gw + dw for gw, dw in zip(current_w, agg_delta)]
            self.set_global_weights(new_w)
            norm_diff = float(np.linalg.norm(agg_delta_flat))
        else:
            norm_diff = 0.0

        return {"round": round_num, "norm_diff": norm_diff}


ABLATION_VARIANTS = {
    "Full DWTA-FL": "full",
    "w/o Drift Weighting (gamma=0)": "wo_drift",
    "w/o Recursive Trust (mu=0)": "wo_trust",
    "w/o Temporal Consistency (nu=0)": "wo_consistency",
    "w/o Volatility Penalty (tau=0)": "wo_volatility",
    "w/o State Gating Rule (q=1)": "wo_state_rule",
}


def run_dwta_variant(
    dataset_name: str,
    variant: str,
    seed: int,
    num_rounds: int,
    sample_limit: Optional[int] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Train one DWTA-FL variant for one seed and return test metrics."""
    import tensorflow as tf
    tf.keras.backend.clear_session()

    dataset = load_dataset(dataset_name, sample_limit=sample_limit, seed=seed)
    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]
    input_dim = x_train.shape[1]

    partitions = partition_dataset_dirichlet(
        x_train, y_train, num_clients=config.NUM_CLIENTS, seed=config.PARTITION_SEED
    )
    clients = {
        cid: FLClient(
            client_id=cid,
            input_dim=input_dim,
            x_data=partitions[cid][0],
            y_data=partitions[cid][1],
            seed=seed + cid,
        )
        for cid in range(config.NUM_CLIENTS)
    }

    server = AblatedDWTAServer(
        input_dim=input_dim, variant=variant, seed=seed, overrides=overrides
    )

    for t in range(num_rounds):
        gw = server.get_global_weights()
        updates = {cid: clients[cid].train(gw) for cid in range(config.NUM_CLIENTS)}
        server.aggregate_round(t, updates)

    preds = np.argmax(server.predict_proba(x_test), axis=1)
    return compute_classification_metrics(y_test, preds)


def run_ablation_study(
    dataset_name: str = "CSE-CIC-IDS2018",
    seeds: Optional[List[int]] = None,
    num_rounds: int = 50,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run ablation across all variants."""
    if seeds is None:
        seeds = [101]

    rows = []
    for var_label, var_code in ABLATION_VARIANTS.items():
        var_metrics = [
            run_dwta_variant(dataset_name, var_code, s, num_rounds, sample_limit)
            for s in seeds
        ]
        rows.append({
            "Variant": var_label,
            "Accuracy (%)": round(np.mean([m["accuracy"] for m in var_metrics]), 2),
            "F1 (%)": round(np.mean([m["f1"] for m in var_metrics]), 2),
            "FPR (%)": round(np.mean([m["fpr"] for m in var_metrics]), 2),
        })

    df = pd.DataFrame(rows)
    print("\n==================== Component Ablation Study (Table 23) ====================")
    print(df.to_string(index=False))
    return df

