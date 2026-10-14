"""
Adversarial Attack Robustness Experiment (Section 4.6, Tables 14-17 and Table S5).

Evaluates robustness of DWTA-FL against adversarial attack types:
  - Label Poisoning (20% clients, 20% flipped)
  - Sign Flipping (20% clients, reverse scale 3)
  - Backdoor (20% clients, 10% injected with triggers)
  - Byzantine (20% clients, 3× norm random)
  - Adaptive Byzantine (20% clients, inverted consensus)
  - Collaborative Collusion (10%, 20%, 30% clients)
  - Covert Gradient Escape (10%, 20%, 30% clients)
  - Extreme Stress (50% compromised clients)

Attacks active in rounds 20–100.
Evaluates Attack Resistance Rate (ARR, Eq. 29), Accuracy, F1, and FPR.
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
from experiments.detection import create_server_instance
from utils.metrics import compute_classification_metrics, compute_arr
from attacks.adversarial import (
    apply_label_poisoning,
    apply_sign_flipping,
    apply_backdoor,
    apply_byzantine_attack,
    apply_adaptive_byzantine,
    apply_collaborative_collusion,
    apply_covert_gradient_escape,
)


def run_attack_experiment(
    method: str,
    attack_name: str,
    seed: int,
    dataset_name: str = "CSE-CIC-IDS2018",
    num_rounds: int = config.NUM_ROUNDS,
    num_clients: int = config.NUM_CLIENTS,
    malicious_fraction: float = config.MALICIOUS_CLIENT_SHARE,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run federated training with adversarial attacks active during rounds 20-100.
    """
    print(f"\n--- Attack Eval: {method} | Attack: {attack_name} | Seed: {seed} ---")

    dataset = load_dataset(dataset_name, sample_limit=sample_limit, seed=seed)
    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]
    feature_names = dataset.get("feature_names", [])
    input_dim = x_train.shape[1]

    partitions = partition_dataset_dirichlet(
        x_train, y_train,
        num_clients=num_clients,
        alpha=config.DIRICHLET_ALPHA,
        seed=config.PARTITION_SEED,
    )

    num_malicious = int(num_clients * malicious_fraction)
    malicious_cids = list(range(num_malicious))  # e.g., clients 0, 1, 2, 3

    clients = {}
    for cid in range(num_clients):
        c_x, c_y = partitions[cid]
        clients[cid] = FLClient(
            client_id=cid,
            input_dim=input_dim,
            x_data=c_x,
            y_data=c_y,
            local_epochs=config.LOCAL_EPOCHS,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            seed=seed + cid,
        )

    server = create_server_instance(method, input_dim, seed, x_train, y_train)

    total_attempts = 0
    resisted_attempts = 0

    for t in range(num_rounds):
        global_w = server.get_global_weights()
        client_updates = {}
        is_attack_active = (t >= int(config.ATTACK_START_ROUND * num_rounds / config.NUM_ROUNDS))

        # 1. Honest updates first to estimate benign consensus
        benign_cids = [c for c in range(num_clients) if c not in malicious_cids]
        benign_deltas = []
        for cid in benign_cids:
            res = clients[cid].train(global_w)
            client_updates[cid] = res
            benign_deltas.append(res["delta_flat"])

        stacked_benign = np.stack(benign_deltas, axis=0)
        benign_mean = np.mean(stacked_benign, axis=0)
        benign_norms = np.linalg.norm(stacked_benign, axis=1)
        benign_median_norm = float(np.median(benign_norms))

        # 2. Malicious client updates
        for cid in malicious_cids:
            if not is_attack_active:
                # Normal training prior to attack start
                client_updates[cid] = clients[cid].train(global_w)
            else:
                total_attempts += 1
                base_x, base_y = partitions[cid]

                if attack_name == "label_poisoning":
                    p_x, p_y = apply_label_poisoning(base_x, base_y, seed=seed + t + cid)
                    clients[cid].set_data(p_x, p_y)
                    res = clients[cid].train(global_w)
                    clients[cid].set_data(base_x, base_y)  # restore
                    client_updates[cid] = res

                elif attack_name == "backdoor":
                    p_x, p_y = apply_backdoor(base_x, base_y, dataset_name, feature_names, seed=seed + t + cid)
                    clients[cid].set_data(p_x, p_y)
                    res = clients[cid].train(global_w)
                    clients[cid].set_data(base_x, base_y)  # restore
                    client_updates[cid] = res

                elif attack_name == "sign_flipping":
                    honest_res = clients[cid].train(global_w)
                    mal_delta_flat = apply_sign_flipping(honest_res["delta_flat"])
                    honest_res["delta_flat"] = mal_delta_flat
                    honest_res["direction_flat"] = -mal_delta_flat / config.LEARNING_RATE
                    client_updates[cid] = honest_res

                elif attack_name == "byzantine":
                    honest_res = clients[cid].train(global_w)
                    mal_delta_flat = apply_byzantine_attack(
                        benign_median_norm, len(benign_mean), seed=seed + t + cid
                    )
                    honest_res["delta_flat"] = mal_delta_flat
                    honest_res["direction_flat"] = -mal_delta_flat / config.LEARNING_RATE
                    client_updates[cid] = honest_res

                elif attack_name == "adaptive_byzantine":
                    honest_res = clients[cid].train(global_w)
                    mal_delta_flat = apply_adaptive_byzantine(benign_mean, benign_median_norm)
                    honest_res["delta_flat"] = mal_delta_flat
                    honest_res["direction_flat"] = -mal_delta_flat / config.LEARNING_RATE
                    client_updates[cid] = honest_res

                elif attack_name == "collusion":
                    honest_res = clients[cid].train(global_w)
                    mal_delta_flat = apply_collaborative_collusion(benign_mean)
                    honest_res["delta_flat"] = mal_delta_flat
                    honest_res["direction_flat"] = -mal_delta_flat / config.LEARNING_RATE
                    client_updates[cid] = honest_res

                elif attack_name == "covert_escape":
                    honest_res = clients[cid].train(global_w)
                    mal_delta_flat = apply_covert_gradient_escape(
                        honest_res["delta_flat"], benign_median_norm, seed=seed + t + cid
                    )
                    honest_res["delta_flat"] = mal_delta_flat
                    honest_res["direction_flat"] = -mal_delta_flat / config.LEARNING_RATE
                    client_updates[cid] = honest_res

                else:
                    client_updates[cid] = clients[cid].train(global_w)

        # Server aggregation
        server_res = server.aggregate_round(t, client_updates)

        # Track resisted attempts (for DWTA-FL: gated to Attacker or weight ~ 0)
        if is_attack_active:
            weights = server_res.get("weights", {})
            gates = server_res.get("gates", {})
            for cid in malicious_cids:
                w_val = weights.get(cid, 0.0)
                g_val = gates.get(cid, 1.0)
                # Count as resisted if gated with κ_a or negligible weight
                if g_val <= config.GATE_ATTACKER_KAPPA_A or w_val <= 0.01:
                    resisted_attempts += 1

    # Final evaluation
    preds_proba = server.predict_proba(x_test)
    preds = np.argmax(preds_proba, axis=1)
    metrics = compute_classification_metrics(y_test, preds)

    arr = compute_arr(resisted_attempts, total_attempts)
    metrics["arr"] = arr

    return {
        "method": method,
        "attack": attack_name,
        "seed": seed,
        "metrics": metrics,
    }


def run_attack_benchmark(
    attacks: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_rounds: int = config.NUM_ROUNDS,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Run attack benchmark across multiple attacks and methods."""
    if attacks is None:
        attacks = ["label_poisoning", "sign_flipping", "byzantine", "adaptive_byzantine", "collusion", "covert_escape"]
    if methods is None:
        methods = ["FedAvg", "FLTrust", "Krum", "Bulyan", "DWTA-FL"]
    if seeds is None:
        seeds = [101]

    rows = []
    for atk in attacks:
        for m in methods:
            m_res = []
            for s in seeds:
                res = run_attack_experiment(
                    method=m,
                    attack_name=atk,
                    seed=s,
                    num_rounds=num_rounds,
                    sample_limit=sample_limit,
                )
                m_res.append(res["metrics"])

            mean_acc = np.mean([x["accuracy"] for x in m_res])
            mean_f1 = np.mean([x["f1"] for x in m_res])
            mean_fpr = np.mean([x["fpr"] for x in m_res])
            mean_arr = np.mean([x["arr"] for x in m_res])

            rows.append({
                "Attack": atk,
                "Method": m,
                "Accuracy (%)": round(mean_acc, 2),
                "F1 (%)": round(mean_f1, 2),
                "FPR (%)": round(mean_fpr, 2),
                "ARR (%)": round(mean_arr, 2),
            })

    df = pd.DataFrame(rows)
    print("\n==================== Attack Robustness Benchmark ====================")
    print(df.to_string(index=False))
    return df
