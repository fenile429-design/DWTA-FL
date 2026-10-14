"""
Drift Resilience Experiment (Section 4.4, Table 12 and Table S4).

Evaluates performance of DWTA-FL against baselines (FedAvg, FLTrust, FedSBS, FedMADE)
under increasing synthetic drift levels (0%, 10%, 20%, 30%) on CSE-CIC-IDS2018.

Drift schedule (Table 5):
  - Rounds 1-20: Warmup (no drift)
  - Rounds 21-60: Label-proportion shift (gradual) on clients 5, 10, 15, 20
  - Rounds 61-100: Covariate shift (sudden) on clients 3, 7, 12, 16, 19
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.loader import load_dataset
from data.partition import partition_dataset_dirichlet, create_client_windows
from data.drift import apply_drift_for_round
from dwta_fl.client import FLClient
from experiments.detection import create_server_instance
from utils.metrics import compute_classification_metrics


def run_drift_experiment_single(
    method: str,
    drift_level: float,  # 0.0, 0.10, 0.20, 0.30
    seed: int,
    dataset_name: str = "CSE-CIC-IDS2018",
    num_rounds: int = config.NUM_ROUNDS,
    num_clients: int = config.NUM_CLIENTS,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run one federated training experiment under the specified drift level.
    """
    print(f"\n--- Drift Eval: {method} | Drift: {int(drift_level*100)}% | Seed: {seed} ---")

    dataset = load_dataset(dataset_name, sample_limit=sample_limit, seed=seed)
    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]
    input_dim = x_train.shape[1]

    # Partition data
    partitions = partition_dataset_dirichlet(
        x_train, y_train,
        num_clients=num_clients,
        alpha=config.DIRICHLET_ALPHA,
        seed=config.PARTITION_SEED,
    )

    # Prepare per-round windows for each client
    client_windows = {}
    for cid in range(num_clients):
        c_x, c_y = partitions[cid]
        client_windows[cid] = create_client_windows(c_x, c_y, num_rounds=num_rounds, seed=seed + cid)

    # Create clients
    clients = {}
    for cid in range(num_clients):
        w_x, w_y = client_windows[cid][0]
        clients[cid] = FLClient(
            client_id=cid,
            input_dim=input_dim,
            x_data=w_x,
            y_data=w_y,
            local_epochs=config.LOCAL_EPOCHS,
            batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            seed=seed + cid,
        )

    # Create server
    server = create_server_instance(method, input_dim, seed, x_train, y_train)

    # Federated loop with drift injection
    for t in range(num_rounds):
        # Update each client's dataset window with scheduled drift if applicable
        for cid in range(num_clients):
            base_x, base_y = client_windows[cid][t]
            if drift_level > 0.0:
                drift_x, drift_y = apply_drift_for_round(
                    client_id=cid,
                    round_num=int(np.ceil((t + 1) * config.NUM_ROUNDS / num_rounds)),  # schedule rescaled to config.NUM_ROUNDS
                    x_data=base_x,
                    y_data=base_y,
                    drift_scale=drift_level,
                    seed=seed + t + cid,
                )
            else:
                drift_x, drift_y = base_x, base_y
            clients[cid].set_data(drift_x, drift_y)

        # Train & aggregate
        global_w = server.get_global_weights()
        client_updates = {}
        for cid in range(num_clients):
            client_updates[cid] = clients[cid].train(global_w)

        server.aggregate_round(t, client_updates)

    # Evaluate on held-out test set
    preds_proba = server.predict_proba(x_test)
    preds = np.argmax(preds_proba, axis=1)
    metrics = compute_classification_metrics(y_test, preds)

    # Final test loss
    test_loss, _ = server.evaluate(x_test, y_test)
    metrics["loss"] = float(test_loss)

    return {
        "method": method,
        "drift_level": drift_level,
        "seed": seed,
        "metrics": metrics,
    }


def run_drift_benchmark(
    drift_levels: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_rounds: int = config.NUM_ROUNDS,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run full Table 12 / Table S4 drift experiment across drift levels and methods.
    """
    if drift_levels is None:
        drift_levels = [0.0, 0.10, 0.20, 0.30]
    if methods is None:
        methods = ["FedAvg", "FLTrust", "FedSBS", "FedMADE", "DWTA-FL"]
    if seeds is None:
        seeds = config.RUN_SEEDS

    rows = []
    for d_lvl in drift_levels:
        for method in methods:
            metrics_list = []
            for seed in seeds:
                res = run_drift_experiment_single(
                    method=method,
                    drift_level=d_lvl,
                    seed=seed,
                    num_rounds=num_rounds,
                    sample_limit=sample_limit,
                )
                metrics_list.append(res["metrics"])

            mean_acc = np.mean([m["accuracy"] for m in metrics_list])
            mean_prec = np.mean([m["precision"] for m in metrics_list])
            mean_rec = np.mean([m["recall"] for m in metrics_list])
            mean_f1 = np.mean([m["f1"] for m in metrics_list])
            mean_fpr = np.mean([m["fpr"] for m in metrics_list])
            mean_loss = np.mean([m["loss"] for m in metrics_list])

            rows.append({
                "Drift Level": f"{int(d_lvl*100)}%",
                "Method": method,
                "Accuracy (%)": round(mean_acc, 2),
                "Precision (%)": round(mean_prec, 2),
                "Recall (%)": round(mean_rec, 2),
                "F1 (%)": round(mean_f1, 2),
                "FPR (%)": round(mean_fpr, 2),
                "Global Loss": round(mean_loss, 3),
            })

    df = pd.DataFrame(rows)
    print("\n==================== Drift Resilience Benchmark (Table 12) ====================")
    print(df.to_string(index=False))
    return df
