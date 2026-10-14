"""
Detection Performance Experiment (Section 4.3, Tables 9-11 and Tables S7-S9).

Evaluates DWTA-FL against baseline federated intrusion detection schemes across:
  - CSE-CIC-IDS2018 (Table 9, Table S7)
  - UNSW-NB15 (Table 10, Table S8)
  - TON_IoT (Table 11, Table S9)
Across the five paired seeds: 101, 202, 303, 404, 505.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.loader import load_dataset
from data.partition import partition_dataset_dirichlet
from dwta_fl.client import FLClient
from dwta_fl.server import DWTAServer
from baselines.fedavg import FedAvgServer
from baselines.fltrust import FLTrustServer
from baselines.fedsbs import FedSBSServer
from baselines.fedmade import FedMADEServer
from baselines.krum import KrumServer
from baselines.bulyan import BulyanServer
from baselines.flex_ids import FLEXIDSServer
from baselines.fl_ids_pp import FLIDSPPServer
from baselines.fl_idpp import FLIDPPServer
from baselines.fed_anids import FedANIDSServer
from utils.metrics import compute_classification_metrics


def create_server_instance(method: str, input_dim: int, seed: int, x_train: np.ndarray, y_train: np.ndarray):
    """Instantiate server for specified method."""
    if method == "DWTA-FL":
        return DWTAServer(input_dim=input_dim, seed=seed)
    elif method == "FedAvg":
        return FedAvgServer(input_dim=input_dim, seed=seed)
    elif method == "FLTrust":
        # 1000 samples for clean root dataset
        idx = np.random.default_rng(seed).choice(len(x_train), size=min(1000, len(x_train)), replace=False)
        return FLTrustServer(input_dim=input_dim, root_x=x_train[idx], root_y=y_train[idx], seed=seed)
    elif method == "FedSBS":
        return FedSBSServer(input_dim=input_dim, seed=seed)
    elif method == "FedMADE":
        # 2000 samples for auxiliary dataset
        idx = np.random.default_rng(seed).choice(len(x_train), size=min(2000, len(x_train)), replace=False)
        return FedMADEServer(input_dim=input_dim, aux_x=x_train[idx], aux_y=y_train[idx], seed=seed)
    elif method == "Krum":
        return KrumServer(input_dim=input_dim, seed=seed)
    elif method == "Bulyan":
        return BulyanServer(input_dim=input_dim, seed=seed)
    elif method == "FLEX-IDS":
        return FLEXIDSServer(input_dim=input_dim, seed=seed)
    elif method == "FL-IDS++":
        return FLIDSPPServer(input_dim=input_dim, seed=seed)
    elif method == "FL-IDPP":
        return FLIDPPServer(input_dim=input_dim, seed=seed)
    elif method == "Fed-ANIDS":
        return FedANIDSServer(input_dim=input_dim, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_federated_training(
    dataset_name: str,
    method: str,
    seed: int,
    num_rounds: int = config.NUM_ROUNDS,
    num_clients: int = config.NUM_CLIENTS,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run one full federated training experiment for a (method, dataset, seed) tuple.
    """
    print(f"\n--- Running {method} on {dataset_name} (Seed: {seed}) ---")

    # 1. Load preprocessed data (80:20 train/test split, median imputer, one-hot, StandardScaler on train only)
    dataset = load_dataset(dataset_name, sample_limit=sample_limit, seed=seed)
    x_train, y_train = dataset["x_train"], dataset["y_train"]
    x_test, y_test = dataset["x_test"], dataset["y_test"]
    input_dim = x_train.shape[1]

    # 2. Partition training data among clients using Dirichlet(α=0.5, partition_seed=42)
    partitions = partition_dataset_dirichlet(
        x_train, y_train,
        num_clients=num_clients,
        alpha=config.DIRICHLET_ALPHA,
        seed=config.PARTITION_SEED,
    )

    # 3. Create clients
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

    # 4. Create server
    server = create_server_instance(method, input_dim, seed, x_train, y_train)

    # 5. Training loop
    for t in range(num_rounds):
        global_w = server.get_global_weights()
        client_updates = {}

        # Parallel client local training
        for cid in range(num_clients):
            res = clients[cid].train(global_w)
            client_updates[cid] = res

        # Server aggregation
        server_res = server.aggregate_round(t, client_updates)

        if (t + 1) % 10 == 0 or t == num_rounds - 1:
            loss, acc = server.evaluate(x_test, y_test)
            print(f"Round {t+1}/{num_rounds} - Test Loss: {loss:.4f}, Test Acc: {acc*100:.2f}%")

    # 6. Final evaluation on held-out test set
    preds_proba = server.predict_proba(x_test)
    preds = np.argmax(preds_proba, axis=1)
    metrics = compute_classification_metrics(y_test, preds)

    print(f"Final Metrics ({method}, {dataset_name}, seed={seed}):")
    print(f"  Accuracy: {metrics['accuracy']:.2f}%")
    print(f"  F1:       {metrics['f1']:.2f}%")
    print(f"  FPR:      {metrics['fpr']:.2f}%")

    client_stats = [
        {
            "benign": int(np.sum(c_y == 0)),
            "malicious": int(np.sum(c_y == 1)),
        }
        for (_, c_y) in partitions
    ]

    return {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "metrics": metrics,
        "history": server.history,
        "weights": server.get_global_weights(),
        "input_dim": input_dim,
        "client_stats": client_stats,
        "num_rounds": num_rounds,
    }


def run_detection_benchmark(
    dataset_name: str = "CSE-CIC-IDS2018",
    methods: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    num_rounds: int = config.NUM_ROUNDS,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run detection evaluation across seeds and methods, reporting mean ± SD.
    """
    if methods is None:
        methods = ["DWTA-FL", "FLEX-IDS", "FL-IDS++", "FL-IDPP", "Fed-ANIDS"]
    if seeds is None:
        seeds = config.RUN_SEEDS

    results_table = []
    for method in methods:
        method_metrics = []
        for seed in seeds:
            res = run_federated_training(
                dataset_name=dataset_name,
                method=method,
                seed=seed,
                num_rounds=num_rounds,
                sample_limit=sample_limit,
            )
            method_metrics.append(res["metrics"])

        accs = [m["accuracy"] for m in method_metrics]
        f1s = [m["f1"] for m in method_metrics]
        fprs = [m["fpr"] for m in method_metrics]

        # Use sample standard deviation (ddof=1)
        ddof = 1 if len(seeds) > 1 else 0
        results_table.append({
            "Method": method,
            "Accuracy (%)": f"{np.mean(accs):.2f} +/- {np.std(accs, ddof=ddof):.2f}",
            "F1 (%)": f"{np.mean(f1s):.2f} +/- {np.std(f1s, ddof=ddof):.2f}",
            "FPR (%)": f"{np.mean(fprs):.2f} +/- {np.std(fprs, ddof=ddof):.2f}",
        })

    df = pd.DataFrame(results_table)
    print(f"\n==================== {dataset_name} Results ====================")
    print(df.to_string(index=False))
    return df
