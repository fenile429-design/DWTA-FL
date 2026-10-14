"""
Federated Partitioning for DWTA-FL.

Implements Dirichlet(α=0.5) class-based non-IID partitioning
as described in Section 4.1.2.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def dirichlet_partition(
    y_train: np.ndarray,
    num_clients: int = config.NUM_CLIENTS,
    alpha: float = config.DIRICHLET_ALPHA,
    seed: int = config.PARTITION_SEED,
) -> List[np.ndarray]:
    """
    Partition the training indices among clients using Dirichlet class split.

    For each class c, a num_clients-dimensional proportion vector is drawn from
    Dirichlet(alpha, ..., alpha) and the class-c training samples are allocated
    in those proportions.

    Args:
        y_train: Training labels array
        num_clients: Number of clients (N=20)
        alpha: Dirichlet concentration parameter (0.5)
        seed: Random seed for reproducibility (42)

    Returns:
        List of arrays, each containing indices for that client.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Initialize client index lists
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        # Get all indices for class c
        class_indices = np.where(y_train == c)[0]
        rng.shuffle(class_indices)

        # Draw proportions from Dirichlet(alpha, ..., alpha)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to counts
        counts = (proportions * len(class_indices)).astype(int)

        # Distribute remaining samples due to rounding
        remainder = len(class_indices) - counts.sum()
        if remainder > 0:
            # Add remaining to random clients
            extra_clients = rng.choice(num_clients, size=remainder, replace=False)
            for ec in extra_clients:
                counts[ec] += 1

        # Assign indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end

    # Convert to numpy arrays and shuffle within each client
    for i in range(num_clients):
        client_indices[i] = np.array(client_indices[i], dtype=np.int64)
        rng.shuffle(client_indices[i])

    return client_indices


def get_client_stats(
    client_indices: List[np.ndarray],
    y_train: np.ndarray,
) -> Dict[int, Dict[str, int]]:
    """
    Compute per-client class distribution statistics (Table S3).

    Returns:
        Dictionary mapping client_id → {"benign": count, "malicious": count}
    """
    stats = {}
    for client_id, indices in enumerate(client_indices):
        labels = y_train[indices]
        stats[client_id] = {
            "benign": int(np.sum(labels == 0)),
            "malicious": int(np.sum(labels == 1)),
            "total": len(indices),
        }
    return stats


def print_partition_stats(
    client_stats: Dict[int, Dict[str, int]],
    dataset_name: str,
):
    """Print partition statistics matching Table S3 format."""
    print(f"\n{'='*60}")
    print(f"Partition Statistics - {dataset_name}")
    print(f"{'='*60}")
    print(f"{'Client':>8} {'Benign':>12} {'Malicious':>12} {'Total':>12}")
    print(f"{'-'*48}")

    total_benign = 0
    total_malicious = 0
    for cid in sorted(client_stats.keys()):
        s = client_stats[cid]
        print(f"{cid+1:>8d} {s['benign']:>12,d} {s['malicious']:>12,d} {s['total']:>12,d}")
        total_benign += s["benign"]
        total_malicious += s["malicious"]

    print(f"{'-'*48}")
    print(f"{'Total':>8} {total_benign:>12,d} {total_malicious:>12,d} {total_benign+total_malicious:>12,d}")


def partition_dataset_dirichlet(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_clients: int = config.NUM_CLIENTS,
    alpha: float = config.DIRICHLET_ALPHA,
    seed: int = config.PARTITION_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition dataset into (X, y) pairs for each client using Dirichlet class distribution.
    """
    indices = dirichlet_partition(y_train, num_clients, alpha, seed)

    # Guarantee non-empty partitions (matters for small sample limits):
    # empty clients borrow one random sample from the largest client.
    rng = np.random.default_rng(seed)
    for cid in range(num_clients):
        if len(indices[cid]) == 0:
            donor = int(np.argmax([len(ix) for ix in indices]))
            pick = int(rng.integers(len(indices[donor])))
            indices[cid] = np.array([indices[donor][pick]], dtype=np.int64)

    partitions = []
    for cid in range(num_clients):
        idx = indices[cid]
        partitions.append((x_train[idx], y_train[idx]))
    return partitions


def create_client_windows(
    x_or_indices: Any,
    y_data: Optional[np.ndarray] = None,
    num_rounds: int = config.NUM_ROUNDS,
    seed: Optional[int] = None,
) -> Any:
    """
    Create per-round data windows for a client.
    Supports either (x_data, y_data) -> List of (w_x, w_y)
    or client_indices_list -> List of List of indices
    """
    if y_data is not None:
        # (x_data, y_data) mode for a single client
        x_data = x_or_indices
        n = len(x_data)
        window_size = max(1, n // num_rounds)
        windows = []
        for t in range(num_rounds):
            start = (t * window_size) % n
            end = min(start + window_size, n)
            if end <= start:
                w_x = np.concatenate([x_data[start:], x_data[:window_size - (n - start)]])
                w_y = np.concatenate([y_data[start:], y_data[:window_size - (n - start)]])
            else:
                w_x = x_data[start:end]
                w_y = y_data[start:end]
            windows.append((w_x, w_y))
        return windows

    # Otherwise list of indices mode
    client_indices = x_or_indices
    windows = []
    for indices in client_indices:
        n = len(indices)
        if n == 0:
            windows.append([np.array([], dtype=np.int64)] * num_rounds)
            continue

        window_size = max(1, n // num_rounds)
        c_windows = []
        for t in range(num_rounds):
            start = (t * window_size) % n
            end = min(start + window_size, n)
            if end <= start:
                w_indices = np.concatenate([indices[start:], indices[:window_size - (n - start)]])
            else:
                w_indices = indices[start:end]
            c_windows.append(w_indices)
        windows.append(c_windows)
    return windows

