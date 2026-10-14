"""
Synthetic Drift Generation for DWTA-FL.

Implements drift injection as described in Section 4.1.3 and Table 5:
  - Label-proportion shift (rounds 21–60): cyclic class ratio alteration
  - Covariate shift (rounds 61–100): feature distribution shift via
    inter-client sample substitution

The drift level is the fraction of samples in a drifting client's window
that are replaced.
"""

import numpy as np
from typing import List, Optional, Tuple
from copy import deepcopy

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def apply_drift_to_windows(
    windows: List[List[np.ndarray]],
    y_train: np.ndarray,
    X_train: np.ndarray,
    drift_level: float = 0.30,
    seed: int = 42,
) -> List[List[np.ndarray]]:
    """
    Apply synthetic drift to client windows according to Table 5.

    Args:
        windows: windows[client_id][round] = indices
        y_train: Training labels
        X_train: Training features
        drift_level: Fraction δ ∈ {0, 0.10, 0.20, 0.30}
        seed: Random seed

    Returns:
        Modified windows with drift applied.
    """
    if drift_level == 0.0:
        return windows

    rng = np.random.default_rng(seed)
    drifted_windows = deepcopy(windows)
    schedule = config.DRIFT_SCHEDULE

    # ---- Phase 1: Label-proportion shift (rounds 21-60) ----
    label_shift_cfg = schedule["label_shift"]
    label_clients = label_shift_cfg["clients"]
    label_rounds = label_shift_cfg["rounds"]

    for t in range(label_rounds[0] - 1, label_rounds[1]):  # 0-indexed
        for client_id in label_clients:
            window_indices = drifted_windows[client_id][t]
            if len(window_indices) == 0:
                continue

            # Gradual onset: linear ramp over rounds 21-40
            ramp_start = label_rounds[0] - 1  # round 20 (0-indexed)
            ramp_end = ramp_start + 20  # round 40 (0-indexed)

            if t <= ramp_end:
                # Linear ramp from 0 to drift_level
                progress = (t - ramp_start) / max(1, ramp_end - ramp_start)
                current_drift = drift_level * progress
            else:
                current_drift = drift_level

            n_replace = int(len(window_indices) * current_drift)
            if n_replace == 0:
                continue

            # Cyclic class ratio alteration: replace some samples with
            # samples from the opposite class
            window_labels = y_train[window_indices]

            # Find samples to replace (preferring majority class)
            majority_class = 0 if np.sum(window_labels == 0) >= np.sum(window_labels == 1) else 1
            minority_class = 1 - majority_class

            # Get indices of majority class in window
            majority_mask = window_labels == majority_class
            majority_positions = np.where(majority_mask)[0]

            n_to_swap = min(n_replace, len(majority_positions))
            if n_to_swap == 0:
                continue

            # Select positions to replace
            swap_positions = rng.choice(majority_positions, size=n_to_swap, replace=False)

            # Find replacement samples (minority class from training pool)
            minority_pool = np.where(y_train == minority_class)[0]
            replacement_indices = rng.choice(minority_pool, size=n_to_swap, replace=True)

            # Replace in window
            new_window = window_indices.copy()
            new_window[swap_positions] = replacement_indices
            drifted_windows[client_id][t] = new_window

    # ---- Phase 2: Covariate shift (rounds 61-100) ----
    covariate_cfg = schedule["covariate_shift"]
    cov_clients = covariate_cfg["clients"]
    cov_rounds = covariate_cfg["rounds"]

    for t in range(cov_rounds[0] - 1, min(cov_rounds[1], len(windows[0]))):
        for client_id in cov_clients:
            window_indices = drifted_windows[client_id][t]
            if len(window_indices) == 0:
                continue

            # Sudden onset at round 61
            current_drift = drift_level
            n_replace = int(len(window_indices) * current_drift)
            if n_replace == 0:
                continue

            # Inter-client sample substitution
            # Pick a random different client to borrow samples from
            other_clients = [c for c in range(len(windows)) if c != client_id]
            donor_client = rng.choice(other_clients)
            donor_indices = drifted_windows[donor_client][t]

            if len(donor_indices) == 0:
                continue

            n_borrow = min(n_replace, len(donor_indices))
            positions_to_replace = rng.choice(
                len(window_indices), size=n_borrow, replace=False
            )
            borrowed = rng.choice(donor_indices, size=n_borrow, replace=True)

            new_window = window_indices.copy()
            new_window[positions_to_replace] = borrowed
            drifted_windows[client_id][t] = new_window

    return drifted_windows


def apply_feature_heterogeneity(
    X_train: np.ndarray,
    client_indices: List[np.ndarray],
    feature_names: List[str],
    target_features: List[str] = None,
    scale_range: Tuple[float, float] = config.FEATURE_HETERO_SCALE_RANGE,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply feature heterogeneity (H2, Section 4.1.4).

    Client-specific multiplicative scaling of selected features
    by a factor drawn from U(0.7, 1.3) per client.

    Args:
        X_train: Training data (will be modified in-place)
        client_indices: Per-client index arrays
        feature_names: List of feature names
        target_features: Features to scale (defaults to config)
        scale_range: Uniform scale range

    Returns:
        Modified X_train
    """
    if target_features is None:
        target_features = config.FEATURE_HETERO_FEATURES

    rng = np.random.default_rng(seed)
    X_modified = X_train.copy()

    # Find feature indices
    feature_indices = []
    for feat in target_features:
        if feat in feature_names:
            feature_indices.append(feature_names.index(feat))

    if not feature_indices:
        print("Warning: No matching features found for heterogeneity scaling")
        return X_modified

    for client_id, indices in enumerate(client_indices):
        if len(indices) == 0:
            continue

        # Draw a per-client scaling factor
        scale_factor = rng.uniform(scale_range[0], scale_range[1])

        for fidx in feature_indices:
            X_modified[indices, fidx] *= scale_factor

    return X_modified


def apply_drift_for_round(
    client_id: int,
    round_num: int,
    x_data: np.ndarray,
    y_data: np.ndarray,
    drift_scale: float = 0.30,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply scheduled synthetic drift to a client's data for a specific round (Table 5).

    Args:
        client_id: Client index (0-indexed, 0..19)
        round_num: Current communication round (1-indexed, 1..100)
        x_data: Client's features for this round
        y_data: Client's labels for this round
        drift_scale: Target drift level δ ∈ {0.10, 0.20, 0.30}
        seed: Random seed

    Returns:
        (drifted_x, drifted_y)
    """
    if drift_scale <= 0.0 or len(y_data) == 0:
        return x_data, y_data

    rng = np.random.default_rng(seed)
    schedule = config.DRIFT_SCHEDULE

    # Phase 1: Label-proportion shift (rounds 21-60, clients 5, 10, 15, 20)
    label_cfg = schedule["label_shift"]
    l_rounds = label_cfg["rounds"]
    if l_rounds[0] <= round_num <= l_rounds[1] and client_id in label_cfg["clients"]:
        # Gradual linear ramp over rounds 21-40
        if round_num <= 40:
            fraction = drift_scale * ((round_num - 20) / 20.0)
        else:
            fraction = drift_scale

        n_replace = int(len(y_data) * fraction)
        if n_replace > 0:
            x_res = x_data.copy()
            y_res = y_data.copy()
            # Invert labels for replaced samples
            replace_indices = rng.choice(len(y_data), size=n_replace, replace=False)
            y_res[replace_indices] = 1 - y_res[replace_indices]
            return x_res, y_res

    # Phase 2: Covariate shift (rounds 61-100, clients 3, 7, 12, 16, 19)
    cov_cfg = schedule["covariate_shift"]
    c_rounds = cov_cfg["rounds"]
    if c_rounds[0] <= round_num <= c_rounds[1] and client_id in cov_cfg["clients"]:
        # Sudden onset
        n_replace = int(len(x_data) * drift_scale)
        if n_replace > 0:
            x_res = x_data.copy()
            y_res = y_data.copy()
            replace_indices = rng.choice(len(x_data), size=n_replace, replace=False)
            # Covariate feature shift: shift feature distribution by adding Gaussian perturbation
            noise = rng.normal(loc=1.0, scale=0.5, size=(n_replace, x_data.shape[1])).astype(np.float32)
            x_res[replace_indices] += noise
            return x_res, y_res

    return x_data, y_data

