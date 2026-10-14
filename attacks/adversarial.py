"""
Adversarial Attack Generators for Federated Learning (Table S5).

Implemented attacks:
  1. Label Poisoning: Flips labels (0 ↔ 1) on 20% of local samples.
  2. Sign Flipping: Reverses model update direction with scale = 3.0.
  3. Backdoor: Injects trigger features on 10% of local rows; target class = benign (0).
  4. Byzantine: Random Gaussian update scaled to 3× legitimate update L2 norm.
  5. Adaptive Byzantine: Malicious direction with norm ≤ 3× benign median norm.
  6. Collaborative Collusion: Shared adversarial update across colluding clients; scale = 3.0.
  7. Covert Gradient Escape: Targeted update with ≤ 15% norm deviation from benign median.
  8. Extreme Stress: Coordinated manipulation with up to 50% compromised clients.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mlp import unflatten_weights, flatten_weights


def apply_label_poisoning(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float = config.LABEL_POISON_FRACTION,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply label poisoning attack.
    Flips binary labels (0 ↔ 1) on fraction (default 20%) of local samples.
    """
    if fraction <= 0.0:
        return x, y

    rng = np.random.default_rng(seed)
    n = len(y)
    num_flip = int(n * fraction)
    flip_indices = rng.choice(n, size=num_flip, replace=False)

    y_poisoned = y.copy()
    y_poisoned[flip_indices] = 1 - y_poisoned[flip_indices]
    return x, y_poisoned


def apply_backdoor(
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    feature_names: List[str],
    fraction: float = config.BACKDOOR_TRIGGER_FRACTION,
    target_class: int = config.BACKDOOR_TARGET_CLASS,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply backdoor attack (Table S5).
    Injects trigger features into 10% of local rows and assigns target class (0 = benign).
    """
    if fraction <= 0.0:
        return x, y

    rng = np.random.default_rng(seed)
    n = len(y)
    num_inject = int(n * fraction)
    inject_indices = rng.choice(n, size=num_inject, replace=False)

    x_backdoor = x.copy()
    y_backdoor = y.copy()
    y_backdoor[inject_indices] = target_class

    # Get dataset specific trigger config
    triggers = config.BACKDOOR_TRIGGERS.get(dataset_name, {})
    for col_name, trigger_val in triggers.items():
        if col_name in feature_names:
            col_idx = feature_names.index(col_name)
            # If numerical trigger, set to float value
            try:
                val = float(trigger_val)
                x_backdoor[inject_indices, col_idx] = val
            except (ValueError, TypeError):
                # If categorical or zeroed
                x_backdoor[inject_indices, col_idx] = 0.0

    return x_backdoor, y_backdoor


def apply_sign_flipping(
    honest_delta_flat: np.ndarray,
    scale: float = config.SIGN_FLIP_SCALE,
) -> np.ndarray:
    """
    Apply sign flipping attack:
    Reverses update direction: Δw_mal = -scale * Δw_honest (scale = 3.0).
    """
    return -scale * honest_delta_flat


def apply_byzantine_attack(
    benign_norm: float,
    dim: int,
    norm_scale: float = config.BYZANTINE_NORM_SCALE,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Byzantine attack:
    Random Gaussian vector scaled to 3× legitimate update L2 norm.
    """
    rng = np.random.default_rng(seed)
    random_vec = rng.standard_normal(dim)
    r_norm = np.linalg.norm(random_vec)
    if r_norm > 0:
        random_vec = random_vec / r_norm
    return random_vec * (norm_scale * benign_norm)


def apply_adaptive_byzantine(
    benign_mean_delta: np.ndarray,
    benign_median_norm: float,
    norm_scale: float = config.ADAPTIVE_BYZANTINE_NORM_SCALE,
) -> np.ndarray:
    """
    Apply Adaptive Byzantine attack:
    Inverts the benign direction with norm bounded by 3× benign median norm.
    """
    direction = -benign_mean_delta
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 0:
        direction = direction / dir_norm
    target_norm = min(norm_scale * benign_median_norm, dir_norm * norm_scale)
    return direction * target_norm


def apply_collaborative_collusion(
    benign_mean_delta: np.ndarray,
    scale: float = config.COLLUSION_SCALE,
) -> np.ndarray:
    """
    Apply Collaborative Collusion attack:
    All colluding clients submit a coordinated adversarial update:
    reversing benign consensus with scale = 3.0.
    """
    return -scale * benign_mean_delta


def apply_covert_gradient_escape(
    benign_delta: np.ndarray,
    benign_median_norm: float,
    max_deviation: float = config.COVERT_NORM_DEVIATION,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply Covert Gradient Escape attack:
    Targeted update designed to evade norm detectors with ≤ 15% norm deviation
    from the benign update norm.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(benign_delta))
    noise_norm = np.linalg.norm(noise)
    if noise_norm > 0:
        noise = noise / noise_norm

    # Perturbed update
    perturbed = -0.5 * benign_delta + 0.5 * noise * benign_median_norm
    p_norm = np.linalg.norm(perturbed)
    if p_norm > 0:
        perturbed = perturbed / p_norm

    # Scale to within (1 ± max_deviation) * benign_median_norm
    factor = 1.0 + rng.uniform(-max_deviation, max_deviation)
    return perturbed * (benign_median_norm * factor)
