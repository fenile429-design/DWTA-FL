"""
Evaluation Metrics for DWTA-FL.

Includes:
  - Accuracy, Precision, Recall, F1 (weighted by class support)
  - False Positive Rate (FPR): FP / (FP + TN)
  - Attack Resistance Rate (ARR) (Eq. 29):
      ARR = (N_resisted / N_attempts) * 100%
  - Adaptive Response Efficiency (ARE) (Eq. 30):
      ARE = ∑_{m=1}^M S_m * I_m / ∑_{m=1}^M c_{a_m}
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary classification metrics:
      - Accuracy (%)
      - Precision (%) [weighted]
      - Recall (%) [weighted]
      - F1 (%) [weighted]
      - FPR (%): FP / (FP + TN) * 100%
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0

    # Binary confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
    }


def compute_arr(
    num_resisted: int,
    num_attempts: int,
) -> float:
    """
    Compute Attack Resistance Rate (ARR) (Eq. 29).

    ARR = (N_resisted / N_attempts) * 100%
    """
    if num_attempts == 0:
        return 100.0
    return float((num_resisted / num_attempts) * 100.0)


def compute_are(
    severities: List[float],
    is_correct: List[bool],
    action_costs: List[float],
) -> float:
    """
    Compute Adaptive Response Efficiency (ARE) (Eq. 30).

    ARE = ∑_{m=1}^M S_m * I_m / ∑_{m=1}^M c_{a_m}

    Args:
        severities: Severity S_m of each event m
        is_correct: I_m = 1 if correct response, 0 otherwise
        action_costs: c_{a_m} cost of selected action
    """
    numerator = sum(s * (1.0 if c else 0.0) for s, c in zip(severities, is_correct))
    denominator = sum(action_costs)

    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)
