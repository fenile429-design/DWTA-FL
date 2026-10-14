"""
Client Implementation for DWTA-FL (Section 3.2, Algorithm 1).

Clients:
  - Hold local data D_i (partitioned non-IID data)
  - Train locally for E=5 epochs with mini-batch size B=64 using Adam (lr=0.001)
  - Send only the model update Δw_i^t = w_i^{t+1} - w_t
  - Do NOT compute or send trust, drift, consistency, or volatility
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mlp import (
    create_mlp,
    get_model_weights,
    set_model_weights,
    flatten_weights,
    compute_model_update,
    compute_update_direction,
)


class FLClient:
    """
    Simulates a Cloud Service Provider (CSP) client node in DWTA-FL.
    """

    def __init__(
        self,
        client_id: int,
        input_dim: int,
        x_data: np.ndarray,
        y_data: np.ndarray,
        local_epochs: int = config.LOCAL_EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        seed: Optional[int] = None,
    ):
        self.client_id = client_id
        self.input_dim = input_dim
        self.x_data = x_data
        self.y_data = y_data
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        # Local model instance
        self.model = create_mlp(
            input_dim=input_dim,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.weight_shapes = [w.shape for w in self.model.get_weights()]

    def set_data(self, x: np.ndarray, y: np.ndarray):
        """Update local window data (e.g. for drift injection or windowing)."""
        self.x_data = x
        self.y_data = y

    def train(
        self,
        global_weights: List[np.ndarray],
    ) -> Dict[str, object]:
        """
        Execute local training round.

        1. Set local model weights to global_weights (w_t)
        2. Train on local data D_i(t) for E epochs
        3. Compute Δw_i^t = w_i^{t+1} - w_t
        4. Return update and metadata

        Returns:
            Dictionary with:
                - client_id
                - delta_weights: List of np.ndarray
                - delta_flat: Flattened Δw_i^t
                - direction_flat: Flattened g_i^t = -Δw_i^t / η
                - num_samples: Number of training samples
                - train_loss: Final training loss
                - train_acc: Final training accuracy
        """
        set_model_weights(self.model, global_weights)

        # Train locally
        history = self.model.fit(
            self.x_data,
            self.y_data,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=True,
        )

        local_weights = get_model_weights(self.model)
        delta_weights = compute_model_update(global_weights, local_weights)
        delta_flat = flatten_weights(delta_weights)

        # Effective direction g_i^t = -Δw_i^t / η_t
        direction_flat = -delta_flat / self.learning_rate

        final_loss = history.history["loss"][-1] if "loss" in history.history else 0.0
        final_acc = history.history["accuracy"][-1] if "accuracy" in history.history else 0.0

        return {
            "client_id": self.client_id,
            "delta_weights": delta_weights,
            "delta_flat": delta_flat,
            "direction_flat": direction_flat,
            "num_samples": len(self.x_data),
            "train_loss": float(final_loss),
            "train_acc": float(final_acc),
        }

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """Evaluate local model on given test set."""
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        return float(loss), float(acc)
