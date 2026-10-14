"""
Common MLP Model for DWTA-FL.

Architecture from Section 4.1.5:
  - Hidden layers: 512, 256, 128 (ReLU activation)
  - Output: 2-unit softmax
  - Optimizer: Adam (lr=0.001)
  - Loss: Sparse categorical cross-entropy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_mlp(
    input_dim: int,
    hidden_layers: List[int] = None,
    activation: str = None,
    output_units: int = None,
    learning_rate: float = None,
    seed: Optional[int] = None,
) -> keras.Model:
    """
    Create the common MLP used across all methods.

    Architecture: input → 512 ReLU → 256 ReLU → 128 ReLU → 2 softmax

    Args:
        input_dim: Number of input features
        hidden_layers: Hidden layer sizes [512, 256, 128]
        activation: Activation function ('relu')
        output_units: Number of output units (2)
        learning_rate: Adam learning rate (0.001)
        seed: Random seed for weight initialization

    Returns:
        Compiled Keras model
    """
    if hidden_layers is None:
        hidden_layers = config.MLP_HIDDEN_LAYERS
    if activation is None:
        activation = config.MLP_ACTIVATION
    if output_units is None:
        output_units = config.MLP_OUTPUT_UNITS
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    if seed is not None:
        tf.random.set_seed(seed)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
            name=f"hidden_{i+1}",
        ))

    model.add(keras.layers.Dense(
        output_units,
        activation="softmax",
        kernel_initializer=keras.initializers.GlorotUniform(seed=seed),
        name="output",
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_weights(model: keras.Model) -> List[np.ndarray]:
    """Get flattened model weights."""
    return model.get_weights()


def set_model_weights(model: keras.Model, weights: List[np.ndarray]):
    """Set model weights."""
    model.set_weights(weights)


def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of weight arrays into a single vector."""
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(
    flat_weights: np.ndarray,
    weight_shapes: List[tuple],
) -> List[np.ndarray]:
    """Unflatten a weight vector back into a list of arrays."""
    weights = []
    offset = 0
    for shape in weight_shapes:
        size = np.prod(shape)
        w = flat_weights[offset:offset + size].reshape(shape)
        weights.append(w)
        offset += size
    return weights


def compute_model_update(
    global_weights: List[np.ndarray],
    local_weights: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute model update: Δw_i^t = w_i^{t+1} - w_t  (Eq. 1)
    """
    return [lw - gw for lw, gw in zip(local_weights, global_weights)]


def compute_update_direction(
    update: List[np.ndarray],
    learning_rate: float = config.LEARNING_RATE,
) -> List[np.ndarray]:
    """
    Compute effective update direction: g_i^t = -Δw_i^t / η_t  (Eq. 1)
    """
    return [-u / learning_rate for u in update]


def count_parameters(model: keras.Model) -> int:
    """Count total trainable parameters."""
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
