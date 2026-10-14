"""
Drift Measurement for DWTA-FL (Section 3.3, Equations 3-5).

The server measures drift on the distribution of each client's update using
Jensen-Shannon divergence between consecutive rounds.

Key equations:
  - Eq. 3: D_i^t = JS(P_i^t || P_i^{t-1})  (JSD of update histograms)
  - Eq. 4: σ_i^t = |D_i^t - D̂_i^{t-1}|  (drift uncertainty)
  - Eq. 5: D̂_i^t = (1 - σ_i^t)[λ D_i^t + (1-λ) D̂_i^{t-1}]  (smoothed drift)
"""

import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_update_histogram(
    update_flat: np.ndarray,
    num_bins: int = config.DRIFT_HISTOGRAM_BINS,
    bin_range: tuple = config.DRIFT_BIN_RANGE,
) -> np.ndarray:
    """
    Map an update vector to a normalised B-bin histogram h(·).

    Values outside the range are placed in the end bins.
    The histogram is normalised to sum to 1 (probability distribution).

    Args:
        update_flat: Flattened update vector
        num_bins: Number of bins (B=30)
        bin_range: Bin edges range [0, 1]

    Returns:
        Normalised histogram P_i^t of shape (num_bins,)
    """
    # Clip values to bin range (values outside go to end bins)
    hist, _ = np.histogram(
        update_flat,
        bins=num_bins,
        range=bin_range,
    )

    # Normalise to probability distribution
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    else:
        # Uniform distribution if all zeros
        hist = np.ones(num_bins, dtype=np.float64) / num_bins

    # Add small epsilon to avoid log(0) in KL divergence
    hist = hist + 1e-12
    hist = hist / hist.sum()

    return hist


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence KL(P || Q) with base-2 logarithm.

    Args:
        p: Probability distribution P
        q: Probability distribution Q

    Returns:
        KL(P || Q)
    """
    # Avoid log(0) by adding small epsilon
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float(np.sum(p * np.log2(p / q)))


def jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    """
    Compute Jensen-Shannon divergence (Eq. 3).

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Bounded: 0 ≤ JS ≤ 1 (with base-2 log).

    Args:
        p: Distribution P_i^t
        q: Distribution P_i^{t-1}

    Returns:
        Jensen-Shannon divergence D_i^t
    """
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return np.clip(jsd, 0.0, 1.0)


def compute_drift_uncertainty(
    raw_drift: float,
    smoothed_drift_prev: float,
) -> float:
    """
    Compute drift uncertainty (Eq. 4).

    σ_i^t = |D_i^t - D̂_i^{t-1}|

    Bounded: 0 ≤ σ_i^t ≤ 1

    Args:
        raw_drift: D_i^t (current raw drift)
        smoothed_drift_prev: D̂_i^{t-1} (previous smoothed drift)

    Returns:
        Drift uncertainty σ_i^t
    """
    return np.clip(abs(raw_drift - smoothed_drift_prev), 0.0, 1.0)


def compute_smoothed_drift(
    raw_drift: float,
    smoothed_drift_prev: float,
    drift_uncertainty: float,
    lambda_smooth: float = config.DRIFT_SMOOTHING_LAMBDA,
) -> float:
    """
    Compute smoothed drift (Eq. 5).

    D̂_i^t = (1 - σ_i^t) * [λ * D_i^t + (1-λ) * D̂_i^{t-1}]

    A large, unconfirmed jump (large σ) is down-weighted;
    a sustained shift raises D̂ over several rounds.

    Args:
        raw_drift: D_i^t
        smoothed_drift_prev: D̂_i^{t-1}
        drift_uncertainty: σ_i^t
        lambda_smooth: λ = 0.7

    Returns:
        Smoothed drift D̂_i^t
    """
    weighted_drift = lambda_smooth * raw_drift + (1 - lambda_smooth) * smoothed_drift_prev
    smoothed = (1 - drift_uncertainty) * weighted_drift
    return np.clip(smoothed, 0.0, 1.0)


class DriftTracker:
    """
    Tracks drift measurements for all clients across rounds.

    Maintains:
        - Previous update histograms P_i^{t-1}
        - Smoothed drift D̂_i^t (initialized to 0)
    """

    def __init__(
        self,
        num_clients: int = config.NUM_CLIENTS,
        num_bins: int = config.DRIFT_HISTOGRAM_BINS,
        bin_range: tuple = config.DRIFT_BIN_RANGE,
        lambda_smooth: float = config.DRIFT_SMOOTHING_LAMBDA,
    ):
        self.num_clients = num_clients
        self.num_bins = num_bins
        self.bin_range = bin_range
        self.lambda_smooth = lambda_smooth

        # State: previous histograms and smoothed drifts
        self.prev_histograms = {}   # client_id -> P_i^{t-1}
        self.smoothed_drifts = {}   # client_id -> D̂_i^t
        self.raw_drifts = {}        # client_id -> D_i^t
        self.drift_uncertainties = {}  # client_id -> σ_i^t

        # Initialize smoothed drift to 0 for all clients
        for i in range(num_clients):
            self.smoothed_drifts[i] = 0.0

    def update(
        self,
        client_id: int,
        update_flat: np.ndarray,
    ) -> dict:
        """
        Compute drift for a client in the current round.

        Args:
            client_id: Client index
            update_flat: Flattened model update Δw_i^t

        Returns:
            Dictionary with drift metrics:
                raw_drift, drift_uncertainty, smoothed_drift
        """
        # Compute current histogram
        current_hist = compute_update_histogram(
            update_flat, self.num_bins, self.bin_range
        )

        if client_id not in self.prev_histograms:
            # First round: no drift (D̂_i^0 = 0)
            self.prev_histograms[client_id] = current_hist
            result = {
                "raw_drift": 0.0,
                "drift_uncertainty": 0.0,
                "smoothed_drift": 0.0,
            }
        else:
            # Compute raw drift (Eq. 3)
            raw_drift = jensen_shannon_divergence(
                current_hist, self.prev_histograms[client_id]
            )

            # Compute drift uncertainty (Eq. 4)
            drift_unc = compute_drift_uncertainty(
                raw_drift, self.smoothed_drifts[client_id]
            )

            # Compute smoothed drift (Eq. 5)
            smoothed = compute_smoothed_drift(
                raw_drift,
                self.smoothed_drifts[client_id],
                drift_unc,
                self.lambda_smooth,
            )

            # Update state
            self.prev_histograms[client_id] = current_hist
            self.smoothed_drifts[client_id] = smoothed
            self.raw_drifts[client_id] = raw_drift
            self.drift_uncertainties[client_id] = drift_unc

            result = {
                "raw_drift": raw_drift,
                "drift_uncertainty": drift_unc,
                "smoothed_drift": smoothed,
            }

        return result

    def get_smoothed_drift(self, client_id: int) -> float:
        """Get current smoothed drift for a client."""
        return self.smoothed_drifts.get(client_id, 0.0)

    def get_drift_uncertainty(self, client_id: int) -> float:
        """Get current drift uncertainty for a client."""
        return self.drift_uncertainties.get(client_id, 0.0)
