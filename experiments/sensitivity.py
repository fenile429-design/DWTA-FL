"""
Hyperparameter Sensitivity Analysis (Section 4.12, Figures 8-11).

Performs a real one-at-a-time sweep of DWTA-FL hyperparameters. Sweep grids
are supplied by the caller; this module stores no result values.
"""

import os, sys
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from experiments.ablation import run_dwta_variant


# config attribute that holds the nominal value of each swept parameter
NOMINAL_ATTR = {
    "theta_d": "DRIFT_THRESHOLD_THETA_D",
    "beta": "TRUST_MEMORY_BETA",
    "gamma": "DRIFT_PENALTY_GAMMA",
    "tau": "VOLATILITY_PENALTY_TAU",
}


def run_sensitivity_sweep(
    grids: Dict[str, List[float]],
    dataset_name: str,
    seed: int,
    num_rounds: int,
    sample_limit: Optional[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sweep each parameter in `grids` while the others stay at config values.

    Returns:
        {param: [{"value": v, "metrics": {...}}, ...]}
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for param, values in grids.items():
        out[param] = []
        for v in values:
            metrics = run_dwta_variant(
                dataset_name=dataset_name,
                variant="full",
                overrides={param: v},
                seed=seed,
                num_rounds=num_rounds,
                sample_limit=sample_limit,
            )
            out[param].append({"value": v, "metrics": metrics})
    return out
