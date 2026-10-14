"""
Cost-Aware Mitigation Selection for DWTA-FL (Section 3.9, Equations 21-23).

Computes action utility, selects the optimal response, and updates adaptive weights:
  - Action set: {Monitor, Access Restriction, Isolation, Quarantine}
  - Eq. 21: J_i^t(a) = λ_S^t * S_i^t * e_a + λ_P^t * PR_i^t * p_a + λ_R^t * T_i^t * r_a - λ_C^t * c_a
  - Eq. 22: a_i^{t*} = argmax_{a ∈ A} J_i^t(a)
  - Eq. 23: λ_k^{t+1} = λ_k^t * exp(ζ * Δ_k^t) / ∑_{j} λ_j^t * exp(ζ * Δ_j^t)
    with adaptation rate ζ = 0.01
"""

import numpy as np
from typing import Dict, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


ACTIONS = ["Monitor", "Access Restriction", "Isolation", "Quarantine"]


class MitigationOrchestrator:
    """
    Manages mitigation action selection and adaptive weights per cloud domain.
    """

    def __init__(
        self,
        domain_id: int,
        zeta: float = config.ORCHESTRATION_ADAPTATION_ZETA,
        action_coefficients: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.domain_id = domain_id
        self.zeta = zeta
        self.action_coefficients = action_coefficients or config.ACTION_COEFFICIENTS

        # Initial weights (Eq. 23 starting point)
        self.weights = {
            "S": float(config.ORCHESTRATION_LAMBDA_S),
            "P": float(config.ORCHESTRATION_LAMBDA_P),
            "R": float(config.ORCHESTRATION_LAMBDA_R),
            "C": float(config.ORCHESTRATION_LAMBDA_C),
        }

        # History tracking for deltas
        self.prev_severity: Optional[float] = None
        self.prev_propagation: Optional[float] = None
        self.prev_trust: Optional[float] = None

    def compute_action_utility(
        self,
        action: str,
        severity: float,
        propagation_risk: float,
        trust: float,
    ) -> float:
        """
        Compute utility J_i^t(a) for a given action (Eq. 21).

        J_i^t(a) = λ_S * S_i^t * e_a + λ_P * PR_i^t * p_a + λ_R * T_i^t * r_a - λ_C * c_a
        """
        coeffs = self.action_coefficients[action]
        ea = coeffs["severity_gain"]
        pa = coeffs["propagation_reduction"]
        ra = coeffs["suitability"]
        ca = coeffs["cost"]

        utility = (
            self.weights["S"] * severity * ea
            + self.weights["P"] * propagation_risk * pa
            + self.weights["R"] * trust * ra
            - self.weights["C"] * ca
        )
        return float(utility)

    def select_action(
        self,
        severity: float,
        propagation_risk: float,
        trust: float,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select optimal mitigation action a_i^{t*} (Eq. 22).

        Returns:
            (selected_action, utilities_dict)
        """
        utilities = {}
        for action in ACTIONS:
            utilities[action] = self.compute_action_utility(
                action, severity, propagation_risk, trust
            )

        best_action = max(ACTIONS, key=lambda a: utilities[a])
        return best_action, utilities

    def update_weights(
        self,
        severity: float,
        propagation_risk: float,
        trust: float,
    ) -> Dict[str, float]:
        """
        Update adaptive weights λ_k^{t+1} from changes in TSV components (Eq. 23).

        Δ_S^t = S_i^t - S_i^{t-1}
        Δ_P^t = PR_i^t - PR_i^{t-1}
        Δ_R^t = T_i^t - T_i^{t-1}
        Δ_C^t = 0
        """
        if self.prev_severity is None:
            delta_s = 0.0
            delta_p = 0.0
            delta_r = 0.0
        else:
            delta_s = severity - self.prev_severity
            delta_p = propagation_risk - self.prev_propagation
            delta_r = trust - self.prev_trust

        delta_c = 0.0
        deltas = {"S": delta_s, "P": delta_p, "R": delta_r, "C": delta_c}

        # Numerator: λ_k^t * exp(ζ * Δ_k^t)
        new_unnorm = {}
        for k in ["S", "P", "R", "C"]:
            new_unnorm[k] = self.weights[k] * np.exp(self.zeta * deltas[k])

        total = sum(new_unnorm.values())
        if total > 0:
            for k in ["S", "P", "R", "C"]:
                self.weights[k] = float(new_unnorm[k] / total)

        # Update previous values
        self.prev_severity = severity
        self.prev_propagation = propagation_risk
        self.prev_trust = trust

        return self.weights.copy()
