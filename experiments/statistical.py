"""
Statistical Significance Analysis for DWTA-FL (Section 4.10, Table 20).

Computes paired t-tests and Wilcoxon signed-rank tests comparing DWTA-FL
against every baseline on paired seeds. All scores must be supplied by the
caller (measured runs); no values are stored in this module.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size for paired samples."""
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def compute_statistical_tests(
    dwta_scores: List[float],
    baseline_scores: List[float],
    baseline_name: str,
) -> Dict[str, Any]:
    """
    Perform paired t-test and Wilcoxon signed-rank test.
    """
    x = np.asarray(dwta_scores, dtype=float)
    y = np.asarray(baseline_scores, dtype=float)

    t_stat, t_pval = stats.ttest_rel(x, y)

    try:
        w_stat, w_pval = stats.wilcoxon(x, y)
    except ValueError:
        w_stat, w_pval = float("nan"), float("nan")

    return {
        "Baseline": baseline_name,
        "DWTA-FL Mean": float(np.mean(x)),
        "Baseline Mean": float(np.mean(y)),
        "t-statistic": float(t_stat),
        "t-test p-value": float(t_pval),
        "Wilcoxon W": float(w_stat),
        "Wilcoxon p-value": float(w_pval),
        "Cohen's d": cohens_d(x, y),
        "Significant (p < 0.05)": bool(t_pval < 0.05),
    }


def run_statistical_analysis(
    paired_data: Dict[str, List[float]],
    reference: str = "DWTA-FL",
) -> List[Dict[str, Any]]:
    """
    Compare `reference` against every other method.

    Args:
        paired_data: {method_name: [per-seed score, ...]} with identical seed order.
    """
    rows = []
    for name, scores in paired_data.items():
        if name == reference:
            continue
        rows.append(compute_statistical_tests(paired_data[reference], scores, name))
    return rows
