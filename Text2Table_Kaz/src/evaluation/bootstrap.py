"""
Bootstrap confidence intervals and significance testing.

Implements Section III-D-2 of the paper:
  - Paired bootstrap test (B=10,000 iterations, alpha=0.05)
  - 95% CI via percentile method (Efron & Tibshirani, 1993)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_bootstrap: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}--{self.ci_upper:.3f}]"


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: list of metric scores.
        n_bootstrap: number of bootstrap resamples (paper: 10,000).
        alpha: significance level (paper: 0.05, giving 95% CI).
        seed: random seed for reproducibility.

    Returns:
        BootstrapResult with mean, CI bounds, and std.
    """
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, 0)

    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1

    observed_mean = sum(values) / n
    variance = sum((x - observed_mean) ** 2 for x in values) / max(n - 1, 1)

    return BootstrapResult(
        mean=observed_mean,
        ci_lower=means[lo_idx],
        ci_upper=means[hi_idx],
        std=math.sqrt(variance),
        n_bootstrap=n_bootstrap,
    )


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Paired bootstrap significance test.

    Tests H0: mean(A) == mean(B) against H1: mean(A) != mean(B).

    Returns:
        p-value (two-tailed).
    """
    assert len(scores_a) == len(scores_b), "Score lists must have equal length"
    rng = random.Random(seed)
    n = len(scores_a)

    observed_diff = sum(scores_a) / n - sum(scores_b) / n

    diffs = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in indices) / n
        mean_b = sum(scores_b[i] for i in indices) / n
        diffs.append(mean_a - mean_b)

    # Two-tailed p-value
    extreme = sum(1 for d in diffs if abs(d) >= abs(observed_diff))
    return extreme / n_bootstrap
