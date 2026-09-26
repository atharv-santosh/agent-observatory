"""The statistics, written out rather than imported.

A regression view that shows two bars and leaves you to squint is not a regression view. What is
needed is a p-value for "did this failure cluster actually get more common", a confidence interval
on how much, and a correction for the fact that asking the question of fifteen clusters at once will
hand you a significant answer by luck.

Permutation rather than a normal approximation because the interesting clusters are the rare ones,
where a chi-square is exactly where it is least trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ProportionTest:
    """Result of comparing one cluster's prevalence between two populations."""

    n_a: int
    k_a: int
    n_b: int
    k_b: int
    p_value: float
    risk_ratio: float
    rr_ci: tuple[float, float]
    diff: float
    diff_ci: tuple[float, float]
    n_permutations: int

    @property
    def rate_a(self) -> float:
        return self.k_a / self.n_a if self.n_a else 0.0

    @property
    def rate_b(self) -> float:
        return self.k_b / self.n_b if self.n_b else 0.0


def permutation_proportion_test(k_a: int, n_a: int, k_b: int, n_b: int, *,
                               n_permutations: int = 20000, seed: int = 0,
                               two_sided: bool = True) -> float:
    """Exact-in-the-limit p-value for a difference in two proportions, by relabelling.

    Under the null the two populations are one pool, so shuffling which run belongs to which
    population and recomputing the difference gives the null distribution directly.
    """
    if n_a == 0 or n_b == 0:
        return 1.0
    observed = k_b / n_b - k_a / n_a
    total, successes = n_a + n_b, k_a + k_b
    if successes == 0 or successes == total:
        return 1.0
    rng = np.random.default_rng(seed)
    # Shuffling a 0/1 pool is a hypergeometric draw, so sample the count directly instead of
    # permuting an array 20000 times.
    draws = rng.hypergeometric(successes, total - successes, n_b, size=n_permutations)
    null = draws / n_b - (successes - draws) / n_a
    if two_sided:
        hits = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    else:
        hits = int(np.sum(null >= observed - 1e-12))
    return (hits + 1) / (n_permutations + 1)


def bootstrap_ratio_ci(k_a: int, n_a: int, k_b: int, n_b: int, *, n_boot: int = 10000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[tuple[float, float],
                                                                    tuple[float, float]]:
    """Percentile bootstrap CIs for (risk ratio, rate difference)."""
    if n_a == 0 or n_b == 0:
        return ((float("nan"), float("nan")), (float("nan"), float("nan")))
    rng = np.random.default_rng(seed + 7)
    pa = rng.binomial(n_a, k_a / n_a, size=n_boot) / n_a
    pb = rng.binomial(n_b, k_b / n_b, size=n_boot) / n_b
    # Haldane-Anscombe style continuity term so an all-zero baseline gives a finite bound.
    eps = 0.5 / max(n_a, n_b)
    ratios = (pb + eps) / (pa + eps)
    diffs = pb - pa
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return ((float(np.percentile(ratios, lo)), float(np.percentile(ratios, hi))),
            (float(np.percentile(diffs, lo)), float(np.percentile(diffs, hi))))


def compare_proportions(k_a: int, n_a: int, k_b: int, n_b: int, *, n_permutations: int = 20000,
                        seed: int = 0) -> ProportionTest:
    """Full two-proportion comparison: permutation p-value plus bootstrap effect size."""
    p = permutation_proportion_test(k_a, n_a, k_b, n_b, n_permutations=n_permutations, seed=seed)
    rr_ci, diff_ci = bootstrap_ratio_ci(k_a, n_a, k_b, n_b, seed=seed)
    rate_a = k_a / n_a if n_a else 0.0
    rate_b = k_b / n_b if n_b else 0.0
    eps = 0.5 / max(n_a or 1, n_b or 1)
    return ProportionTest(
        n_a=n_a, k_a=k_a, n_b=n_b, k_b=k_b, p_value=p,
        risk_ratio=(rate_b + eps) / (rate_a + eps),
        rr_ci=rr_ci, diff=rate_b - rate_a, diff_ci=diff_ci, n_permutations=n_permutations)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """BH step-up: returns (adjusted p-values, rejected flags) in the input order."""
    m = len(p_values)
    if m == 0:
        return ([], [])
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[idx] * m / rank)
        adjusted[idx] = min(running, 1.0)
    return (adjusted, [adjusted[i] <= alpha for i in range(m)])


def wilson_interval(k: int, n: int, *, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, for reporting a precision without lying about small samples."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
