#!/usr/bin/env python3
"""
Canonical tie-aware Weighted AREA utilities.

Exact expression ties receive average expression ranks rather than arbitrary
sample-order ranks.

For phenotype score w_i and average expression rank r_i:

    C = sum_i (w_i - mean(w)) * (r_i - mean(r))

Canonical tie-aware AREA:

    ES = -2 * C / (n * sum_i w_i)

for nonnegative phenotype weights with positive total.

Under random permutation of phenotype scores relative to fixed ranks:

    Var(C) = sum(w_centered^2) * sum(r_centered^2) / (n - 1)

This automatically accounts for expression ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import erfc


@dataclass(frozen=True)
class TieAwareAreaResult:
    es: float
    linear_rank_statistic: float
    z: float
    pvalue_gaussian: float
    n: int
    n_unique_expression: int
    fraction_samples_in_ties: float
    max_tie_group_size: int


def average_expression_ranks(expression):
    expression = np.asarray(expression, dtype=float)
    if expression.ndim != 1:
        raise ValueError("expression must be one-dimensional.")
    if np.any(~np.isfinite(expression)):
        raise ValueError("expression contains non-finite values.")
    return stats.rankdata(expression, method="average").astype(float)


def tie_summary(expression):
    expression = np.asarray(expression, dtype=float)
    _, counts = np.unique(expression, return_counts=True)
    tied = counts[counts > 1]
    return {
        "n_unique_expression": int(len(counts)),
        "fraction_samples_in_ties": (
            float(np.sum(tied) / len(expression)) if len(tied) else 0.0
        ),
        "max_tie_group_size": int(np.max(tied)) if len(tied) else 1,
    }


def centered_linear_rank_statistic(expression, phenotype_scores):
    expression = np.asarray(expression, dtype=float)
    w = np.asarray(phenotype_scores, dtype=float)

    if expression.shape != w.shape:
        raise ValueError("expression and phenotype_scores must match.")
    if np.any(~np.isfinite(w)):
        raise ValueError("phenotype_scores contains non-finite values.")

    ranks = average_expression_ranks(expression)
    wc = w - np.mean(w)
    rc = ranks - np.mean(ranks)
    c = float(np.dot(wc, rc))
    return c, ranks


def permutation_variance_linear_rank(ranks, phenotype_scores):
    ranks = np.asarray(ranks, dtype=float)
    w = np.asarray(phenotype_scores, dtype=float)

    if ranks.shape != w.shape:
        raise ValueError("ranks and phenotype_scores must match.")

    n = len(w)
    if n < 2:
        raise ValueError("Need at least two observations.")

    wc = w - np.mean(w)
    rc = ranks - np.mean(ranks)

    var_c = float(np.dot(wc, wc) * np.dot(rc, rc) / (n - 1))

    if not np.isfinite(var_c) or var_c <= 0:
        raise ValueError("Permutation variance is non-positive.")

    return var_c


def tie_aware_area_es(expression, phenotype_scores):
    expression = np.asarray(expression, dtype=float)
    w = np.asarray(phenotype_scores, dtype=float)

    if expression.shape != w.shape:
        raise ValueError("expression and phenotype_scores must match.")
    if np.any(~np.isfinite(w)):
        raise ValueError("phenotype_scores contains non-finite values.")
    if np.min(w) < -1e-12:
        raise ValueError(
            "AREA geometry requires nonnegative scores. "
            "Use a positive affine recoding for the curve."
        )

    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("Phenotype scores must have positive total.")

    c, _ = centered_linear_rank_statistic(expression, w)
    n = len(w)
    return float(-2.0 * c / (n * total))


def tie_aware_area_test(expression, phenotype_scores):
    expression = np.asarray(expression, dtype=float)
    w = np.asarray(phenotype_scores, dtype=float)

    c, ranks = centered_linear_rank_statistic(expression, w)
    var_c = permutation_variance_linear_rank(ranks, w)

    z = float(-c / math.sqrt(var_c))
    p = float(erfc(abs(z) / math.sqrt(2.0)))
    es = tie_aware_area_es(expression, w)
    ties = tie_summary(expression)

    return TieAwareAreaResult(
        es=es,
        linear_rank_statistic=c,
        z=z,
        pvalue_gaussian=p,
        n=len(w),
        n_unique_expression=ties["n_unique_expression"],
        fraction_samples_in_ties=ties["fraction_samples_in_ties"],
        max_tie_group_size=ties["max_tie_group_size"],
    )


def tie_block_curve(expression, phenotype_scores):
    """
    Sample-order-invariant cumulative curve at unique-expression block boundaries.
    Within an exact-expression tie block, the curve is interpreted linearly.
    """
    expression = np.asarray(expression, dtype=float)
    w = np.asarray(phenotype_scores, dtype=float)

    if expression.shape != w.shape:
        raise ValueError("expression and phenotype_scores must match.")
    if np.min(w) < -1e-12:
        raise ValueError("Curve construction requires nonnegative scores.")

    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("Phenotype score total must be positive.")

    order = np.argsort(expression, kind="mergesort")
    x_sorted = expression[order]
    w_sorted = w[order]

    unique, first, counts = np.unique(
        x_sorted, return_index=True, return_counts=True
    )
    block_weight = np.add.reduceat(w_sorted, first)

    cum_n = np.cumsum(counts)
    cum_w = np.cumsum(block_weight)

    x = np.concatenate(([0.0], cum_n / len(expression)))
    y = np.concatenate(([0.0], cum_w / total))
    return x, y, unique


def curve_area_es(expression, phenotype_scores):
    x, y, _ = tie_block_curve(expression, phenotype_scores)
    area_curve = float(np.trapezoid(y, x))
    return float(2.0 * (area_curve - 0.5))
