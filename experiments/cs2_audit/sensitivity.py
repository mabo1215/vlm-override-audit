"""Cluster-aware uncertainty and placebo sensitivity analysis.

Referring expressions that share an image are not independent Bernoulli
trials (docs/revision_suggestions.tex P0-4), so every interval and placebo
here resamples or permutes whole images, never individual expression rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .audit import IOU_THRESHOLD, Policy, validate_prediction_frame


def _require_single_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    validate_prediction_frame(frame)
    if frame["split"].nunique() != 1 or frame["split"].iat[0] != split:
        raise ValueError(f"expected a single split named {split!r}")
    return frame.reset_index(drop=True)


def _net_accuracy_gain(
    frame: pd.DataFrame, routed: np.ndarray, iou_threshold: float
) -> float:
    baseline_correct = frame["baseline_iou"].to_numpy(dtype=float) >= iou_threshold
    challenger_correct = frame["challenger_iou"].to_numpy(dtype=float) >= iou_threshold
    help_count = int((routed & ~baseline_correct & challenger_correct).sum())
    harm_count = int((routed & baseline_correct & ~challenger_correct).sum())
    return (help_count - harm_count) / len(frame)


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile bootstrap interval for one statistic."""

    point_estimate: float
    mean: float
    lower: float
    upper: float
    alpha: float
    n_bootstrap: int
    cluster_unit: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "point_estimate": self.point_estimate,
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "cluster_unit": self.cluster_unit,
            "seed": self.seed,
        }


def image_cluster_bootstrap(
    frame: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapInterval:
    """Percentile bootstrap that resamples whole images with replacement.

    ``statistic_fn`` maps a (possibly resampled, duplicate-index) DataFrame
    to a single float.  Resampling images rather than rows keeps every
    expression from one image together in every replicate.
    """

    validate_prediction_frame(frame)
    image_ids = frame["image_id"].unique()
    if len(image_ids) == 0:
        raise ValueError("cannot bootstrap an empty manifest")

    rng = np.random.default_rng(seed)
    point_estimate = float(statistic_fn(frame))
    grouped = {image_id: sub for image_id, sub in frame.groupby("image_id")}

    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        chosen = rng.choice(image_ids, size=len(image_ids), replace=True)
        resampled = pd.concat([grouped[image_id] for image_id in chosen], ignore_index=True)
        samples[i] = float(statistic_fn(resampled))

    return BootstrapInterval(
        point_estimate=point_estimate,
        mean=float(samples.mean()),
        lower=float(np.quantile(samples, alpha / 2)),
        upper=float(np.quantile(samples, 1 - alpha / 2)),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        cluster_unit="image_id",
        seed=seed,
    )


def auroc_help_harm(
    frame: pd.DataFrame, *, iou_threshold: float = IOU_THRESHOLD
) -> dict[str, object]:
    """Diagnostic AUROC of ``score`` for distinguishing help vs. harm rows.

    Restricted to rows where baseline and challenger correctness differ,
    since help/harm are only defined there.  Ties are handled with mid-rank
    (average rank) scoring, the standard tie-corrected Mann-Whitney AUROC.
    """

    validate_prediction_frame(frame)
    baseline_correct = frame["baseline_iou"].to_numpy(dtype=float) >= iou_threshold
    challenger_correct = frame["challenger_iou"].to_numpy(dtype=float) >= iou_threshold
    help_mask = ~baseline_correct & challenger_correct
    harm_mask = baseline_correct & ~challenger_correct
    eligible = help_mask | harm_mask

    n_help = int(help_mask.sum())
    n_harm = int(harm_mask.sum())
    if n_help == 0 or n_harm == 0:
        return {
            "auroc": None,
            "n_help": n_help,
            "n_harm": n_harm,
            "reason": "need at least one help and one harm example",
        }

    scores = frame.loc[eligible, "score"].to_numpy(dtype=float)
    labels = help_mask[eligible].astype(int)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i : j + 1] = ranks_sorted[i : j + 1].mean()
        i = j + 1
    ranks = np.empty(len(scores))
    ranks[order] = ranks_sorted

    sum_ranks_positive = ranks[labels == 1].sum()
    auroc = (sum_ranks_positive - n_help * (n_help + 1) / 2) / (n_help * n_harm)
    return {"auroc": float(auroc), "n_help": n_help, "n_harm": n_harm}


def image_block_permutation_placebo(
    frame: pd.DataFrame,
    split: str,
    policy: Policy,
    *,
    n_repeats: int = 200,
    seed: int = 0,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, object]:
    """Placebo null: shuffle each image's mean score across images.

    The permutation unit is the image, not the expression: shuffling
    expression scores within one image barely perturbs a real signal and
    does not test cross-image association (docs/revision_suggestions.tex
    P0-8/E3), so every expression belonging to one image is reassigned
    another image's mean score as a single block.
    """

    working = _require_single_split(frame, split)
    observed_routed = policy.route(working["score"].to_numpy(dtype=float))
    observed_gain = _net_accuracy_gain(working, observed_routed, iou_threshold)

    image_ids = working["image_id"].to_numpy()
    unique_images = pd.unique(image_ids)
    image_mean_score = (
        working.groupby("image_id")["score"].mean().reindex(unique_images).to_numpy()
    )

    rng = np.random.default_rng(seed)
    null_gains = np.empty(n_repeats)
    for i in range(n_repeats):
        permuted_means = rng.permutation(image_mean_score)
        mapping = dict(zip(unique_images, permuted_means))
        permuted_scores = np.array([mapping[image_id] for image_id in image_ids])
        routed = policy.route(permuted_scores)
        null_gains[i] = _net_accuracy_gain(working, routed, iou_threshold)

    p_value = float((np.sum(null_gains >= observed_gain) + 1) / (n_repeats + 1))
    return {
        "permutation_unit": "image_id (block-shuffled image-mean score)",
        "n_repeats": n_repeats,
        "seed": seed,
        "policy": policy.name,
        "observed_net_accuracy_gain": observed_gain,
        "null_mean": float(null_gains.mean()),
        "null_ci95": [
            float(np.quantile(null_gains, 0.025)),
            float(np.quantile(null_gains, 0.975)),
        ],
        "p_value_one_sided_greater": p_value,
        "null_hypothesis": (
            "score carries no image-level information about the help/harm "
            "outcome beyond route mass"
        ),
    }
