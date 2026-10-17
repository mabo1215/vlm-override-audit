"""Uncertified reference policies compared against the CS^2 audit.

These answer R5/R6 in docs/Design.md: whether the frozen scorer beats simple
references at the same route mass, and what a non-deployable fixed-candidate
hindsight oracle can achieve.  None of these carries a certification claim;
only the
fixed-policy audit in audit.py (used through select -> certify -> confirm)
does.  Every function returns dictionaries shaped like
``AuditResult.to_dict()`` so that report.py can render them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .audit import (
    IOU_THRESHOLD,
    AuditResult,
    Policy,
    evaluate_policy,
    validate_one_row_per_image,
    validate_prediction_frame,
)


def no_switch_reference(
    frame: pd.DataFrame, split: str, *, iou_threshold: float = IOU_THRESHOLD
) -> AuditResult:
    """Protected-incumbent reference: never switch to the challenger."""

    return evaluate_policy(
        frame,
        Policy("no_switch", threshold=float("inf")),
        split=split,
        iou_threshold=iou_threshold,
    )


def raw_score_reference(
    frame: pd.DataFrame,
    split: str,
    threshold: float,
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
    iou_threshold: float = IOU_THRESHOLD,
) -> AuditResult:
    """Fixed a-priori threshold applied without select/certify isolation.

    Route/help/harm counts are exact, but the status is forced to
    ``descriptive`` because this threshold was not selected on a split
    disjoint from the one it is evaluated on, so it may not carry a CP claim.
    """

    result = evaluate_policy(
        frame,
        Policy(f"raw_score_{threshold:.2f}", threshold),
        split=split,
        alpha=alpha,
        delta=delta,
        iou_threshold=iou_threshold,
    )
    return replace(
        result,
        cp_upper=None,
        cp_delta=None,
        status="descriptive (uncertified: no select/certify isolation)",
    )


def oracle_reference(
    frame: pd.DataFrame, split: str, *, iou_threshold: float = IOU_THRESHOLD
) -> AuditResult:
    """Non-deployable upper bound: switch iff it strictly helps.

    Routes exactly the help-eligible rows (baseline wrong, challenger
    correct) and nothing else, so harm is zero by construction.  This bounds
    how much accuracy gain is even available in the candidate space.
    """

    validate_prediction_frame(frame)
    validate_one_row_per_image(frame, splits=(split,))
    baseline_iou = frame["baseline_iou"].to_numpy(dtype=float)
    challenger_iou = frame["challenger_iou"].to_numpy(dtype=float)
    baseline_correct = baseline_iou >= iou_threshold
    challenger_correct = challenger_iou >= iou_threshold
    routed = ~baseline_correct & challenger_correct

    n = int(len(frame))
    route_count = int(routed.sum())
    return AuditResult(
        split=split,
        policy="oracle",
        n=n,
        route_count=route_count,
        help_count=route_count,
        harm_count=0,
        incumbent_correct=int(baseline_correct.sum()),
        challenger_correct=int(challenger_correct.sum()),
        route_rate=route_count / n,
        conditional_harm=(0.0 if route_count else None),
        unconditional_harm=0.0,
        net_accuracy_gain=route_count / n,
        cp_upper=None,
        cp_delta=None,
        status="fixed-candidate hindsight oracle (non-deployable: uses ground truth)",
    )


def relation_verifier_reference(
    frame: pd.DataFrame,
    split: str,
    score_column: str,
    threshold: float,
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
    iou_threshold: float = IOU_THRESHOLD,
) -> AuditResult:
    """Evaluate an alternative scorer column as an uncertified baseline.

    ``score_column`` lets a manifest carry a second, independently trained
    scorer (e.g. a relation-verifier) alongside the frozen CS^2 score,
    without changing the audit's core routing/statistics code.
    """

    if score_column not in frame.columns:
        raise ValueError(
            f"manifest is missing optional column {score_column!r} required "
            "for this baseline"
        )
    working = frame.copy()
    working["score"] = frame[score_column]
    result = evaluate_policy(
        working,
        Policy(f"relation_verifier_{threshold:.2f}", threshold),
        split=split,
        alpha=alpha,
        delta=delta,
        iou_threshold=iou_threshold,
    )
    return replace(
        result,
        cp_upper=None,
        cp_delta=None,
        status="uncertified baseline (learned routing, no CP claim)",
    )


@dataclass(frozen=True)
class RandomReferenceSummary:
    """Distribution of outcomes under uniformly random routing."""

    split: str
    n: int
    route_count: int
    n_repeats: int
    seed: int
    mean_help: float
    mean_harm: float
    mean_conditional_harm: float | None
    mean_net_accuracy_gain: float
    net_accuracy_gain_samples: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "n": self.n,
            "route_count": self.route_count,
            "n_repeats": self.n_repeats,
            "seed": self.seed,
            "mean_help": self.mean_help,
            "mean_harm": self.mean_harm,
            "mean_conditional_harm": self.mean_conditional_harm,
            "mean_net_accuracy_gain": self.mean_net_accuracy_gain,
            "net_accuracy_gain_samples": self.net_accuracy_gain_samples,
        }

    def to_audit_row(self) -> dict[str, object]:
        """Render as an ``AuditResult``-shaped row for the main table."""

        return {
            "split": self.split,
            "policy": "matched_random",
            "n": self.n,
            "route_count": self.route_count,
            "help_count": round(self.mean_help, 2),
            "harm_count": round(self.mean_harm, 2),
            "cp_upper": None,
            "net_accuracy_gain": self.mean_net_accuracy_gain,
            "status": (
                f"descriptive (mean of {self.n_repeats} random-route "
                f"repeats, seed={self.seed})"
            ),
        }


def matched_random_reference(
    frame: pd.DataFrame,
    split: str,
    route_count: int,
    *,
    n_repeats: int = 200,
    seed: int = 0,
    iou_threshold: float = IOU_THRESHOLD,
) -> RandomReferenceSummary:
    """Route a uniformly random subset of the given size, repeated.

    ``route_count`` should be the frozen CS^2 policy's actual route count on
    the same split, so the comparison is at matched route mass.
    """

    validate_prediction_frame(frame)
    if frame["split"].nunique() != 1 or frame["split"].iat[0] != split:
        raise ValueError(f"expected a single split named {split!r}")

    n = len(frame)
    if not 0 <= route_count <= n:
        raise ValueError(f"route_count must be in [0, {n}], received {route_count}")

    baseline_correct = frame["baseline_iou"].to_numpy(dtype=float) >= iou_threshold
    challenger_correct = frame["challenger_iou"].to_numpy(dtype=float) >= iou_threshold

    rng = np.random.default_rng(seed)
    help_samples = np.empty(n_repeats)
    harm_samples = np.empty(n_repeats)
    net_gain_samples = np.empty(n_repeats)
    cond_harm_samples: list[float | None] = []
    for i in range(n_repeats):
        routed = np.zeros(n, dtype=bool)
        if route_count:
            routed_idx = rng.choice(n, size=route_count, replace=False)
            routed[routed_idx] = True
        help_count = int((routed & ~baseline_correct & challenger_correct).sum())
        harm_count = int((routed & baseline_correct & ~challenger_correct).sum())
        help_samples[i] = help_count
        harm_samples[i] = harm_count
        net_gain_samples[i] = (help_count - harm_count) / n
        cond_harm_samples.append(harm_count / route_count if route_count else None)

    defined_cond_harm = [value for value in cond_harm_samples if value is not None]
    mean_conditional_harm = (
        float(np.mean(defined_cond_harm)) if defined_cond_harm else None
    )

    return RandomReferenceSummary(
        split=split,
        n=n,
        route_count=route_count,
        n_repeats=n_repeats,
        seed=seed,
        mean_help=float(help_samples.mean()),
        mean_harm=float(harm_samples.mean()),
        mean_conditional_harm=mean_conditional_harm,
        mean_net_accuracy_gain=float(net_gain_samples.mean()),
        net_accuracy_gain_samples=[float(value) for value in net_gain_samples],
    )
