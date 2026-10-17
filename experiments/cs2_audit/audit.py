"""Auditable fixed-policy risk evaluation for selective switching.

The module deliberately separates policy selection from certification.  A policy
is selected on ``select`` rows, frozen, and evaluated on an independent
``certify`` split.  Confirmation is reported separately and is never used to
change the policy.  The implementation is intentionally small so that the
same functions can be used by the vGPU run and by the paper's smoke test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta


IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class Policy:
    """A pre-declared score-threshold policy.

    ``threshold`` is a score threshold, not a calibrated probability.  Ties
    are routed (``score >= threshold``) and are therefore deterministic.
    """

    name: str
    threshold: float

    def route(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        return values >= self.threshold


@dataclass(frozen=True)
class AuditResult:
    """Counts and estimands for one fixed policy on one split."""

    split: str
    policy: str
    n: int
    route_count: int
    help_count: int
    harm_count: int
    incumbent_correct: int
    challenger_correct: int
    route_rate: float
    conditional_harm: float | None
    unconditional_harm: float
    net_accuracy_gain: float
    cp_upper: float | None
    cp_delta: float | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def clopper_pearson_upper(k: int, m: int, delta: float) -> float | None:
    """Return a one-sided CP upper bound, or ``None`` for zero routes.

    For ``m == 0`` conditional harm is undefined.  The old manuscript treated
    this case as a zero risk estimate; this explicit branch prevents that
    mistake from reappearing in downstream analyses.
    """

    if not 0 <= k <= m:
        raise ValueError(f"require 0 <= k <= m, received k={k}, m={m}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), received {delta}")
    if m == 0:
        return None
    if k == m:
        return 1.0
    upper = float(beta.ppf(1.0 - delta, k + 1, m - k))
    if not np.isfinite(upper):
        raise ArithmeticError(f"non-finite CP bound for k={k}, m={m}")
    return min(1.0, max(0.0, upper))


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Validate the minimum prediction manifest schema and finite values."""

    required = {
        "split",
        "image_id",
        "score",
        "baseline_iou",
        "challenger_iou",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction manifest is empty")
    for column in ("score", "baseline_iou", "challenger_iou"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"column {column!r} contains non-finite values")
    if frame["split"].isna().any() or frame["image_id"].isna().any():
        raise ValueError("split and image_id must not contain missing values")


def validate_record_consistency(frame: pd.DataFrame, *, tie_tolerance: float = 1e-6) -> None:
    """Strict record rules for rebuilt and new manifests (review items J7, J8).

    * a switch score in (0, tie_tolerance) is a tie and must be stored as 0;
    * a record without a switch must have alternative IoU equal to default IoU.

    The archived 2026-08 manifests predate these rules and are audited as they
    were; this check is applied to every manifest built since.
    """

    score = frame["score"].astype(float)
    ties = (score > 0) & (score < tie_tolerance)
    if ties.any():
        raise ValueError(f"{int(ties.sum())} switch scores lie in (0, {tie_tolerance}); store ties as 0")
    no_switch = score == 0
    gap = (frame.loc[no_switch, "challenger_iou"] - frame.loc[no_switch, "baseline_iou"]).abs()
    if (gap > 1e-12).any():
        raise ValueError(f"{int((gap > 1e-12).sum())} records without a switch have a different alternative IoU")


def validate_image_disjoint(frame: pd.DataFrame) -> None:
    """Fail if an image appears in more than one protocol split."""

    memberships = frame[["image_id", "split"]].drop_duplicates()
    counts = memberships.groupby("image_id")["split"].nunique()
    overlap = counts[counts > 1]
    if not overlap.empty:
        examples = list(overlap.index[:5])
        raise ValueError(
            "image-level split leakage detected for image_id values "
            f"{examples}"
        )


def validate_one_row_per_image(
    frame: pd.DataFrame,
    *,
    splits: Sequence[str] = ("select", "certify", "confirm"),
) -> None:
    """Require the declared image-level sampling unit for formal audits.

    Referring-expression datasets can contain several expressions for one
    image.  A row-level CP calculation is not an image-level certificate when
    those rows are retained.  The reduction is deliberately a separate,
    auditable preprocessing step (``reduce_manifest.py``); this validator only
    prevents an accidental change of estimand inside the audit.
    """

    scoped = frame[frame["split"].isin(splits)]
    duplicated = scoped[scoped.duplicated("image_id", keep=False)]
    if duplicated.empty:
        return
    examples = (
        duplicated.groupby(["split", "image_id"], sort=True)
        .size()
        .head(5)
        .to_dict()
    )
    raise ValueError(
        "formal image-level audit requires one row per image in "
        f"{list(splits)}; duplicate image/expression rows found: {examples}. "
        "Run reduce_manifest.py with a recorded seed first."
    )


def _as_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame["score"].to_numpy(dtype=float),
        frame["baseline_iou"].to_numpy(dtype=float),
        frame["challenger_iou"].to_numpy(dtype=float),
    )


def evaluate_policy(
    frame: pd.DataFrame,
    policy: Policy,
    *,
    split: str,
    alpha: float = 0.05,
    delta: float = 0.05,
    certification_family_size: int = 1,
    iou_threshold: float = IOU_THRESHOLD,
) -> AuditResult:
    """Evaluate route/help/harm and an optional simultaneous CP bound.

    ``frame`` must already be a single split.  ``certification_family_size``
    applies Bonferroni only when the certification set is used to test a
    family of policies.  The normal four-way protocol selects one policy on
    ``select`` and therefore uses the pointwise value 1 on ``certify``.
    """

    validate_prediction_frame(frame)
    if frame["split"].nunique() != 1 or frame["split"].iat[0] != split:
        raise ValueError(f"expected a single split named {split!r}")
    validate_one_row_per_image(frame, splits=(split,))
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), received {alpha}")
    if certification_family_size < 1:
        raise ValueError("certification_family_size must be positive")

    scores, baseline_iou, challenger_iou = _as_arrays(frame)
    # Policies that need more than the switch score (e.g. a verifier gate)
    # provide route_frame; plain threshold policies route on the score.
    routed = policy.route_frame(frame) if hasattr(policy, "route_frame") else policy.route(scores)
    baseline_correct = baseline_iou >= iou_threshold
    challenger_correct = challenger_iou >= iou_threshold
    help_mask = routed & ~baseline_correct & challenger_correct
    harm_mask = routed & baseline_correct & ~challenger_correct
    n = int(len(frame))
    route_count = int(routed.sum())
    help_count = int(help_mask.sum())
    harm_count = int(harm_mask.sum())
    cp_delta = delta / certification_family_size
    cp_upper = clopper_pearson_upper(harm_count, route_count, cp_delta)

    if route_count == 0:
        conditional_harm = None
        status = "zero-route; conditional harm undefined"
    else:
        conditional_harm = harm_count / route_count
        status = "certified" if cp_upper <= alpha else "not-certified"

    return AuditResult(
        split=split,
        policy=policy.name,
        n=n,
        route_count=route_count,
        help_count=help_count,
        harm_count=harm_count,
        incumbent_correct=int(baseline_correct.sum()),
        challenger_correct=int(challenger_correct.sum()),
        route_rate=route_count / n,
        conditional_harm=conditional_harm,
        unconditional_harm=harm_count / n,
        net_accuracy_gain=(help_count - harm_count) / n,
        cp_upper=cp_upper,
        cp_delta=cp_delta if route_count else None,
        status=status,
    )


def select_policy(
    frame: pd.DataFrame,
    policies: Sequence[Policy],
    *,
    alpha: float = 0.05,
    min_route_count: int = 1,
    iou_threshold: float = IOU_THRESHOLD,
) -> Policy:
    """Select one policy using labels from the select split only.

    This selection rule is fixed before certification: among policies with at
    least ``min_route_count`` selected routes and observed conditional harm at
    most ``alpha``, maximize net accuracy gain; ties prefer higher coverage
    and then the lower threshold.  If no policy passes the selection screen,
    choose the policy with the largest observed net gain and still certify it
    independently.  The fallback is intentionally explicit rather than
    silently converting selection failure into a zero-route result.
    """

    validate_prediction_frame(frame)
    if frame["split"].nunique() != 1 or frame["split"].iat[0] != "select":
        raise ValueError("policy selection requires exactly the 'select' split")
    if not policies:
        raise ValueError("at least one pre-declared policy is required")

    results = [
        evaluate_policy(
            frame,
            policy,
            split="select",
            alpha=alpha,
            delta=0.05,
            certification_family_size=1,
            iou_threshold=iou_threshold,
        )
        for policy in policies
    ]
    admissible = [
        result
        for result in results
        if result.route_count >= min_route_count
        and result.conditional_harm is not None
        and result.conditional_harm <= alpha
    ]
    candidates = admissible if admissible else results
    chosen = max(
        candidates,
        key=lambda result: (
            result.net_accuracy_gain,
            result.route_rate,
            -next(policy.threshold for policy in policies if policy.name == result.policy),
        ),
    )
    return next(policy for policy in policies if policy.name == chosen.policy)


def _selection_trace(
    frame: pd.DataFrame,
    policies: Sequence[Policy],
    *,
    alpha: float,
    min_route_count: int,
    iou_threshold: float,
) -> tuple[Policy, list[dict[str, object]], str]:
    """Return the frozen policy, every select outcome, and selection reason."""

    results = [
        evaluate_policy(
            frame,
            policy,
            split="select",
            alpha=alpha,
            delta=0.05,
            certification_family_size=1,
            iou_threshold=iou_threshold,
        )
        for policy in policies
    ]
    admissible = [
        result
        for result in results
        if result.route_count >= min_route_count
        and result.conditional_harm is not None
        and result.conditional_harm <= alpha
    ]
    candidates = admissible if admissible else results
    chosen = max(
        candidates,
        key=lambda result: (
            result.net_accuracy_gain,
            result.route_rate,
            -next(policy.threshold for policy in policies if policy.name == result.policy),
        ),
    )
    selected = next(policy for policy in policies if policy.name == chosen.policy)
    reason = "screened-policy" if admissible else "fallback-no-policy-passed-select-screen"
    trace = []
    for result in results:
        row = result.to_dict()
        row["selection_eligible"] = result in admissible
        row["selected"] = result.policy == selected.name
        trace.append(row)
    return selected, trace, reason


def run_protocol(
    frame: pd.DataFrame,
    policies: Sequence[Policy],
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, object]:
    """Run select → freeze → certify → confirm on a manifest.

    The returned JSON-compatible dictionary records both the selected policy
    and all counts needed to audit the result.  Confirmation has no CP claim:
    it is a held-out utility report for the already frozen policy.
    """

    validate_prediction_frame(frame)
    validate_image_disjoint(frame)
    expected = {"train", "select", "certify", "confirm"}
    observed = set(frame["split"].unique())
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"manifest is missing required splits: {missing}")

    validate_one_row_per_image(frame)

    selected, selection_trace, selection_reason = _selection_trace(
        frame[frame["split"] == "select"],
        policies,
        alpha=alpha,
        iou_threshold=iou_threshold,
        min_route_count=1,
    )
    certify = evaluate_policy(
        frame[frame["split"] == "certify"],
        selected,
        split="certify",
        alpha=alpha,
        delta=delta,
        certification_family_size=1,
        iou_threshold=iou_threshold,
    )
    confirm = evaluate_policy(
        frame[frame["split"] == "confirm"],
        selected,
        split="confirm",
        alpha=alpha,
        delta=delta,
        certification_family_size=1,
        iou_threshold=iou_threshold,
    )
    confirm_utility_gate = confirm.net_accuracy_gain >= 0.0
    certification_gate = certify.status == "certified"
    deployment_status = (
        "admitted"
        if certification_gate and confirm_utility_gate
        else "not-admitted"
    )
    return {
        "protocol": {
            "splits": ["train", "select", "certify", "confirm"],
            "alpha": alpha,
            "delta": delta,
            "iou_threshold": iou_threshold,
            "certification_family_size": 1,
            "selection_policy_count": len(policies),
            "policy_frozen_before_certify": True,
            "image_disjoint": True,
            "sampling_unit": "one preselected expression per image",
            "one_row_per_image": True,
            "unique_images": {
                split: int(frame.loc[frame["split"] == split, "image_id"].nunique())
                for split in ("train", "select", "certify", "confirm")
            },
            "selection_reason": selection_reason,
            "certification_gate": certification_gate,
            "confirm_utility_gate": confirm_utility_gate,
            "deployment_status": deployment_status,
        },
        "selected_policy": asdict(selected),
        "selection_trace": selection_trace,
        "certify": certify.to_dict(),
        "confirm": confirm.to_dict(),
    }


def frame_from_records(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build and validate a manifest from model-output records."""

    frame = pd.DataFrame(list(records))
    validate_prediction_frame(frame)
    return frame
