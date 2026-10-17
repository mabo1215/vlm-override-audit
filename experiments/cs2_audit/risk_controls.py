"""Small, auditable risk-control and train-only discriminator primitives."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import pandas as pd

from .audit import AuditResult, Policy, evaluate_policy


# Labels and ground-truth-derived columns must never become discriminator
# features.  Keep this guard here, next to the train-only fit, so callers do
# not have to remember the leakage policy themselves.
_FORBIDDEN_FEATURE_TOKENS = (
    "iou",
    "ground_truth",
    "groundtruth",
    "gt_",
    "label",
    "help",
    "harm",
    "correct",
)


def validate_discriminator_features(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze model-side, non-label discriminator features."""

    columns = tuple(str(column) for column in feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be a non-empty list of unique names")
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            "discriminator features may not contain ground-truth/label-derived "
            f"columns: {forbidden}"
        )
    return columns


def rcps_binary_harm_reference(
    frame: pd.DataFrame,
    policy: Policy,
    *,
    split: str,
    alpha: float = 0.05,
    delta: float = 0.05,
) -> dict[str, object]:
    """Report the binary-harm RCPS comparator.

    For this Bernoulli harm event and a fixed policy, the exact finite-sample
    RCPS upper bound is the same Clopper--Pearson upper bound used by the audit.
    Keeping this adapter explicit prevents the paper from presenting an
    algebraically identical calculation as an independent empirical result.
    """

    result: AuditResult = evaluate_policy(
        frame,
        policy,
        split=split,
        alpha=alpha,
        delta=delta,
        certification_family_size=1,
    )
    output = result.to_dict()
    output.update(
        {
            "method": "RCPS-binary-harm",
            "equivalent_to": "Clopper-Pearson conditional-harm bound for a fixed policy",
            "claim_status": "comparator; not an additional independent certificate",
        }
    )
    return output


@dataclass(frozen=True)
class ImprovementDiscriminator:
    """A deterministic logistic discriminator trained only on ``train``."""

    feature_columns: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    train_rows: int
    eligible_rows: int
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _feature_matrix(frame, self.feature_columns)
        normalized = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        logits = normalized @ np.asarray(self.weights) + self.bias
        logits = np.clip(logits, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-logits))


def _feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"discriminator features are missing columns: {missing}")
    matrix = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("discriminator features must be finite numeric columns")
    return matrix


def fit_improvement_discriminator(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    seed: int = 20270822,
    steps: int = 600,
    learning_rate: float = 0.15,
    l2: float = 1e-2,
) -> ImprovementDiscriminator:
    """Fit help-vs-harm on train labels, never on select/certify/confirm.

    Rows that are neither help nor harm are excluded from the training target.
    Features must be exported model-side scores or metadata, not IoUs or any
    field derived from the ground-truth box.
    """

    feature_columns = validate_discriminator_features(feature_columns)
    if steps < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("steps must be positive; learning_rate and l2 must be valid")
    train = frame[frame["split"] == "train"].copy()
    if train.empty:
        raise ValueError("improvement discriminator requires a non-empty train split")
    matrix = _feature_matrix(train, feature_columns)
    baseline_correct = train["baseline_iou"].to_numpy(dtype=float) >= 0.5
    challenger_correct = train["challenger_iou"].to_numpy(dtype=float) >= 0.5
    help_mask = ~baseline_correct & challenger_correct
    harm_mask = baseline_correct & ~challenger_correct
    eligible = help_mask | harm_mask
    if int(eligible.sum()) < 2 or len(np.unique(help_mask[eligible].astype(int))) < 2:
        raise ValueError("train split needs both help and harm examples")
    x = matrix[eligible]
    y = help_mask[eligible].astype(float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    x = (x - means) / scales
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.01, size=x.shape[1])
    bias = 0.0
    for _ in range(steps):
        logits = np.clip(x @ weights + bias, -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y
        weights -= learning_rate * ((x.T @ error) / len(x) + l2 * weights)
        bias -= learning_rate * float(error.mean())
    return ImprovementDiscriminator(
        feature_columns=tuple(feature_columns),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
        train_rows=int(len(train)),
        eligible_rows=int(eligible.sum()),
        seed=seed,
    )


def add_improvement_score(
    frame: pd.DataFrame,
    model: ImprovementDiscriminator,
    *,
    column: str = "improvement_score",
) -> pd.DataFrame:
    """Return a copy with train-fitted discriminator scores for all splits."""

    output = frame.copy()
    output[column] = model.score(frame)
    return output
