"""Select pre-declared qualitative examples from a frozen confirmation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit import IOU_THRESHOLD, Policy, validate_one_row_per_image, validate_prediction_frame


def _key(seed: int, category: str, image_id: object) -> str:
    return hashlib.sha256(f"{seed}\x1f{category}\x1f{image_id}".encode()).hexdigest()


def select_examples(
    frame: pd.DataFrame,
    policy: Policy,
    *,
    seed: int = 20270822,
    iou_threshold: float = IOU_THRESHOLD,
    split: str = "confirm",
) -> dict[str, object]:
    """Choose at most one help, harm, and no-route row without label peeking.

    Labels are used only to define the already-declared example categories;
    this function never changes the policy or any audit statistic.  The output
    is intentionally ID-first: rendering is marked pending unless box columns
    are present in the supplied artifact.
    """

    validate_prediction_frame(frame)
    subset = frame[frame["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"qualitative selection requires a non-empty {split!r} split")
    validate_one_row_per_image(subset, splits=(split,))
    baseline_correct = subset["baseline_iou"].to_numpy(dtype=float) >= iou_threshold
    challenger_correct = subset["challenger_iou"].to_numpy(dtype=float) >= iou_threshold
    routed = policy.route(subset["score"].to_numpy(dtype=float))
    subset["_category"] = "none"
    subset.loc[routed & ~baseline_correct & challenger_correct, "_category"] = "help"
    subset.loc[routed & baseline_correct & ~challenger_correct, "_category"] = "harm"
    subset.loc[~routed, "_category"] = "no_route"

    selected: dict[str, object] = {}
    counts: dict[str, int] = {}
    for category in ("help", "harm", "no_route"):
        candidates = subset[subset["_category"] == category].copy()
        counts[category] = int(len(candidates))
        if candidates.empty:
            selected[category] = None
            continue
        candidates["_key"] = candidates["image_id"].map(lambda value: _key(seed, category, value))
        row = candidates.sort_values("_key", kind="mergesort").iloc[0]
        selected[category] = {
            column: row[column]
            for column in frame.columns
            if column in row.index and not str(column).startswith("_")
        }

    has_boxes = any(
        column in frame.columns
        for column in ("baseline_box", "challenger_box", "baseline_bbox", "challenger_bbox")
    )
    return {
        "split": split,
        "policy": {"name": policy.name, "threshold": policy.threshold},
        "seed": seed,
        "iou_threshold": iou_threshold,
        "selection_rule": "minimum SHA-256(seed, category, image_id) within each category",
        "candidate_counts": counts,
        "selected": selected,
        "render_status": "ready" if has_boxes else "render_pending",
        "render_note": (
            "box columns are present in the manifest"
            if has_boxes
            else "original prediction boxes are not in the audit manifest"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20270822)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    result = select_examples(
        frame,
        Policy(name=f"score_ge_{args.threshold:.2f}", threshold=args.threshold),
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
