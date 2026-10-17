"""Create a fresh, image-disjoint four-way split without reading labels.

The split is assigned to image IDs before any model fitting or policy
selection.  It is intentionally a separate command from the audit so that
the resulting CSV and its hash can be registered before generation starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SPLITS = ("train", "select", "certify", "confirm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_counts(n: int, fractions: Sequence[float]) -> list[int]:
    raw = np.asarray(fractions, dtype=float) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[int(index)] += 1
    return [int(value) for value in counts]


def resplit_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    fractions: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Assign every image group to exactly one protocol split."""

    if "image_id" not in frame.columns:
        raise ValueError("input manifest must contain image_id")
    if len(fractions) != 4 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain four positive values")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    duplicates = frame.duplicated("image_id", keep=False)
    if bool(duplicates.any()):
        examples = frame.loc[duplicates, "image_id"].astype(str).head(8).tolist()
        raise ValueError(
            "independent split requires one row per image; reduce the raw "
            f"manifest first (examples: {examples})"
        )

    image_ids = frame["image_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(image_ids))
    counts = _allocate_counts(len(image_ids), fractions)
    assigned = np.empty(len(image_ids), dtype=object)
    start = 0
    for split, count in zip(SPLITS, counts):
        for row_index in order[start : start + count]:
            assigned[int(row_index)] = split
        start += count

    output = frame.copy()
    output["split"] = assigned
    output = output.sort_values(["split", "image_id"], kind="mergesort").reset_index(drop=True)
    memberships = output.groupby("image_id")["split"].nunique()
    if int(memberships.max()) != 1:
        raise AssertionError("image-level split assignment produced leakage")
    summary = {
        "seed": int(seed),
        "fractions": {split: float(value) for split, value in zip(SPLITS, fractions)},
        "rows_by_split": output.groupby("split").size().astype(int).to_dict(),
        "images_by_split": output.groupby("split")["image_id"].nunique().astype(int).to_dict(),
        "image_count": int(output["image_id"].nunique()),
        "one_row_per_image": True,
        "selection_rule": "numpy.default_rng(seed).permutation(image_id), contiguous allocation",
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fractions", default="0.25,0.25,0.25,0.25")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(","))
    frame = pd.read_csv(args.input_csv)
    output, summary = resplit_frame(frame, seed=args.seed, fractions=fractions)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    summary["input_sha256"] = _sha256(args.input_csv)
    summary["output_sha256"] = _sha256(args.output_csv)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
