"""Register a label-blind distribution-shift stress subset.

The default rule targets long referring expressions containing a predeclared
spatial/relational cue on the held-out ``confirm`` split.  The rule reads only
metadata (split and expression text); it never reads IoU, help, harm, or a
model score.  The emitted registry records the rule and both file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_CUES = (
    "left",
    "right",
    "behind",
    "front",
    "between",
    "near",
    "next to",
    "under",
    "above",
    "below",
    "closest",
    "farthest",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_column(frame: pd.DataFrame) -> str:
    for name in ("expression_text", "sentence", "expression", "text"):
        if name in frame.columns:
            return name
    raise ValueError(
        "stress registration needs expression text in one of "
        "expression_text/sentence/expression/text"
    )


def register_subset(
    frame: pd.DataFrame,
    *,
    cues: tuple[str, ...] = DEFAULT_CUES,
    min_tokens: int = 8,
    split: str = "confirm",
) -> tuple[pd.DataFrame, dict[str, object]]:
    text_column = _text_column(frame)
    if "split" not in frame.columns:
        raise ValueError("manifest must contain split")
    if min_tokens < 1 or not cues:
        raise ValueError("min_tokens must be positive and cues must be non-empty")
    normalized = frame[text_column].fillna("").astype(str).str.lower()
    cue_pattern = re.compile(r"(?:" + "|".join(re.escape(cue) for cue in cues) + r")")
    cue_mask = normalized.map(lambda value: bool(cue_pattern.search(value)))
    token_mask = normalized.map(lambda value: len(value.split()) >= min_tokens)
    subset = frame.loc[(frame["split"] == split) & cue_mask & token_mask].copy()
    if subset.empty:
        raise ValueError("stress rule selected zero rows; inspect metadata before running a model")
    if "image_id" in subset.columns and subset["image_id"].duplicated().any():
        raise ValueError("stress subset must remain one row per image")
    summary = {
        "registry_version": "distribution_shift_stress_v1",
        "selection_rule": {
            "split": split,
            "min_tokens": int(min_tokens),
            "relational_cues": list(cues),
            "uses_labels": False,
            "uses_model_scores": False,
        },
        "text_column": text_column,
        "rows": int(len(subset)),
        "images": int(subset["image_id"].nunique()) if "image_id" in subset else None,
    }
    return subset, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--registry-json", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--split", default="confirm")
    parser.add_argument("--cues", default=",".join(DEFAULT_CUES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    cues = tuple(value.strip().lower() for value in args.cues.split(",") if value.strip())
    subset, summary = register_subset(
        frame, cues=cues, min_tokens=args.min_tokens, split=args.split
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.registry_json.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output_csv, index=False)
    summary["input_sha256"] = _sha256(args.input_csv)
    summary["subset_sha256"] = _sha256(args.output_csv)
    args.registry_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
