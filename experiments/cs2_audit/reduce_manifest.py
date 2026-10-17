"""Reduce a prediction manifest to one pre-declared expression per image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit import validate_image_disjoint, validate_prediction_frame, validate_one_row_per_image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(seed: int, split: object, image_id: object, row_token: object) -> str:
    payload = f"{seed}\x1f{split}\x1f{image_id}\x1f{row_token}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _label_independent_token(row: pd.Series, identity_columns: tuple[str, ...]) -> str:
    """Return a canonical row token that does not depend on input order.

    The token is built only from immutable, label-free expression identity
    fields.  It is deliberately separate from the source position used to
    retrieve the selected row after sorting.
    """

    values: dict[str, str] = {}
    for column in identity_columns:
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            values[column] = text
    if not values:
        raise ValueError("each row must contain a non-empty label-independent identity")
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


RULES = ("identity", "legacy-source-row")


def reduce_one_expression_per_image(
    frame: pd.DataFrame,
    *,
    seed: int = 20270822,
    rule: str = "identity",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select one stable row per image within each split.

    The selection key is a SHA-256 digest rather than pandas' row order.  This
    makes the selected expression reproducible across machines while keeping
    the expression choice independent of IoU and score labels.

    ``rule="identity"`` hashes the canonical expression identity only, so the
    choice does not depend on input order.  ``rule="legacy-source-row"`` is
    the rule that produced the audited System A records (2026-08-22): it also
    hashes the row's position in the input manifest, which follows generation
    order.  Both rules read no score or IoU.
    """

    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if rule == "legacy-source-row":
        return _reduce_legacy_source_row(frame, seed=seed)

    validate_prediction_frame(frame)
    validate_image_disjoint(frame)
    identity_columns = tuple(
        column
        for column in ("expression_id", "expression_text", "query", "text")
        if column in frame.columns
    )
    if not identity_columns:
        raise ValueError(
            "input manifest needs an immutable label-independent identity column "
            "(expression_id, expression_text, query, or text)"
        )
    expression_column = "expression_id" if "expression_id" in frame.columns else None

    working = frame.copy()
    working["_source_position"] = range(len(working))
    tokens = [_label_independent_token(row, identity_columns) for _, row in working.iterrows()]
    working["_label_independent_token"] = tokens
    duplicate_tokens = working.duplicated(
        subset=["split", "image_id", "_label_independent_token"], keep=False
    )
    if duplicate_tokens.any():
        raise ValueError(
            "duplicate label-independent identities occur within an image; "
            "the reduction would be ambiguous"
        )
    working["_selection_key"] = [
        _stable_key(seed, split, image_id, token)
        for split, image_id, token in zip(
            working["split"], working["image_id"], tokens
        )
    ]
    chosen_indices = (
        working.sort_values(["split", "image_id", "_selection_key"], kind="mergesort")
        .groupby(["split", "image_id"], sort=False, as_index=False)
        .head(1)["_source_position"]
        .astype(int)
        .tolist()
    )
    reduced = (
        frame.iloc[sorted(chosen_indices)]
        .reset_index(drop=True)
        .copy()
    )
    validate_one_row_per_image(reduced, splits=tuple(sorted(reduced["split"].unique())))
    summary: dict[str, object] = {
        "sampling_unit": "one preselected expression per image",
        "selection_rule": (
            "minimum SHA-256(seed, split, image_id, canonical label-independent identity); "
            "source position is used only to retrieve the selected row"
        ),
        "seed": seed,
        "expression_column": expression_column,
        "identity_columns": list(identity_columns),
        "input_rows": int(len(frame)),
        "output_rows": int(len(reduced)),
        "input_images": int(frame["image_id"].nunique()),
        "output_images": int(reduced["image_id"].nunique()),
        "rows_by_split": frame.groupby("split").size().astype(int).to_dict(),
        "images_by_split": frame.groupby("split")["image_id"].nunique().astype(int).to_dict(),
        "reduced_rows_by_split": reduced.groupby("split").size().astype(int).to_dict(),
    }
    return reduced, summary


def _reduce_legacy_source_row(frame: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """The 2026-08-22 rule: minimum SHA-256(seed, split, image_id, expression_id + source row)."""

    validate_prediction_frame(frame)
    validate_image_disjoint(frame)
    if "expression_id" not in frame.columns:
        raise ValueError("the legacy rule needs an expression_id column")
    working = frame.copy()
    working["_source_row"] = range(len(working))
    working["_selection_key"] = [
        _stable_key(seed, split, image_id, f"{token}\x1f{source_row}")
        for source_row, (split, image_id, token) in enumerate(
            zip(working["split"], working["image_id"], working["expression_id"].astype(str))
        )
    ]
    chosen_indices = (
        working.sort_values(["split", "image_id", "_selection_key"], kind="mergesort")
        .groupby(["split", "image_id"], sort=False, as_index=False)
        .head(1)["_source_row"]
        .astype(int)
        .tolist()
    )
    reduced = frame.iloc[sorted(chosen_indices)].reset_index(drop=True).copy()
    validate_one_row_per_image(reduced, splits=tuple(sorted(reduced["split"].unique())))
    summary: dict[str, object] = {
        "sampling_unit": "one preselected expression per image",
        "selection_rule": "legacy-source-row: minimum SHA-256(seed, split, image_id, expression_id, source_row)",
        "seed": seed,
        "expression_column": "expression_id",
        "input_rows": int(len(frame)),
        "output_rows": int(len(reduced)),
        "input_images": int(frame["image_id"].nunique()),
        "output_images": int(reduced["image_id"].nunique()),
        "rows_by_split": frame.groupby("split").size().astype(int).to_dict(),
        "images_by_split": frame.groupby("split")["image_id"].nunique().astype(int).to_dict(),
        "reduced_rows_by_split": reduced.groupby("split").size().astype(int).to_dict(),
    }
    return reduced, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--seed", type=int, default=20270822)
    parser.add_argument("--rule", choices=RULES, default="identity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    reduced, summary = reduce_one_expression_per_image(frame, seed=args.seed, rule=args.rule)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(args.output_csv, index=False)
    summary["input_sha256"] = sha256_file(args.input_csv)
    summary["output_sha256"] = sha256_file(args.output_csv)
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
