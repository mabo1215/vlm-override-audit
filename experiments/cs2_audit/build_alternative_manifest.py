"""Join a model-generated candidate artifact to the frozen audit manifest.

The generator does not see labels.  This post-processing command is the only
place where ground-truth boxes are read to compute ``challenger_iou`` for the
audit and to attach the model-side features used by the train-only
improvement discriminator.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

from .build_manifest import box_iou


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ground_truth_metadata(parquet_glob: str) -> dict[str, dict[str, object]]:
    paths = sorted(glob.glob(parquet_glob, recursive=True)) if any(
        char in parquet_glob for char in "*?["
    ) else [parquet_glob]
    output: dict[str, dict[str, object]] = {}
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["ref_id", "bbox", "raw_image_info", "sentences", "file_name"],
        )
        for _, row in frame.iterrows():
            key = str(row["ref_id"])
            if key in output:
                continue
            info = json.loads(row["raw_image_info"])
            width, height = float(info["width"]), float(info["height"])
            x1, y1, x2, y2 = (float(value) for value in row["bbox"])
            sentences = row["sentences"]
            if isinstance(sentences, list) and sentences:
                first = sentences[0]
                text = first.get("sent", first.get("raw", "")) if isinstance(first, dict) else str(first)
            else:
                text = ""
            output[key] = {
                "box": (x1 / width, y1 / height, x2 / width, y2 / height),
                "file_name": str(row["file_name"]),
                "expression_text": str(text),
            }
    if not output:
        raise FileNotFoundError(f"no ground-truth metadata resolved from {parquet_glob}")
    return output


def load_candidates(pattern: str) -> dict[str, dict[str, object]]:
    paths = sorted(glob.glob(pattern, recursive=True)) if any(
        char in pattern for char in "*?["
    ) else [pattern]
    candidates: dict[str, dict[str, object]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("expression_id", ""))
                if not key or row.get("status") != "ok":
                    continue
                if key in candidates:
                    raise ValueError(f"duplicate candidate for expression_id {key!r} at {path}:{line_number}")
                candidates[key] = row
    if not candidates:
        raise FileNotFoundError(f"no successful candidate rows resolved from {pattern}")
    return candidates


def build_alternative_manifest(
    base: pd.DataFrame,
    candidates: dict[str, dict[str, object]],
    truth: dict[str, dict[str, object]],
    *,
    score_field: str = "candidate_confidence",
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"split", "image_id", "expression_id", "baseline_iou"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"base manifest is missing columns: {missing}")
    rows: list[dict[str, object]] = []
    stats = {"base_rows": int(len(base)), "missing_candidate": 0, "missing_truth": 0, "ok": 0}
    for source in base.to_dict("records"):
        key = str(source["expression_id"])
        candidate = candidates.get(key)
        gt = truth.get(key)
        if candidate is None:
            stats["missing_candidate"] += 1
            continue
        if gt is None:
            stats["missing_truth"] += 1
            continue
        if score_field not in candidate:
            raise ValueError(f"candidate {key!r} has no score field {score_field!r}")
        box = tuple(float(value) for value in candidate["candidate_box"])
        row = dict(source)
        row.update(
            {
                "score": float(candidate[score_field]),
                "challenger_iou": box_iou(box, gt["box"]),
                "expression_text": str(candidate.get("expression_text") or gt["expression_text"]),
                "candidate_box": json.dumps(list(box), separators=(",", ":")),
                "candidate_confidence": float(candidate.get("candidate_confidence", 0.0)),
                "candidate_count": int(candidate.get("candidate_count", 0)),
                "candidate_box_area": float(candidate.get("candidate_box_area", 0.0)),
                "candidate_aspect_ratio": float(candidate.get("candidate_aspect_ratio", 0.0)),
                "query_token_count": int(candidate.get("query_token_count", 0)),
                "candidate_generator": str(candidate.get("model", "unknown")),
            }
        )
        rows.append(row)
        stats["ok"] += 1
    if stats["missing_candidate"] or stats["missing_truth"]:
        raise ValueError(f"alternative manifest is incomplete: {stats}")
    output = pd.DataFrame(rows)
    if output["image_id"].duplicated().any():
        raise ValueError("alternative manifest has duplicate image_id rows")
    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--gt-parquet-glob", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--score-field", default="candidate_confidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_manifest)
    candidates = load_candidates(args.candidate_jsonl)
    truth = load_ground_truth_metadata(args.gt_parquet_glob)
    output, summary = build_alternative_manifest(base, candidates, truth, score_field=args.score_field)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    summary.update(
        {
            "input_sha256": _sha256(args.base_manifest),
            "output_sha256": _sha256(args.output_csv),
            "candidate_count": len(candidates),
            "score_field": args.score_field,
            "label_join_stage": "post-generation; generator never reads GT",
        }
    )
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
