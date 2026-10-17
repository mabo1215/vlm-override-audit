"""Build the CS^2 audit CSV manifest from raw candidate-generation and
comparative-scoring JSON logs, plus the RefCOCO-family ground-truth Parquet.

This mirrors the incumbent/challenger/margin construction used by the
frozen candidate-generation pipeline that produced these logs: the
``"restricted"``-sourced candidate is the incumbent, the candidate with the
highest value of the chosen scorer field is the challenger (ties resolve to
the incumbent), and ``score = challenger_value - incumbent_value`` is the
switching-confidence value that ``audit.py``'s ``score >= threshold`` policy
class routes on. Ground-truth boxes and generation/scoring logs are read
from caller-supplied paths only -- no local repository or host path is
hardcoded here, so this script works against any frozen candidate generator
that emits the same JSON shape, not a specific machine's checkout.

Two independently-drawn, image-disjoint pools are expected as input (e.g. a
"calibration" pool and a held-out "confirmation" pool). The confirmation
pool becomes the ``confirm`` split verbatim; the calibration pool is
re-partitioned at the image level into ``train``/``select``/``certify`` so
that policy selection and certification never share images (or the
confirmation pool), per docs/Design.md's four-way protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def load_ground_truth(parquet_glob: str) -> dict[str, tuple[float, float, float, float]]:
    """Map ``ref_id`` (as used for ``example_id`` in the logs) to a normalized GT box."""

    truth: dict[str, tuple[float, float, float, float]] = {}
    paths = sorted(glob.glob(parquet_glob, recursive=True)) if any(
        ch in parquet_glob for ch in "*?["
    ) else [parquet_glob]
    for path in paths:
        frame = pd.read_parquet(path, columns=["ref_id", "bbox", "raw_image_info"])
        for _, row in frame.iterrows():
            key = str(row["ref_id"])
            if key in truth:
                continue
            info = json.loads(row["raw_image_info"])
            width, height = float(info["width"]), float(info["height"])
            x1, y1, x2, y2 = (float(v) for v in row["bbox"])
            truth[key] = (x1 / width, y1 / height, x2 / width, y2 / height)
    if not truth:
        raise FileNotFoundError(f"no ground-truth rows resolved from: {parquet_glob}")
    return truth


def load_generation_logs(pattern: str) -> dict[str, dict]:
    """``example_id`` -> generation-log row (``restricted_iou``, ``image_id``, ...)."""

    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(pattern)):
        payload = json.loads(Path(path).read_text())
        for row in payload.get("logs", []):
            if "error" in row:
                continue
            rows[str(row["example_id"])] = row
    if not rows:
        raise FileNotFoundError(f"no generation-log rows resolved from: {pattern}")
    return rows


def load_score_logs(pattern: str, field: str) -> dict[str, dict]:
    """``example_id`` -> score-log entry, kept only if every candidate exposes ``field``."""

    entries: dict[str, dict] = {}
    for path in sorted(glob.glob(pattern)):
        payload = json.loads(Path(path).read_text())
        for entry in payload.get("scores", []):
            candidates = entry.get("candidate_scores")
            if not candidates or field not in candidates[0]:
                continue
            entries[str(entry["example_id"])] = entry
    if not entries:
        raise FileNotFoundError(f"no score-log rows resolved from: {pattern}")
    return entries


def build_rows(
    generation: dict[str, dict],
    scores: dict[str, dict],
    ground_truth: dict[str, tuple[float, float, float, float]],
    field: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    stats = {"candidates": len(generation), "no_gt": 0, "no_score": 0, "no_restricted": 0, "ok": 0}
    for example_id, gen_row in generation.items():
        gt = ground_truth.get(example_id)
        entry = scores.get(example_id)
        if gt is None:
            stats["no_gt"] += 1
            continue
        if entry is None:
            stats["no_score"] += 1
            continue
        sources = list(entry["candidate_sources"])
        if "restricted" not in sources:
            stats["no_restricted"] += 1
            continue
        boxes = [tuple(b) for b in entry["candidate_boxes"]]
        values = [float(s[field]) for s in entry["candidate_scores"]]
        incumbent = sources.index("restricted")
        ious = [box_iou(b, gt) for b in boxes]
        best = int(np.argmax(values))
        if values[best] <= values[incumbent]:
            best = incumbent
        rows.append(
            {
                "image_id": int(gen_row["image_id"]),
                "expression_id": example_id,
                "score": values[best] - values[incumbent],
                "baseline_iou": float(gen_row["restricted_iou"]),
                "challenger_iou": float(ious[best]),
            }
        )
        stats["ok"] += 1
    return rows, stats


def assign_splits(
    cal_rows: list[dict[str, object]],
    conf_rows: list[dict[str, object]],
    *,
    seed: int,
    select_frac: float,
    certify_frac: float,
) -> None:
    """Mutate rows in place, adding a ``split`` key.

    ``conf_rows`` (the pre-held-out confirmation pool) always becomes
    ``confirm``. ``cal_rows`` is partitioned at the image level into
    train/select/certify; the remainder after ``select_frac``/``certify_frac``
    becomes ``train``.
    """

    if not 0.0 < select_frac < 1.0 or not 0.0 < certify_frac < 1.0:
        raise ValueError("select_frac and certify_frac must be in (0, 1)")
    if select_frac + certify_frac >= 1.0:
        raise ValueError("select_frac + certify_frac must leave room for a nonempty train split")

    for row in conf_rows:
        row["split"] = "confirm"

    images = sorted({row["image_id"] for row in cal_rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    n = len(images)
    n_select = int(round(n * select_frac))
    n_certify = int(round(n * certify_frac))
    select_images = set(images[:n_select])
    certify_images = set(images[n_select : n_select + n_certify])
    for row in cal_rows:
        if row["image_id"] in select_images:
            row["split"] = "select"
        elif row["image_id"] in certify_images:
            row["split"] = "certify"
        else:
            row["split"] = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-generation-glob", required=True)
    parser.add_argument("--calibration-score-glob", required=True)
    parser.add_argument("--calibration-gt-parquet", required=True)
    parser.add_argument("--confirmation-generation-glob", required=True)
    parser.add_argument("--confirmation-score-glob", required=True)
    parser.add_argument("--confirmation-gt-parquet", required=True)
    parser.add_argument(
        "--scorer-field",
        default="pairwise_score",
        choices=["pairwise_score", "listwise_posterior"],
    )
    parser.add_argument("--split-seed", type=int, default=20270822)
    parser.add_argument("--select-frac", type=float, default=0.45)
    parser.add_argument("--certify-frac", type=float, default=0.45)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cal_gt = load_ground_truth(args.calibration_gt_parquet)
    cal_gen = load_generation_logs(args.calibration_generation_glob)
    cal_scores = load_score_logs(args.calibration_score_glob, args.scorer_field)
    cal_rows, cal_stats = build_rows(cal_gen, cal_scores, cal_gt, args.scorer_field)

    conf_gt = load_ground_truth(args.confirmation_gt_parquet)
    conf_gen = load_generation_logs(args.confirmation_generation_glob)
    conf_scores = load_score_logs(args.confirmation_score_glob, args.scorer_field)
    conf_rows, conf_stats = build_rows(conf_gen, conf_scores, conf_gt, args.scorer_field)

    overlap = {row["image_id"] for row in cal_rows} & {row["image_id"] for row in conf_rows}
    if overlap:
        raise ValueError(
            f"calibration and confirmation pools share {len(overlap)} image_id(s); "
            "they must be drawn image-disjoint upstream"
        )

    assign_splits(
        cal_rows,
        conf_rows,
        seed=args.split_seed,
        select_frac=args.select_frac,
        certify_frac=args.certify_frac,
    )

    frame = pd.DataFrame(cal_rows + conf_rows)[
        ["split", "image_id", "expression_id", "score", "baseline_iou", "challenger_iou"]
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    print(f"calibration pool: {cal_stats}")
    print(f"confirmation pool: {conf_stats}")
    print(f"rows per split:\n{frame['split'].value_counts()}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
