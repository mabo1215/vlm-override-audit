"""CPU-only descriptive checks for the current revision cycle.

This script does not select a policy or change any audit artifact.  It reads
the frozen one-expression-per-image manifests, reuses the already selected
threshold for each benchmark, and records (i) pairwise image-pool overlap and
(ii) continuous IoU summaries alongside the binary help/harm convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments" / "cs2_audit" / "results" / "revision_cpu_audit_20261016.json"

def _benchmarks() -> dict[str, dict[str, object]]:
    """Primary records and frozen thresholds from the registry (records.py)."""

    from experiments.cs2_audit.records import check_thresholds, primary_manifests

    check_thresholds()
    return {name: {"manifest": path, "threshold": t} for name, (path, t) in primary_manifests().items()}


def _summarize(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    frame = frame.copy()
    frame["routed"] = frame["score"] >= threshold
    frame["delta_iou"] = frame["challenger_iou"] - frame["baseline_iou"]
    frame["help"] = (frame["baseline_iou"] < 0.5) & (frame["challenger_iou"] >= 0.5)
    frame["harm"] = (frame["baseline_iou"] >= 0.5) & (frame["challenger_iou"] < 0.5)

    result: dict[str, object] = {
        "threshold": threshold,
        "rows": int(len(frame)),
        "unique_images": int(frame["image_id"].nunique()),
    }
    for split in ("train", "select", "certify", "confirm"):
        part = frame.loc[frame["split"] == split].copy()
        routed = part.loc[part["routed"]]
        result[split] = {
            "rows": int(len(part)),
            "route_count": int(part["routed"].sum()),
            "help_count": int(part["help"].sum()),
            "harm_count": int(part["harm"].sum()),
            "mean_baseline_iou": float(part["baseline_iou"].mean()),
            "mean_challenger_iou": float(part["challenger_iou"].mean()),
            "mean_delta_iou": float(part["delta_iou"].mean()),
            "routed_mean_baseline_iou": (
                float(routed["baseline_iou"].mean()) if len(routed) else None
            ),
            "routed_mean_challenger_iou": (
                float(routed["challenger_iou"].mean()) if len(routed) else None
            ),
            "routed_mean_delta_iou": (
                float(routed["delta_iou"].mean()) if len(routed) else None
            ),
        }
    return result


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, object] = {}
    benchmarks = _benchmarks()
    for name, config in benchmarks.items():
        frame = pd.read_csv(config["manifest"])
        required = {"split", "image_id", "score", "baseline_iou", "challenger_iou"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        if frame["image_id"].duplicated().any():
            raise ValueError(f"{name}: manifest is not one-expression-per-image")
        frames[name] = frame
        summaries[name] = _summarize(frame, config["threshold"])

    image_sets = {name: set(frame["image_id"].astype(int)) for name, frame in frames.items()}
    names = list(image_sets)
    overlaps: dict[str, object] = {
        "pool_sizes": {name: len(ids) for name, ids in image_sets.items()},
        "pairwise_intersections": {},
        "triple_intersection": len(set.intersection(*(image_sets[name] for name in names))),
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            key = f"{left} & {right}"
            overlaps["pairwise_intersections"][key] = len(image_sets[left] & image_sets[right])

    payload = {
        "purpose": "CPU-only revision descriptive analysis; no policy selection or rerun",
        "source_manifests": {name: str(config["manifest"].relative_to(ROOT)) for name, config in benchmarks.items()},
        "benchmarks": summaries,
        "image_pool_overlap": overlaps,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
