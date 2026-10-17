"""Rebuild System A's audit records with two consistency rules (review J7, J8).

The archived manifests took the default box's IoU from the generation log
but chose the alternative from the scoring log.  When the generator could not
parse the VLM's answer, the scoring log's default was the top detector box
while the generation log recorded IoU 0, so the two disagreed.  Separately,
floating-point margins in (0, 1e-6) were counted as switches although ties
should keep the default.

This command rewrites each record so that
1. the default IoU is the IoU of the default box that was actually scored
   (``score_restricted_iou`` from ``check_default_frame.py``), and
2. a switch score below the tie tolerance becomes 0, and a record without a
   switch has alternative IoU equal to default IoU.

The archived manifests are read only; outputs go to a new directory with a
per-benchmark change summary.

Example:
    python -m experiments.cs2_audit.rebuild_system_a_records \
        --per-row results/default_frame_check_20261016/per_row.csv.gz \
        --manifest "RefCOCO+:20270822=<archived reduced manifest>" ... \
        --out-dir results/system_a_records_v2_20261016
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import validate_record_consistency

TIE_TOLERANCE = 1e-6


def rebuild(manifest: pd.DataFrame, per_row: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    per_row = per_row.dropna(subset=["example_id"])  # generation error rows carry no id
    if per_row["example_id"].duplicated().any():
        raise ValueError("the per-row check must have one row per example id")
    scored = per_row.set_index("example_id")["score_restricted_iou"]
    missing = ~manifest["expression_id"].isin(scored.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} records are missing from the per-row check")
    out = manifest.copy()
    new_baseline = manifest["expression_id"].map(scored).astype(float)
    tie = (out["score"] > 0) & (out["score"] < TIE_TOLERANCE)
    out.loc[tie, "score"] = 0.0
    out["baseline_iou"] = new_baseline
    no_switch = out["score"] == 0.0
    out.loc[no_switch, "challenger_iou"] = out.loc[no_switch, "baseline_iou"]
    changes = {
        "records": int(len(out)),
        "default_iou_changed": int((new_baseline - manifest["baseline_iou"]).abs().gt(1e-12).sum()),
        "default_correctness_changed": int(((new_baseline >= 0.5) != (manifest["baseline_iou"] >= 0.5)).sum()),
        "ties_set_to_no_switch": int(tie.sum()),
        "switches_before": int((manifest["score"] > 0).sum()),
        "switches_after": int((out["score"] > 0).sum()),
    }
    validate_record_consistency(out, tie_tolerance=TIE_TOLERANCE)
    return out, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-row", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True, help="BENCHMARK:SEED=path")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    per_row = pd.read_csv(args.per_row, dtype={"example_id": str})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for item in args.manifest:
        label, path = item.split("=", 1)
        benchmark, seed = label.split(":", 1)
        manifest = pd.read_csv(path, dtype={"expression_id": str})
        rebuilt, changes = rebuild(manifest, per_row[per_row["benchmark"] == benchmark])
        name = f"{benchmark.replace('+', 'plus')}_reduced_seed{seed}.csv"
        rebuilt.to_csv(args.out_dir / name, index=False)
        summary[f"{benchmark}:{seed}"] = {"source": Path(path).name, "output": name, **changes}
    (args.out_dir / "rebuild_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
