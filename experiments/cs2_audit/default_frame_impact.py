"""Measure how the default-box coordinate-frame error affects the audit.

The generator normalized the VLM's direct-answer box by the original image
size, although the VLM answers in the frame of its resized input.  The
per-row table written by ``check_default_frame.py`` holds the IoU of the box
as generated (``used_iou``) and of the box converted with the resized frame
(``fixed_iou``).  This command substitutes ``fixed_iou`` for the default box in
an audited manifest, keeps every switch score unchanged, reruns the same
select -> certify -> confirm protocol, and reports both runs side by side.

Rows the generator could not parse keep their recorded values.  For rows
without a switch (score exactly zero) the alternative is the default itself,
so its IoU is replaced as well.

Example:
    python -m experiments.cs2_audit.default_frame_impact \
        --per-row results/default_frame_check_20261016/per_row.csv.gz \
        --manifest "RefCOCO+=<audited manifest csv>" \
        --output results/default_frame_check_20261016/impact_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import Policy, run_protocol

REPORTED = (
    "route_count",
    "help_count",
    "harm_count",
    "conditional_harm",
    "cp_upper",
    "net_accuracy_gain",
    "incumbent_correct",
    "status",
)


def corrected_manifest(manifest: pd.DataFrame, per_row: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a copy of ``manifest`` with frame-corrected default IoUs."""

    rows = per_row.set_index("example_id")
    joined = manifest.join(rows, on="expression_id", rsuffix="_check")
    if joined["status"].isna().any():
        raise ValueError("every manifest row must appear in the per-row frame check")
    fixed = manifest.copy()
    pixel = (joined["status"] == "pixel").to_numpy()
    fixed.loc[pixel, "baseline_iou"] = joined.loc[pixel, "fixed_iou"].to_numpy()
    same_box = joined["score_restricted_matches_gen"].astype("boolean").fillna(False).to_numpy()
    no_switch = pixel & (manifest["score"].to_numpy() == 0.0) & same_box
    fixed.loc[no_switch, "challenger_iou"] = joined.loc[no_switch, "fixed_iou"].to_numpy()
    flips = {}
    for split in ("select", "certify", "confirm"):
        mask = (manifest["split"] == split).to_numpy()
        before = manifest.loc[mask, "baseline_iou"] >= 0.5
        after = fixed.loc[mask, "baseline_iou"] >= 0.5
        flips[split] = int((before != after).sum())
    stats = {
        "rows": int(len(manifest)),
        "pixel_rows": int(pixel.sum()),
        "out_of_range_rows": int(joined["out_of_range"].astype("boolean").fillna(False).sum()),
        "no_switch_rows_corrected": int(no_switch.sum()),
        "default_correct_flips": flips,
    }
    return fixed, stats


def audit(frame: pd.DataFrame, thresholds: list[float], alpha: float, delta: float) -> dict[str, object]:
    policies = [Policy(name=f"score_ge_{t:.2f}", threshold=t) for t in thresholds]
    result = run_protocol(frame, policies, alpha=alpha, delta=delta)
    return {
        "selected_policy": result["certify"]["policy"],
        **{split: {key: result[split][key] for key in REPORTED} for split in ("certify", "confirm")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-row", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True, help="BENCHMARK=path to an audited manifest")
    parser.add_argument("--thresholds", default="0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
    per_row = pd.read_csv(args.per_row, dtype={"example_id": str})
    summary: dict[str, object] = {}
    for item in args.manifest:
        benchmark, path = item.split("=", 1)
        manifest = pd.read_csv(path, dtype={"expression_id": str})
        fixed, stats = corrected_manifest(manifest, per_row[per_row["benchmark"] == benchmark])
        summary[benchmark] = {
            "manifest": Path(path).name,
            **stats,
            "as_generated": audit(manifest, thresholds, args.alpha, args.delta),
            "frame_corrected": audit(fixed, thresholds, args.alpha, args.delta),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
