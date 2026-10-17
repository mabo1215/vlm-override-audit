"""Figure 1: audited confirmation examples, chosen without looking at outcomes (review item J5).

Selection.  On System A's audited RefCOCO+ record, the frozen rule (tau = 0.4)
puts every confirmation image into one category: help, harm or no switch.
Within each category, images are ordered by SHA-256(seed, category, image id)
with the seed of the original pre-declaration (``select_qualitative.py``).  The
first image whose COCO licence permits reuse under CC BY (licence 4,
Attribution; 7, no known copyright restrictions; 8, U.S. Government Work) is
shown, provided that its Flickr photo page still exists with a CC BY-compatible
licence so that the creator can be credited (the page checks are recorded in a
file, because they cannot be repeated bit for bit).  Images that fail either
check are skipped in that same order, and the skips are recorded.

Boxes.  The default and alternative boxes come from System A's archived score
logs (the scored ``restricted`` candidate and the candidate the audited rule
switches to); the IoUs recomputed against the public ground truth must equal
the audited record.

Usage (two steps; the second needs the selected COCO images on disk):
    python -m experiments.cs2_audit.qualitative_figure select ...
    python -m experiments.cs2_audit.qualitative_figure render ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .records import primary_manifests
from .select_qualitative import _key
from .verifier import _iou, records_from_system_a

SEED = 20270822
ALLOWED_LICENSES = {4, 7, 8}
CATEGORIES = ("help", "harm", "no_route")
BENCHMARK = "RefCOCO+"
SCORE_DIR = "cs2_scaleup_refcocoplus"


def categorize(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    part = frame[frame["split"] == "confirm"].copy()
    base = part["baseline_iou"].to_numpy(dtype=float) >= 0.5
    alt = part["challenger_iou"].to_numpy(dtype=float) >= 0.5
    routed = part["score"].to_numpy(dtype=float) >= threshold
    part["category"] = "none"
    part.loc[routed & ~base & alt, "category"] = "help"
    part.loc[routed & base & ~alt, "category"] = "harm"
    part.loc[~routed, "category"] = "no_route"
    return part


def select(frame: pd.DataFrame, threshold: float, licenses: pd.DataFrame, flickr: dict) -> dict:
    part = categorize(frame, threshold)
    lic = licenses.set_index("image_id")
    out = {"seed": SEED, "threshold": threshold, "allowed_licenses": sorted(ALLOWED_LICENSES), "categories": {}}
    for category in CATEGORIES:
        cands = part[part["category"] == category].copy()
        cands["key"] = cands["image_id"].map(lambda v: _key(SEED, category, v))
        cands = cands.sort_values("key", kind="mergesort")
        skipped = []
        chosen = None
        for row in cands.itertuples(index=False):
            info = lic.loc[int(row.image_id)]
            if int(info["license"]) not in ALLOWED_LICENSES:
                skipped.append({"image_id": int(row.image_id), "license": int(info["license"]), "reason": "COCO licence"})
                continue
            check = flickr.get(str(int(row.image_id)))
            if check is None:
                raise SystemExit(f"{category}: check the Flickr page of image {int(row.image_id)} ({info['flickr_url']}) and record it")
            if check["status"] != "ok":
                skipped.append({"image_id": int(row.image_id), "license": int(info["license"]), "reason": f"Flickr page {check['status']}"})
                continue
            chosen = {
                "image_id": int(row.image_id),
                "expression_id": str(row.expression_id),
                "score": float(row.score),
                "baseline_iou": float(row.baseline_iou),
                "challenger_iou": float(row.challenger_iou),
                "file_name": str(info["file_name"]),
                "license": int(info["license"]),
                "license_name": str(info["license_name"]),
                "license_url": str(info["license_url"]),
                "flickr_url": str(info["flickr_url"]),
                "flickr_photo_id": check["flickr_photo_id"],
                "flickr_owner": check["owner"],
                "flickr_title": check["title"],
                "flickr_licence": check["licence"],
            }
            break
        out["categories"][category] = {"candidates": int(len(cands)), "skipped_for_license": skipped, "selected": chosen}
    return out


def attach_boxes(selection: dict, score_root: Path, gt_csv: Path) -> dict:
    """Default and alternative boxes from the archived score logs; IoUs checked against the record."""

    ids = {c["selected"]["expression_id"]: name for name, c in selection["categories"].items() if c["selected"]}
    records = records_from_system_a(
        sorted((score_root / SCORE_DIR / "scores").glob("*confirmation*_scores.json")), BENCHMARK, {k: "confirm" for k in ids}
    )
    truth = pd.read_csv(gt_csv, dtype={"expression_id": str})
    truth = truth[truth["benchmark"] == BENCHMARK].set_index("expression_id")
    for record in records:
        category = ids[record["expression_id"]]
        chosen = selection["categories"][category]["selected"]
        gt = truth.loc[record["expression_id"], ["x1", "y1", "x2", "y2"]].astype(float).tolist()
        default = record["candidates"][record["default_index"]]["box"]
        alternative = record["candidates"][record["chosen_index"]]["box"]
        iou_b, iou_c = _iou(default, gt), _iou(alternative, gt)
        if not (np.isclose(iou_b, chosen["baseline_iou"], atol=1e-9) and np.isclose(iou_c, chosen["challenger_iou"], atol=1e-9)):
            raise SystemExit(f"{category}: recomputed IoUs {iou_b:.6f}/{iou_c:.6f} differ from the audited record")
        chosen.update(default_box=default, alternative_box=alternative, gt_box=gt, expression=record["text"],
                      iou_check="recomputed IoUs equal the audited record")
    return selection


def attach_release_boxes(selection: dict, records_dir: Path) -> dict:
    """The same boxes from the public release's candidate records (no logs or annotations needed)."""

    import gzip

    ids = {c["selected"]["expression_id"]: name for name, c in selection["categories"].items() if c["selected"]}
    with gzip.open(records_dir / "RefCOCOplus_candidates.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["ref_id"] not in ids:
                continue
            chosen = selection["categories"][ids[row["ref_id"]]]["selected"]
            iou_b, iou_c = row["candidate_ious"][row["default_index"]], row["candidate_ious"][row["alternative_index"]]
            if not (np.isclose(iou_b, chosen["baseline_iou"], atol=1e-9) and np.isclose(iou_c, chosen["challenger_iou"], atol=1e-9)):
                raise SystemExit(f"{row['ref_id']}: released IoUs differ from the audited record")
            chosen.update(default_box=row["candidate_boxes"][row["default_index"]],
                          alternative_box=row["candidate_boxes"][row["alternative_index"]],
                          iou_check="released IoUs equal the audited record")
    return selection


def render(selection: dict, image_root: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    from .make_diagnosis_figures import INK  # noqa: F401  (shared font set-up)

    titles = {"help": "(a) help", "harm": "(b) harm", "no_route": "(c) no switch"}
    colours = {"default": "#d62728", "alternative": "#1f5fbf", "ground truth": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
    for ax, category in zip(axes, CATEGORIES):
        chosen = selection["categories"][category]["selected"]
        image = Image.open(image_root / chosen["file_name"]).convert("RGB")
        w, h = image.size
        ax.imshow(image)
        boxes = [("ground truth", chosen["gt_box"], (0, (3, 1.5)))]
        boxes.append(("default", chosen["default_box"], "-"))
        if category != "no_route":
            boxes.append(("alternative", chosen["alternative_box"], "-"))
        for label, (x1, y1, x2, y2), style in boxes:
            ax.add_patch(Rectangle((x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h, fill=False,
                                   edgecolor=colours[label], linewidth=1.6, linestyle=style))
        ax.set_xticks([])
        ax.set_yticks([])
        if category == "no_route":
            iou = f"default IoU {chosen['baseline_iou']:.2f}, no switch"
        else:
            iou = f"IoU {chosen['baseline_iou']:.2f} $\\rightarrow$ {chosen['challenger_iou']:.2f}"
        ax.set_title(f"{titles[category]}: “{chosen['expression']}”", fontsize=6.6, loc="left")
        ax.set_xlabel(iou, fontsize=6.6)
    handles = [plt.Line2D([], [], color=c, linewidth=1.6, linestyle=(0, (3, 1.5)) if k == "ground truth" else "-", label=k)
               for k, c in colours.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=300, metadata={"CreationDate": None})  # reproducible bytes
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="choose the examples and attach their boxes (no images needed)")
    p.add_argument("--licenses", type=Path, required=True, help="COCO train2014 licence table (image_id, file_name, license, ...)")
    p.add_argument("--score-root", type=Path, help="directory holding cs2_scaleup_refcocoplus/scores/")
    p.add_argument("--gt-csv", type=Path)
    p.add_argument("--release-records", type=Path, help="released System A candidate records (instead of logs and ground truth)")
    p.add_argument("--flickr-checks", type=Path, required=True, help="recorded Flickr page checks")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("render", help="draw the figure from a selection file")
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        path, threshold = primary_manifests("System A")[BENCHMARK]
        frame = pd.read_csv(path, dtype={"expression_id": str})
        flickr = json.loads(args.flickr_checks.read_text(encoding="utf-8"))
        selection = select(frame, threshold, pd.read_csv(args.licenses), flickr["checks"])
        selection["flickr_checks"] = {k: flickr[k] for k in ("checked_utc", "method", "rule")}
        if args.release_records is not None:
            selection = attach_release_boxes(selection, args.release_records)
        else:
            selection = attach_boxes(selection, args.score_root, args.gt_csv)
        selection["record_file"] = str(path.name)
        args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: (v["selected"]["image_id"], v["selected"]["license"], len(v["skipped_for_license"])) for k, v in selection["categories"].items()}))
    else:
        render(json.loads(args.selection.read_text(encoding="utf-8")), args.image_root, args.output)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
