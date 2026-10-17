"""Export the per-expression records of the public release (review item J14).

The release carries public COCO image ids and RefCOCO-family reference ids, the
expression-level records before the one-expression-per-image reduction, the
model's boxes and candidate sets with their scores, and the IoU of every
candidate with the ground truth.  It carries no ground-truth boxes, no
expression text and no images, which stay with their original providers.

System A (``system-a``):
* ``<benchmark>_expressions.csv.gz``: every scored expression in the audit
  schema (set, COCO image id, reference id, switch score, default and
  alternative IoU), with the two consistency rules of the audited records
  (scored-default IoU, tie tolerance).  Reducing it with the legacy
  source-row rule reproduces the registered records (checked here).
* ``<benchmark>_candidates.jsonl.gz``: per expression, the candidate boxes
  (normalized), their sources, pairwise and listwise scores, and IoUs.

System B (``system-b``): the generation records of G1, G2 and G3 without
expression or answer text, with every candidate's IoU added.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_manifest import box_iou, load_ground_truth
from .rebuild_system_a_records import rebuild
from .records import ROOT, load_registry
from .reduce_manifest import _reduce_legacy_source_row

ARCHIVED = {
    "RefCOCO+": "experiments/cs2_audit/results/refcocoplus_scaleup_20260822/predictions.csv",
    "RefCOCOg": "experiments/cs2_audit/results/refcocog_scaleup_20260822/predictions_refcocog.csv",
    "RefCOCO": "experiments/cs2_audit/results/refcoco_scaleup_20260823/predictions_refcoco.csv",
}
TAGS = {"RefCOCO+": "refcocoplus", "RefCOCOg": "refcocog", "RefCOCO": "refcoco"}
TIE_TOLERANCE = 1e-6
TEXT_FIELDS = ("text", "default_text", "sample_texts")


def slug(name: str) -> str:
    return name.replace("+", "plus")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl_gz(path: Path, rows) -> None:
    # mtime=0 keeps the archive bytes reproducible
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(frame.to_csv(index=False).encode("utf-8"))


def expression_records(name: str, per_row: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / ARCHIVED[name])
    part = per_row[per_row["benchmark"] == name].copy()
    part = part.dropna(subset=["example_id"])
    part["example_id"] = part["example_id"].astype(int)
    out, _ = rebuild(manifest, part)
    return out


def check_reductions(name: str, expressions: pd.DataFrame) -> dict[str, bool]:
    """The legacy source-row reduction of the released records must equal every registered reduction."""

    entry = load_registry()["systems"]["System A"][name]
    seeds = {"20270822": entry["primary"], **entry.get("reductions", {})}
    result = {}
    for seed, rel in seeds.items():
        reduced, _ = _reduce_legacy_source_row(expressions, seed=int(seed))
        registered = pd.read_csv(ROOT / rel)
        mine = reduced[registered.columns].reset_index(drop=True)
        for column in registered.columns:
            if pd.api.types.is_float_dtype(registered[column]):
                same = np.allclose(mine[column].astype(float), registered[column].astype(float), rtol=0.0, atol=1e-12)
            else:
                same = bool((mine[column].astype(str).to_numpy() == registered[column].astype(str).to_numpy()).all())
            if len(mine) != len(registered) or not same:
                raise SystemExit(f"{name} seed {seed}: the legacy reduction of the released records differs from the registry ({column})")
        result[seed] = True
    return result


def candidate_rows(name: str, expressions: pd.DataFrame, scores_root: Path, truth: dict) -> list[dict]:
    entries: dict[str, dict] = {}
    for path in sorted(glob.glob(str(scores_root / f"cs2_scaleup_{TAGS[name]}" / "scores" / "*_scores.json"))):
        for entry in json.loads(Path(path).read_text()).get("scores", []):
            entries[str(entry["example_id"])] = entry
    rows = []
    mismatches = 0
    for record in expressions.itertuples(index=False):
        entry = entries[str(record.expression_id)]
        sources = list(entry["candidate_sources"])
        default = sources.index("restricted")
        gt = truth[str(record.expression_id)]
        ious = [float(box_iou(tuple(box), gt)) for box in entry["candidate_boxes"]]
        pairwise = [float(item["pairwise_score"]) for item in entry["candidate_scores"]]
        listwise = [float(item["listwise_posterior"]) for item in entry["candidate_scores"]]
        best = int(np.argmax(pairwise))
        margin = pairwise[best] - pairwise[default]
        alternative = best if margin >= TIE_TOLERANCE else default
        score = float(margin) if margin >= TIE_TOLERANCE else 0.0
        mismatches += (
            abs(score - float(record.score)) > 1e-9
            or abs(ious[default] - float(record.baseline_iou)) > 1e-9
            or abs(ious[alternative] - float(record.challenger_iou)) > 1e-9
        )
        rows.append(
            {
                "benchmark": name,
                "set": record.split,
                "coco_image_id": int(record.image_id),
                "ref_id": str(record.expression_id),
                "file_name": entry["file_name"],
                "candidate_boxes": [list(map(float, box)) for box in entry["candidate_boxes"]],
                "candidate_sources": sources,
                "candidate_pairwise_scores": pairwise,
                "candidate_listwise_posteriors": listwise,
                "candidate_ious": ious,
                "default_index": default,
                "alternative_index": alternative,
                "switch_score": score,
            }
        )
    if mismatches:
        raise SystemExit(f"{name}: {mismatches} candidate sets do not reproduce the released records")
    return rows


def cmd_system_a(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_row = pd.read_csv(ROOT / "experiments/cs2_audit/results/default_frame_check_20261016/per_row.csv.gz")
    gt: dict[str, list[str]] = {}
    for item in args.gt:
        name, path = item.split("=", 1)
        gt.setdefault(name, []).append(path)
    summary = {}
    for name in ARCHIVED:
        expressions = expression_records(name, per_row)
        reductions = check_reductions(name, expressions)
        truth: dict = {}
        for path in gt[name]:
            for key, box in load_ground_truth(path).items():
                truth.setdefault(key, box)
        rows = candidate_rows(name, expressions, args.scores_root, truth)
        expr_path = args.out_dir / f"{slug(name)}_expressions.csv.gz"
        cand_path = args.out_dir / f"{slug(name)}_candidates.jsonl.gz"
        write_csv_gz(expr_path, expressions)
        write_jsonl_gz(cand_path, rows)
        summary[name] = {
            "expressions": int(len(expressions)),
            "images": int(expressions["image_id"].nunique()),
            "by_set": {k: int(v) for k, v in expressions["split"].value_counts().items()},
            "legacy_reduction_reproduces_registry": reductions,
            "candidate_sets_reproduce_records": True,
            "files": {expr_path.name: sha256(expr_path), cand_path.name: sha256(cand_path)},
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: {f: v[f] for f in ("expressions", "images")} for k, v in summary.items()}))


def cmd_system_b(args: argparse.Namespace) -> None:
    from .join_system_b_labels import load_generation, load_truth

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for arm, (pattern, gt_csv) in {"g1": (args.g1, args.g1_gt), "g2": (args.g2, args.g2_gt), "g3": (args.g3, args.g3_gt)}.items():
        paths = sorted(Path(p) for p in glob.glob(pattern))
        records = load_generation(paths).to_dict("records")
        truth = load_truth(gt_csv)
        rows = []
        for record in sorted(records, key=lambda r: str(r["record_key"])):
            row = {k: v for k, v in record.items() if k not in TEXT_FIELDS and not (isinstance(v, float) and np.isnan(v))}
            row["image_path"] = Path(str(record.get("image_path", record.get("file_name")))).name
            gt = truth[str(record["record_key"])]
            row["candidate_ious"] = [float(box_iou(tuple(c["box"]), gt)) for c in record["candidates"]]
            for key in ("default_index", "chosen_index", "listwise_chosen_index"):
                if key in row and row[key] is not None:
                    row[key] = int(row[key])
            rows.append(row)
        out = args.out_dir / f"{arm}_records.jsonl.gz"
        write_jsonl_gz(out, rows)
        summary[arm] = {"records": len(rows), "generation_sha256": {p.name: sha256(p) for p in paths}, "file": out.name, "sha256": sha256(out)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v["records"] for k, v in summary.items()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("system-a")
    p.add_argument("--scores-root", type=Path, required=True)
    p.add_argument("--gt", action="append", required=True, help="BENCHMARK=parquet (repeatable)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=cmd_system_a)
    p = sub.add_parser("system-b")
    for arm in ("g1", "g2", "g3"):
        p.add_argument(f"--{arm}", required=True, help="glob of the generation shards")
        p.add_argument(f"--{arm}-gt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=cmd_system_b)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
