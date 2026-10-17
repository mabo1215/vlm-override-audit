"""Prepare label-free generation inputs and separate ground-truth files.

``system-a``  System B is run on exactly System A's audited records: the same
              benchmarks, images, expressions and set assignments.  The
              expression text is copied from System A's scoring logs, so both
              systems read the identical string.
``refclef``   One expression per RefCLEF (ReferItGame, SAIAPR TC-12) test
              image, chosen by a label-free hash of public ids, and a fixed
              10/30/30/30 image split (Design.md section 15.5).
``verifier-pool``  New training images for verifier V1 (section 15.4).
``rebuild-pool``   Rebuild a System A cohort parquet that survives only as
              logs (RefCOCOg) from the public train split, checked against the
              scoring logs and the archived per-row default IoUs.

Each command writes two files: ``records.csv`` (no label columns; the only
input of ``system_b.py``) and ``gt.csv`` (normalized ground-truth boxes, read
only by ``join_system_b_labels.py``).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RECORD_COLUMNS = ["benchmark", "split", "expression_id", "image_id", "file_name", "text", "image_path"]
GT_COLUMNS = ["benchmark", "expression_id", "x1", "y1", "x2", "y2"]
REFCLEF_SEED = 20271016
REFCLEF_FRACTIONS = {"train": 0.10, "select": 0.30, "certify": 0.30, "confirm": 0.30}
LOG_TAGS = {"RefCOCO+": "refcocoplus", "RefCOCOg": "refcocog", "RefCOCO": "refcoco"}


def _first_text(row: dict) -> str:
    """System A's rule: ``captions[0]`` if present, else ``sentences[0].raw``."""

    captions = row.get("captions")
    if captions is not None and len(captions):
        return str(captions[0])
    sentences = row.get("sentences")
    if sentences is not None and len(sentences):
        first = sentences[0]
        return str(first["raw"] if isinstance(first, dict) else first)
    raise ValueError(f"no expression text for ref {row.get('ref_id')}")


def system_a_inputs(manifests: dict[str, Path], logs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, truth = [], []
    for benchmark, manifest_path in manifests.items():
        tag = LOG_TAGS[benchmark]
        base = logs_root / f"cs2_scaleup_{tag}"
        manifest = pd.read_csv(manifest_path, dtype={"expression_id": str})
        texts, files = {}, {}
        for path in sorted(base.glob("scores/*_scores.json")):
            for entry in json.loads(path.read_text()).get("scores", []):
                texts[str(entry["example_id"])] = entry["referring_expression"]
                files[str(entry["example_id"])] = entry["file_name"]
        parquet = pd.concat(
            [pd.read_parquet(p) for p in sorted(base.glob(f"{tag}_scaleup_*.parquet"))], ignore_index=True
        ).drop_duplicates("ref_id")
        parquet["ref_id"] = parquet["ref_id"].astype(str)
        parquet = parquet.set_index("ref_id")
        for row in manifest.itertuples(index=False):
            ref = str(row.expression_id)
            meta = parquet.loc[ref].to_dict()
            text = texts.get(ref)
            if text is None:
                raise ValueError(f"{benchmark} {ref}: not in System A scoring logs")
            if text != _first_text(meta):
                raise ValueError(f"{benchmark} {ref}: scoring-log text differs from the parquet's first text")
            info = json.loads(meta["raw_image_info"])
            width, height = float(info["width"]), float(info["height"])
            x1, y1, x2, y2 = (float(v) for v in meta["bbox"])
            records.append(
                {
                    "benchmark": benchmark, "split": row.split, "expression_id": ref, "image_id": int(row.image_id),
                    "file_name": files[ref], "text": text, "image_path": files[ref],
                }
            )
            truth.append({"benchmark": benchmark, "expression_id": ref, "x1": x1 / width, "y1": y1 / height, "x2": x2 / width, "y2": y2 / height})
    return pd.DataFrame(records, columns=RECORD_COLUMNS), pd.DataFrame(truth, columns=GT_COLUMNS)


def _hash_key(seed: int, image_id: int, ref_id: int) -> str:
    return hashlib.sha256(f"{seed}\x1fpool\x1f{image_id}\x1f{ref_id}".encode("utf-8")).hexdigest()


def refclef_inputs(parquet_paths: list[str], image_dir: Path, seed: int = REFCLEF_SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    for path in parquet_paths:
        frame = pd.read_parquet(path)
        for row in frame.to_dict("records"):
            info = row["image_info"]
            image_id = int(info["id"])
            refs = list(row["ref_list"])
            if not refs:
                continue
            best = min(refs, key=lambda ref: _hash_key(seed, image_id, int(ref["ref_info"]["ref_id"])))
            ref_info, ann = best["ref_info"], best["ann_info"]
            x, y, w, h = (float(v) for v in ann["bbox"])  # COCO-style x, y, width, height in pixels
            width, height = float(info["width"]), float(info["height"])
            out = image_dir / f"{image_id}.jpg"
            if not out.exists():
                out.write_bytes(row["image"]["bytes"])
            chosen.append(
                {
                    "image_id": image_id, "expression_id": str(int(ref_info["ref_id"])),
                    "file_name": str(info["file_name"]), "text": str(ref_info["sentences"][0]["raw"]),
                    "image_path": str(out), "box": (x / width, y / height, (x + w) / width, (y + h) / height),
                    "candidates_on_image": len(refs),
                }
            )
    table = pd.DataFrame(chosen).drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    if table["file_name"].str.contains("COCO_", regex=False).any():
        raise ValueError("RefCLEF inputs must not contain COCO images")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(table))
    bounds = np.cumsum([int(round(f * len(table))) for f in REFCLEF_FRACTIONS.values()])[:-1]
    split = np.empty(len(table), dtype=object)
    for name, part in zip(REFCLEF_FRACTIONS, np.split(order, bounds)):
        split[part] = name
    table["split"] = split
    table["benchmark"] = "RefCLEF"
    truth = pd.DataFrame(
        [{"benchmark": "RefCLEF", "expression_id": r.expression_id, "x1": r.box[0], "y1": r.box[1], "x2": r.box[2], "y2": r.box[3]} for r in table.itertuples()],
        columns=GT_COLUMNS,
    )
    summary = {
        "images": int(len(table)),
        "split_counts": table["split"].value_counts().to_dict(),
        "seed": seed,
        "rule": "one ref per image: minimum SHA-256 of 'seed, pool, image_id, ref_id'; first sentence (raw)",
        "category_filter": "none",
        "source_files": [Path(p).name for p in parquet_paths],
    }
    return table[RECORD_COLUMNS], truth, summary


VERIFIER_SEED = 20271017


def _log_entries(path: str) -> tuple[list[dict], int]:
    """Entries of a System A scoring log (``scores``) or candidate log (``logs``).

    Returns the entries that carry an example id and the number of logged
    generation errors without one (such records are in no audited set).
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("scores") or data.get("logs") or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: unexpected log layout")
    kept = [e for e in entries if "example_id" in e]
    errors = [e for e in entries if "example_id" not in e]
    if any("error" not in e for e in errors):
        raise ValueError(f"{path}: an entry has neither an example id nor an error")
    return kept, len(errors)


def _coco_image_id(entry: dict) -> int:
    if entry.get("image_id") is not None:
        return int(entry["image_id"])
    match = re.search(r"(\d+)\.jpg$", str(entry["file_name"]))
    if match is None:
        raise ValueError(f"no image id in log entry {entry.get('example_id')}")
    return int(match.group(1))


def logged_image_ids(patterns: list[str]) -> tuple[set[int], int]:
    """COCO image ids of every entry in System A's scoring and candidate logs."""

    ids: set[int] = set()
    files = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            files += 1
            ids |= {_coco_image_id(entry) for entry in _log_entries(path)[0]}
    return ids, files


def verifier_pool_inputs(
    train_parquets: dict[str, Path],
    exclude_parquets: list[str],
    images: int,
    seed: int = VERIFIER_SEED,
    exclude_logs: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """New training images for verifier V1 (Design.md section 15.4).

    Images are drawn from the official train splits of the three benchmarks
    and must not appear in any excluded file (System A's calibration and
    confirmation pools, pilot cohorts).  Every expression of a drawn image is
    kept; all records are in the training set.
    """

    excluded: set[int] = set()
    for pattern in exclude_parquets:
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                ids = pd.read_parquet(path, columns=["image_id"])["image_id"]
            except (KeyError, ValueError):
                continue  # a parquet without image ids cannot hold excluded images
            excluded |= set(pd.to_numeric(ids, errors="coerce").dropna().astype(int))
    log_ids, log_files = logged_image_ids(list(exclude_logs or []))
    if exclude_logs and not log_files:
        raise ValueError(f"no log files match {exclude_logs}")
    excluded |= log_ids
    frames = {}
    for benchmark, path in train_parquets.items():
        frame = pd.read_parquet(path)
        if "split" in frame and not frame["split"].astype(str).str.lower().eq("train").all():
            raise ValueError(f"{benchmark}: the verifier pool must come from the train split")
        frames[benchmark] = frame
    pool = sorted(set().union(*(set(f["image_id"].astype(int)) for f in frames.values())) - excluded)
    if len(pool) < images:
        raise ValueError(f"only {len(pool)} eligible images for {images} requested")
    rng = np.random.default_rng(seed)
    drawn = set(np.asarray(pool)[rng.choice(len(pool), size=images, replace=False)].tolist())
    records, truth = [], []
    for benchmark, frame in frames.items():
        for row in frame[frame["image_id"].astype(int).isin(drawn)].to_dict("records"):
            info = json.loads(row["raw_image_info"])
            width, height = float(info["width"]), float(info["height"])
            x1, y1, x2, y2 = (float(v) for v in row["bbox"])
            file_name = str(row.get("image_path") or row["file_name"]).rsplit("/", 1)[-1]
            ref = str(int(row["ref_id"]))
            records.append({"benchmark": benchmark, "split": "train", "expression_id": ref, "image_id": int(row["image_id"]),
                            "file_name": file_name, "text": _first_text(row), "image_path": file_name})
            truth.append({"benchmark": benchmark, "expression_id": ref, "x1": x1 / width, "y1": y1 / height, "x2": x2 / width, "y2": y2 / height})
    table = pd.DataFrame(records, columns=RECORD_COLUMNS).drop_duplicates(["benchmark", "expression_id"])
    summary = {
        "source": "verifier V1 training pool",
        "seed": seed,
        "eligible_images": len(pool),
        "excluded_images": len(excluded),
        "drawn_images": len(drawn),
        "records_by_benchmark": table["benchmark"].value_counts().to_dict(),
        "excluded_files": exclude_parquets,
        "excluded_log_patterns": list(exclude_logs or []),
        "excluded_log_files": log_files,
        "excluded_images_from_logs": len(log_ids),
    }
    return table, pd.DataFrame(truth, columns=GT_COLUMNS).drop_duplicates(["benchmark", "expression_id"]), summary


def rebuild_pool(
    log_patterns: list[str],
    public_parquets: list[str],
    per_row: pd.DataFrame | None = None,
    benchmark: str | None = None,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict]:
    """Rebuild a System A cohort parquet from the public split it was drawn from.

    The cohort's refs are the ``example_id`` of every scoring- and candidate-log
    entry.  Each ref must appear once in the public parquets with the logged
    image file and, where a scoring log exists, the logged expression text.
    With ``per_row`` (the archived default-frame check), the IoU of every
    scored default box against the rebuilt ground truth must equal the
    archived ``score_restricted_iou``; this ties the rebuilt boxes to the ones
    System A was audited with.
    """

    files: dict[str, str] = {}
    texts: dict[str, str] = {}
    scored: dict[str, dict] = {}
    log_files = []
    errors_without_id = 0
    for pattern in log_patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            log_files.append(path)
            entries, errors = _log_entries(path)
            errors_without_id += errors
            for entry in entries:
                ref = str(entry["example_id"])
                name = Path(str(entry["file_name"])).name
                if files.setdefault(ref, name) != name:
                    raise ValueError(f"ref {ref}: logs disagree on the image file")
                if "referring_expression" in entry:
                    texts[ref] = str(entry["referring_expression"])
                    scored[ref] = entry
    if not files:
        raise ValueError(f"no log entries match {log_patterns}")
    public = pd.concat([pd.read_parquet(p) for p in public_parquets], ignore_index=True)
    public["ref_id"] = public["ref_id"].astype(str)
    if public["ref_id"].duplicated().any():
        raise ValueError("the public parquets repeat a ref id")
    public = public.set_index("ref_id", drop=False)
    missing = sorted(set(files) - set(public.index), key=int)
    if missing:
        raise ValueError(f"{len(missing)} logged refs are not in the public parquets, e.g. {missing[:5]}")
    rows = public.loc[sorted(files, key=int)].reset_index(drop=True)
    file_mismatch = text_mismatch = 0
    truth: dict[str, tuple[float, float, float, float]] = {}
    for row in rows.to_dict("records"):
        ref = str(row["ref_id"])
        info = json.loads(row["raw_image_info"])
        file_mismatch += Path(str(info["file_name"])).name != files[ref]
        if ref in texts:
            text_mismatch += texts[ref] != _first_text(row)
        width, height = float(info["width"]), float(info["height"])
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        truth[ref] = (x1 / width, y1 / height, x2 / width, y2 / height)
    if file_mismatch or text_mismatch:
        raise ValueError(f"rebuilt pool disagrees with the logs: {file_mismatch} files, {text_mismatch} texts")
    summary: dict[str, object] = {
        "refs": len(rows),
        "images": int(rows["image_id"].nunique()),
        "scored_refs": len(scored),
        "log_errors_without_id": errors_without_id,
        "log_files": [Path(p).name for p in log_files],
        "public_parquets": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in public_parquets},
        "file_mismatches": file_mismatch,
        "text_mismatches": text_mismatch,
    }
    if per_row is not None:
        from .system_b import box_iou

        check = per_row[(per_row["benchmark"] == benchmark) & per_row["example_id"].notna() & per_row["score_has_restricted"].astype(bool)]
        worst, checked = 0.0, 0
        for item in check.itertuples(index=False):
            ref = str(int(item.example_id))
            entry = scored.get(ref)
            if entry is None:
                raise ValueError(f"ref {ref} has a per-row IoU but no scoring-log entry")
            box = entry["candidate_boxes"][int(item.score_restricted_index)]
            worst = max(worst, abs(box_iou(box, truth[ref]) - float(item.score_restricted_iou)))
            checked += 1
        if checked == 0:
            raise ValueError(f"no per-row IoUs for {benchmark}")
        if worst > tolerance:
            raise ValueError(f"rebuilt boxes do not reproduce the archived default IoUs (max difference {worst:.3g})")
        summary.update({"iou_checked_refs": checked, "iou_max_abs_difference": worst, "iou_tolerance": tolerance})
    return rows, summary


def write(out_dir: Path, records: pd.DataFrame, truth: pd.DataFrame, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_dir / "records.csv", index=False)
    truth.to_csv(out_dir / "gt.csv", index=False)
    summary = {**summary, "records_sha256": hashlib.sha256((out_dir / "records.csv").read_bytes()).hexdigest()}
    (out_dir / "inputs_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("system-a")
    p.add_argument("--manifest", action="append", required=True, help="BENCHMARK=path to System A's audited manifest")
    p.add_argument("--logs-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("verifier-pool")
    p.add_argument("--train-parquet", action="append", required=True, help="BENCHMARK=path to the official train split")
    p.add_argument("--exclude-parquet", action="append", required=True, help="glob of parquets whose image_ids are excluded")
    p.add_argument("--exclude-logs", action="append", default=[], help="glob of System A scoring/candidate logs whose images are excluded")
    p.add_argument("--images", type=int, default=4000)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("rebuild-pool")
    p.add_argument("--logs", action="append", required=True, help="glob of System A scoring and candidate logs of one benchmark")
    p.add_argument("--public-parquet", action="append", required=True, help="public split parquet(s) the cohort was drawn from")
    p.add_argument("--per-row", type=Path, help="archived default-frame check (per_row.csv.gz)")
    p.add_argument("--benchmark", help="benchmark name in the per-row file, e.g. RefCOCOg")
    p.add_argument("--out", type=Path, required=True, help="output parquet, e.g. .../refcocog_scaleup_rebuilt.parquet")
    p = sub.add_parser("refclef")
    p.add_argument("--parquet-glob", required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "system-a":
        manifests = dict(item.split("=", 1) for item in args.manifest)
        records, truth = system_a_inputs({k: Path(v) for k, v in manifests.items()}, args.logs_root)
        summary = {
            "source": "System A audited records",
            "manifests": {k: Path(v).name for k, v in manifests.items()},
            "counts": records.groupby(["benchmark", "split"]).size().unstack(fill_value=0).to_dict("index"),
        }
        write(args.out_dir, records, truth, summary)
    elif args.command == "verifier-pool":
        parquets = {k: Path(v) for k, v in (item.split("=", 1) for item in args.train_parquet)}
        records, truth, summary = verifier_pool_inputs(parquets, args.exclude_parquet, args.images, exclude_logs=args.exclude_logs)
        write(args.out_dir, records, truth, summary)
    elif args.command == "rebuild-pool":
        per_row = pd.read_csv(args.per_row) if args.per_row else None
        rows, summary = rebuild_pool(args.logs, args.public_parquet, per_row, args.benchmark)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(args.out, index=False)
        summary["output_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
        args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        records, truth, summary = refclef_inputs(sorted(glob.glob(args.parquet_glob)), args.image_dir)
        write(args.out_dir, records, truth, summary)


if __name__ == "__main__":
    main()
