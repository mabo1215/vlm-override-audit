"""Verifier V1: an independent, harm-aware veto for proposed switches (G2).

Design.md section 15.4.  For a record with default box b and candidate box c,
V1 predicts Pr(harm | b, c): the probability that replacing a correct default
by c makes the answer wrong.  It never sees a re-ranker score, an IoU with the
ground truth or any other label-derived quantity:

* image--text features: SigLIP and CLIP similarity between the expression and
  each box crop (10% context padding), and their candidate-minus-default
  differences (computed on the GPU host by ``crop_similarities``);
* geometry: IoU(b, c), log area ratio, centre offset, aspect ratios and
  distances to the image border.

Training uses training-set records only.  The model is chosen by 5-fold
image-grouped cross-validation on 80% of training images and calibrated
isotonically on the held-out 20%; the frozen model is saved with its SHA-256.
``GatedPolicy`` switches only if s >= tau and v <= nu, and plugs into the
existing select -> certify -> confirm protocol unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .audit import Policy

IOU = 0.5
SIM_MODELS = ("siglip", "clip")
FORBIDDEN = re.compile(r"iou_gt|gt_|ground|label|help|harm|correct|score|theta|switch", re.IGNORECASE)
GEOMETRY = (
    "iou_bc",
    "log_area_ratio",
    "centre_dx",
    "centre_dy",
    "aspect_b",
    "aspect_c",
    "border_b",
    "border_c",
    "area_b",
    "area_c",
)


# ---------------------------------------------------------------------------
# features (pure functions)
# ---------------------------------------------------------------------------


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def geometry_features(b: Sequence[float], c: Sequence[float]) -> dict[str, float]:
    """Label-free geometry of a (default, candidate) pair in normalized coordinates."""

    def area(box):
        return max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))

    def aspect(box):
        return math.log(max(1e-6, box[2] - box[0]) / max(1e-6, box[3] - box[1]))

    def border(box):
        return min(box[0], box[1], 1.0 - box[2], 1.0 - box[3])

    return {
        "iou_bc": _iou(b, c),
        "log_area_ratio": math.log(area(c) / area(b)),
        "centre_dx": (c[0] + c[2]) / 2 - (b[0] + b[2]) / 2,
        "centre_dy": (c[1] + c[3]) / 2 - (b[1] + b[3]) / 2,
        "aspect_b": aspect(b),
        "aspect_c": aspect(c),
        "border_b": border(b),
        "border_c": border(c),
        "area_b": area(b),
        "area_c": area(c),
    }


def padded_crop_box(box: Sequence[float], width: int, height: int, pad: float = 0.10) -> tuple[int, int, int, int]:
    """Pixel crop around a normalized box, enlarged by ``pad`` of its size on each side."""

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = max(0.0, x1 - pad * w), min(1.0, x2 + pad * w)
    y1, y2 = max(0.0, y1 - pad * h), min(1.0, y2 + pad * h)
    eps = 1e-9  # keep exact products such as 0.55 * 100 from rounding up
    left, top = int(math.floor(x1 * width + eps)), int(math.floor(y1 * height + eps))
    right = max(left + 1, int(math.ceil(x2 * width - eps)))
    bottom = max(top + 1, int(math.ceil(y2 * height - eps)))
    return left, top, right, bottom


def feature_columns() -> list[str]:
    columns = list(GEOMETRY)
    for model in SIM_MODELS:
        columns += [f"{model}_sim_b", f"{model}_sim_c", f"{model}_sim_diff"]
    return columns


def assert_label_free(columns: Sequence[str]) -> None:
    bad = [c for c in columns if FORBIDDEN.search(c) and c not in feature_columns()]
    if bad:
        raise ValueError(f"verifier features must not include label-derived columns: {bad}")


# ---------------------------------------------------------------------------
# pairs and labels
# ---------------------------------------------------------------------------


def as_index(value) -> int | None:
    """A candidate index read back from a data frame: None/NaN -> None, float -> int."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def pair_rows(record: dict, similarities: dict[str, list[float]]) -> list[dict[str, object]]:
    """One row per (default, candidate) pair of a generation record.

    ``similarities[model][i]`` is the image--text similarity of candidate i.
    The default is candidate ``record['default_index']``.
    """

    d = as_index(record.get("default_index"))
    if d is None:
        return []
    chosen = as_index(record.get("chosen_index"))
    boxes = [c["box"] for c in record["candidates"]]
    rows = []
    for i, box in enumerate(boxes):
        if i == d:
            continue
        row = {"record_key": record["record_key"], "image_id": int(record["image_id"]), "candidate_index": i,
               "is_chosen": i == chosen}
        row.update(geometry_features(boxes[d], box))
        for model in SIM_MODELS:
            sims = similarities[model]
            row[f"{model}_sim_b"], row[f"{model}_sim_c"] = sims[d], sims[i]
            row[f"{model}_sim_diff"] = sims[i] - sims[d]
        rows.append(row)
    return rows


def harm_labels(pairs: pd.DataFrame, records: dict[str, dict], truth: dict[str, Sequence[float]]) -> np.ndarray:
    """Harm of each pair: default correct and candidate wrong (training labels only)."""

    labels = []
    for row in pairs.itertuples(index=False):
        record = records[row.record_key]
        key = record.get("truth_key")  # NaN for System B rows of a mixed frame
        gt = truth[key if isinstance(key, str) else row.record_key]
        b = record["candidates"][as_index(record["default_index"])]["box"]
        c = record["candidates"][int(row.candidate_index)]["box"]
        labels.append(_iou(b, gt) >= IOU and _iou(c, gt) < IOU)
    return np.asarray(labels, dtype=int)


SYSTEM_A_PREFIX = "SystemA:"
SYSTEM_A_DIRS = {"RefCOCO": "cs2_scaleup_refcoco", "RefCOCO+": "cs2_scaleup_refcocoplus", "RefCOCOg": "cs2_scaleup_refcocog"}


def records_from_system_a(
    score_paths: Sequence[Path],
    benchmark: str,
    splits: dict[str, str],
    tie: float = 1e-6,
    image_ids: dict[str, int] | None = None,
) -> list[dict]:
    """System A score logs as verifier records (same shape as System B records).

    ``splits`` maps expression id to its set; records outside it are skipped.
    The default is the scored ``restricted`` candidate; the chosen candidate
    follows the audited rule with the tie tolerance.  Keys carry the
    ``SystemA:`` prefix so that System A and System B records of the same
    expression never collide; ``truth_key`` is the unprefixed key of the
    ground-truth table.  An expression scored twice must have the same
    candidate set both times.
    """

    by_key: dict[str, dict] = {}
    for path in sorted(score_paths):
        for entry in json.loads(Path(path).read_text()).get("scores", []):
            key = str(entry["example_id"])
            if key not in splits or "restricted" not in entry["candidate_sources"]:
                continue
            values = [float(c["pairwise_score"]) for c in entry["candidate_scores"]]
            default = entry["candidate_sources"].index("restricted")
            best = int(np.argmax(values))
            chosen = default if best == default or values[best] - values[default] < tie else best
            record = {
                "record_key": f"{SYSTEM_A_PREFIX}{benchmark}:{key}",
                "truth_key": f"{benchmark}:{key}",
                "benchmark": benchmark,
                "split": splits[key],
                "expression_id": key,
                "image_id": (image_ids or {}).get(key),
                "file_name": entry["file_name"],
                "image_path": entry["file_name"],
                "text": entry["referring_expression"],
                "candidates": [{"box": list(b), "source": s} for b, s in zip(entry["candidate_boxes"], entry["candidate_sources"])],
                "default_index": default,
                "chosen_index": chosen,
            }
            previous = by_key.get(key)
            if previous is not None and previous["candidates"] != record["candidates"]:
                raise ValueError(f"System A scored {benchmark}:{key} twice with different candidates")
            by_key.setdefault(key, record)
    return list(by_key.values())


def export_system_a(scores_root: Path, records_csv: Path) -> tuple[list[dict], dict]:
    """System A records for every System B record whose expression System A scored.

    The System B record table supplies each expression's set and image id
    (System B reuses System A's primary records and set assignments).
    """

    table = pd.read_csv(records_csv, dtype={"expression_id": str})
    records: list[dict] = []
    coverage: dict[str, dict] = {}
    for benchmark, rows in table.groupby("benchmark"):
        paths = sorted((scores_root / SYSTEM_A_DIRS[benchmark] / "scores").glob("*_scores.json"))
        splits = dict(zip(rows["expression_id"], rows["split"]))
        found = records_from_system_a(paths, benchmark, splits, image_ids=dict(zip(rows["expression_id"], rows["image_id"].astype(int))))
        records += found
        counts = pd.Series([r["split"] for r in found]).value_counts().to_dict()
        coverage[benchmark] = {
            "score_files": len(paths),
            "system_b_records": int(len(rows)),
            "system_a_records": len(found),
            "missing": int(len(rows) - len(found)),
            "by_split": {k: int(v) for k, v in counts.items()},
        }
    return records, coverage


def attach_verifier(manifest: pd.DataFrame, predictions: pd.DataFrame, benchmark: str, key_prefix: str = "") -> pd.DataFrame:
    """Add ``verifier_harm`` to an audit manifest; records without a switch get 0.

    ``key_prefix`` is ``SYSTEM_A_PREFIX`` for a System A manifest.
    """

    keyed = predictions.set_index("record_key")["verifier_harm"]
    keys = key_prefix + benchmark + ":" + manifest["expression_id"].astype(str)
    out = manifest.copy()
    out["verifier_harm"] = keys.map(keyed)
    missing = out["verifier_harm"].isna() & (out["score"] > 0)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} switching records have no verifier prediction")
    out["verifier_harm"] = out["verifier_harm"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# training (CPU)
# ---------------------------------------------------------------------------


@dataclass
class FrozenVerifier:
    model: object
    calibrator: object
    columns: list[str]
    metadata: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert_label_free(list(frame.columns.intersection(self.columns)))
        raw = self.model.predict_proba(frame[self.columns].to_numpy(dtype=float))[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


def _candidates(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}
    for c in (0.01, 0.1, 1.0):
        models[f"logistic_C{c}"] = lambda c=c: make_pipeline(
            StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
        )
    for depth in (3, 5):
        for rate in (0.05, 0.1):
            models[f"hgb_d{depth}_lr{rate}"] = lambda depth=depth, rate=rate: HistGradientBoostingClassifier(
                max_depth=depth, learning_rate=rate, max_iter=300, class_weight="balanced", random_state=seed
            )
    return models


def train_verifier(pairs: pd.DataFrame, labels: np.ndarray, *, seed: int = 20271016) -> FrozenVerifier:
    """Model choice by grouped 5-fold CV on 80% of images; isotonic calibration on 20%."""

    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    columns = feature_columns()
    assert_label_free(columns)
    images = np.array(sorted(pairs["image_id"].unique()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(images, size=max(1, int(round(0.2 * len(images)))), replace=False).tolist())
    calib_mask = pairs["image_id"].isin(held).to_numpy()
    fit_x = pairs.loc[~calib_mask, columns].to_numpy(dtype=float)
    fit_y, fit_groups = labels[~calib_mask], pairs.loc[~calib_mask, "image_id"].to_numpy()
    scores = {}
    for name, make in _candidates(seed).items():
        fold_auc = []
        for train_idx, test_idx in GroupKFold(n_splits=5).split(fit_x, fit_y, fit_groups):
            if len(set(fit_y[test_idx])) < 2 or len(set(fit_y[train_idx])) < 2:
                continue
            model = make().fit(fit_x[train_idx], fit_y[train_idx])
            fold_auc.append(roc_auc_score(fit_y[test_idx], model.predict_proba(fit_x[test_idx])[:, 1]))
        scores[name] = float(np.mean(fold_auc)) if fold_auc else float("nan")
    best = max((n for n in scores if not math.isnan(scores[n])), key=lambda n: scores[n])
    model = _candidates(seed)[best]().fit(fit_x, fit_y)
    raw = model.predict_proba(pairs.loc[calib_mask, columns].to_numpy(dtype=float))[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, labels[calib_mask])
    metadata = {
        "selected_model": best,
        "cv_auroc": scores,
        "fit_pairs": int((~calib_mask).sum()),
        "calibration_pairs": int(calib_mask.sum()),
        "fit_images": int(len(images) - len(held)),
        "calibration_images": int(len(held)),
        "harm_rate_fit": float(fit_y.mean()),
        "seed": seed,
        "columns": columns,
    }
    return FrozenVerifier(model, calibrator, columns, metadata)


def save_verifier(verifier: FrozenVerifier, path: Path) -> str:
    path.write_bytes(pickle.dumps(verifier))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps({**verifier.metadata, "sha256": digest}, indent=2, sort_keys=True), encoding="utf-8")
    return digest


# ---------------------------------------------------------------------------
# gated rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatedPolicy(Policy):
    """Switch iff the switch score is at least tau and V1's harm probability is at most nu."""

    nu: float = 1.0
    verifier_column: str = "verifier_harm"

    def route_frame(self, frame: pd.DataFrame) -> np.ndarray:
        score = frame["score"].to_numpy(dtype=float)
        harm = frame[self.verifier_column].to_numpy(dtype=float)
        return (score >= self.threshold) & (harm <= self.nu)


def gated_family(taus: Sequence[float], nus: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.30)) -> list[GatedPolicy]:
    """Rules ordered by tau ascending, then nu descending (the declared tie order)."""

    return [
        GatedPolicy(name=f"score_ge_{tau:.6g}_veto_gt_{nu:.2f}", threshold=float(tau), nu=float(nu))
        for tau in sorted(taus)
        for nu in sorted(nus, reverse=True)
    ]


# ---------------------------------------------------------------------------
# GPU feature extraction
# ---------------------------------------------------------------------------


class CropSimilarity:
    """Image--text similarity of padded box crops with SigLIP and CLIP."""

    def __init__(self, siglip_path: str, clip_path: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.models = {}
        for name, path in (("siglip", siglip_path), ("clip", clip_path)):
            processor = AutoProcessor.from_pretrained(path)
            model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
            self.models[name] = (processor, model)

    def __call__(self, image, text: str, boxes: Sequence[Sequence[float]]) -> dict[str, list[float]]:
        torch = self.torch
        width, height = image.size
        crops = [image.crop(padded_crop_box(b, width, height)) for b in boxes]
        out = {}
        with torch.no_grad():
            for name, (processor, model) in self.models.items():
                kwargs = {"padding": "max_length"} if name == "siglip" else {"padding": True}
                inputs = processor(text=[text], images=crops, return_tensors="pt", truncation=True, **kwargs).to(self.device)
                image_emb = model.get_image_features(pixel_values=inputs["pixel_values"].half())
                text_emb = model.get_text_features(input_ids=inputs["input_ids"], **({"attention_mask": inputs["attention_mask"]} if "attention_mask" in inputs else {}))
                image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                out[name] = (image_emb.float() @ text_emb.float().T).squeeze(-1).cpu().tolist()
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("features", help="GPU: pair features for generation records (no labels)")
    p.add_argument("--generation", type=Path, nargs="+", required=True)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--siglip", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("export-system-a", help="CPU: System A's archived candidate sets as verifier records (no labels)")
    p.add_argument("--scores-root", type=Path, required=True, help="directory holding cs2_scaleup_<benchmark>/scores/")
    p.add_argument("--records", type=Path, required=True, help="System B G1 record table (sets and image ids)")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("train", help="CPU: train and freeze V1 on training records")
    p.add_argument("--pairs", type=Path, nargs="+", required=True)
    p.add_argument("--generation", type=Path, nargs="+", required=True)
    p.add_argument("--gt-csv", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict", help="CPU: V1 harm probability of each record's chosen candidate")
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--verifier", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "features":
        from .join_system_b_labels import load_generation
        from .system_b import append_record, completed_keys, load_image

        records = load_generation(args.generation)
        done = completed_keys(args.output)
        model = CropSimilarity(args.siglip, args.clip, device=args.device)
        with args.output.open("a", encoding="utf-8") as handle:
            for record in records.to_dict("records"):
                if record["record_key"] in done or as_index(record.get("default_index")) is None:
                    continue
                image = load_image({"image_path": record.get("image_path", record["file_name"])}, args.image_root)
                sims = model(image, record["text"], [c["box"] for c in record["candidates"]])
                for row in pair_rows(record, sims) or [{"record_key": record["record_key"], "empty": True}]:
                    append_record(handle, row)
    elif args.command == "export-system-a":
        from .system_b import append_record

        records, coverage = export_system_a(args.scores_root, args.records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                append_record(handle, record)
        summary = {"records": len(records), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "coverage": coverage}
        args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "train":
        from .join_system_b_labels import load_generation, load_truth

        pairs = pd.concat([pd.read_json(path, lines=True) for path in args.pairs], ignore_index=True)
        pairs = pairs[~pairs.get("empty", False).fillna(False).astype(bool)] if "empty" in pairs else pairs
        records = {r["record_key"]: r for r in load_generation(args.generation).to_dict("records") if r["split"] == "train"}
        pairs = pairs[pairs["record_key"].isin(records)].reset_index(drop=True)
        truth = {}
        for path in args.gt_csv:
            truth.update(load_truth(path))
        labels = harm_labels(pairs, records, truth)
        verifier = train_verifier(pairs, labels)
        digest = save_verifier(verifier, args.output)
        print(json.dumps({**verifier.metadata, "sha256": digest}, indent=2))
    else:
        verifier = pickle.loads(args.verifier.read_bytes())
        pairs = pd.read_json(args.pairs, lines=True)
        chosen = pairs["is_chosen"] if "is_chosen" in pairs else pd.Series(False, index=pairs.index)
        pairs = pairs[chosen.fillna(False).astype(bool)].reset_index(drop=True)
        pairs["verifier_harm"] = verifier.predict(pairs) if len(pairs) else np.empty(0)
        pairs[["record_key", "verifier_harm"]].to_csv(args.output, index=False)
        print(f"wrote {args.output} ({len(pairs)} chosen candidates)")


if __name__ == "__main__":
    main()
