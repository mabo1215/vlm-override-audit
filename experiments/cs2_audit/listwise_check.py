"""Post hoc check: does System A's listwise score change the audit outcome?

System A's generator scored every record in two ways before routing: a
listwise posterior over all candidates (two permutations, two forward passes)
and the pairwise score (every ordered candidate pair, n(n-1) passes) that the
pipeline routes on and that the audit certifies.  An exploratory run on an
earlier, defective RefCOCO+ candidate pool had once favoured the listwise
score, so a reader may ask whether the negative audit result is specific to
the pairwise score.

This command rebuilds the audited records with the listwise margin as the
switch score, on exactly the same images, expressions, splits and default
boxes, and repeats the audit.  It first rebuilds the pairwise records from the
same logs and fails unless they reproduce the registered records, so both
scores are compared on verified inputs.  The listwise analysis is post hoc:
labels of these records have been read before, so its results are diagnostic
and are never presented as a certificate.

Example (archived logs and ground truth):
    python -m experiments.cs2_audit.listwise_check \
        --scores-root <dir holding cs2_scaleup_<benchmark>/scores> \
        --gt "RefCOCO+=<calibration parquet>" --gt "RefCOCO+=<confirmation parquet>" ... \
        --output experiments/cs2_audit/results/listwise_check_20261016.json

Example (public release; the candidate files carry every candidate's IoU):
    python -m experiments.cs2_audit.listwise_check --release-records data/system_a --output <file>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import Policy, clopper_pearson_upper, run_protocol
from .build_manifest import box_iou, load_ground_truth
from .paper_numbers import outcome
from .records import ROOT, load_registry
from .sensitivity import auroc_help_harm

TIE_TOLERANCE = 1e-6
TAGS = {"RefCOCO+": "refcocoplus", "RefCOCOg": "refcocog", "RefCOCO": "refcoco"}
DECLARED = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
DENSE = tuple(round(0.05 * i, 2) for i in range(1, 20))
FIELDS = {"pairwise": "pairwise_score", "listwise": "listwise_posterior"}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_paths(scores_dir: Path) -> list[str]:
    return sorted(glob.glob(str(scores_dir / "*_scores.json")))


def load_scores(scores_dir: Path) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for path in score_paths(scores_dir):
        for entry in json.loads(Path(path).read_text()).get("scores", []):
            entries[str(entry["example_id"])] = entry
    if not entries:
        raise FileNotFoundError(f"no score logs under {scores_dir}")
    return entries


def switch_choice(values: list[float], incumbent: int) -> tuple[int, float]:
    """Highest-valued candidate and its margin over the default; ties keep the default."""

    best = int(np.argmax(values))
    margin = values[best] - values[incumbent]
    if margin < TIE_TOLERANCE:
        return incumbent, 0.0
    return best, float(margin)


def load_release(path: Path) -> dict[str, dict]:
    """Released candidate records in the score-log layout, with each candidate's IoU."""

    import gzip

    entries: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            entries[str(row["ref_id"])] = {
                "candidate_sources": row["candidate_sources"],
                "candidate_boxes": row["candidate_boxes"],
                "candidate_scores": [
                    {"pairwise_score": p, "listwise_posterior": q}
                    for p, q in zip(row["candidate_pairwise_scores"], row["candidate_listwise_posteriors"])
                ],
                "candidate_ious": row["candidate_ious"],
            }
    return entries


def rebuild(records: pd.DataFrame, scores: dict[str, dict], truth: dict | None) -> tuple[dict[str, pd.DataFrame], dict]:
    """Records for both scores, plus the check against the registered records."""

    rows = {name: [] for name in FIELDS}
    agree = proposals_both = passes_pairwise = passes_listwise = 0
    mismatch = {"score": 0, "baseline_iou": 0, "challenger_iou": 0}
    for record in records.itertuples(index=False):
        entry = scores[str(record.expression_id)]
        sources = list(entry["candidate_sources"])
        incumbent = sources.index("restricted")
        if "candidate_ious" in entry:
            ious = [float(v) for v in entry["candidate_ious"]]
        else:
            gt = truth[str(record.expression_id)]
            ious = [box_iou(tuple(box), gt) for box in entry["candidate_boxes"]]
        n = len(sources)
        passes_pairwise += n * (n - 1)
        passes_listwise += 2 if n > 1 else 0
        chosen = {}
        for name, field in FIELDS.items():
            values = [float(item[field]) for item in entry["candidate_scores"]]
            best, margin = switch_choice(values, incumbent)
            chosen[name] = best if margin > 0 else None
            rows[name].append(
                {
                    "split": record.split,
                    "image_id": int(record.image_id),
                    "expression_id": str(record.expression_id),
                    "score": margin,
                    "baseline_iou": ious[incumbent],
                    "challenger_iou": ious[best],
                }
            )
        pair = rows["pairwise"][-1]
        mismatch["score"] += abs(pair["score"] - float(record.score)) > 1e-9
        mismatch["baseline_iou"] += abs(pair["baseline_iou"] - float(record.baseline_iou)) > 1e-9
        mismatch["challenger_iou"] += abs(pair["challenger_iou"] - float(record.challenger_iou)) > 1e-9
        if chosen["pairwise"] is not None and chosen["listwise"] is not None:
            proposals_both += 1
            agree += chosen["pairwise"] == chosen["listwise"]
    frames = {name: pd.DataFrame(value) for name, value in rows.items()}
    check = {
        "records": int(len(records)),
        "pairwise_mismatches_vs_registered": {k: int(v) for k, v in mismatch.items()},
        "both_propose_a_switch": proposals_both,
        "same_alternative_when_both_propose": agree,
        "mean_pairwise_passes": passes_pairwise / len(records),
        "mean_listwise_passes": passes_listwise / len(records),
    }
    return frames, check


def hindsight(part: pd.DataFrame, alpha: float = 0.05) -> dict[str, object]:
    """Best threshold chosen on the evaluated split itself (never a valid rule)."""

    scores = part["score"].to_numpy(dtype=float)
    base = part["baseline_iou"].to_numpy() >= 0.5
    alt = part["challenger_iou"].to_numpy() >= 0.5
    best = None
    certifiable = 0
    for tau in np.unique(scores[scores > 0]):
        routed = scores >= tau
        m = int(routed.sum())
        k = int((routed & base & ~alt).sum())
        upper = clopper_pearson_upper(k, m, 0.05)
        certifiable += upper <= alpha
        if best is None or upper < best["upper"]:
            best = {"threshold": float(tau), "m": m, "k": k, "upper": float(upper)}
    return {"thresholds_with_upper_at_most_alpha": int(certifiable), "lowest_upper": best}


def top_m(part: pd.DataFrame, m: int) -> dict[str, object]:
    """Harm among the m highest-scoring proposals (stable order on ties)."""

    ranked = part[part["score"] > 0].sort_values("score", ascending=False, kind="mergesort").head(m)
    routed = part.index.isin(ranked.index)
    return outcome(part, routed)


def protocol_summary(frame: pd.DataFrame, family: tuple[float, ...]) -> dict[str, object]:
    policies = [Policy(name=f"score_ge_{t:.2f}", threshold=t) for t in family]
    result = run_protocol(frame, policies)
    tau = float(result["selected_policy"]["threshold"])
    out = {"threshold": tau, "selection_reason": result["protocol"]["selection_reason"]}
    for split in ("certify", "confirm"):
        part = frame[frame["split"] == split].reset_index(drop=True)
        out[split] = outcome(part, part["score"].to_numpy() >= tau)
        out[split]["status"] = result[split]["status"]
    return out


def benchmark(records: pd.DataFrame, scores: dict[str, dict], truth: dict | None, matched_m: dict[str, int]) -> dict:
    frames, check = rebuild(records, scores, truth)
    out: dict[str, object] = {"check": check}
    for name, frame in frames.items():
        result: dict[str, object] = {
            "declared_family": protocol_summary(frame, DECLARED),
            "dense_family": protocol_summary(frame, DENSE),
        }
        for split in ("select", "certify", "confirm"):
            part = frame[frame["split"] == split].reset_index(drop=True)
            result[split] = {
                "always_switch": outcome(part, part["score"].to_numpy() > 0),
                "auroc": auroc_help_harm(part)["auroc"],
                "hindsight": hindsight(part),
                "top_m_matched_to_selected_pairwise": top_m(part, matched_m[split]),
            }
        out[name] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-root", type=Path)
    parser.add_argument("--gt", action="append", default=[], help="BENCHMARK=parquet (repeatable)")
    parser.add_argument("--release-records", type=Path, help="directory of released <benchmark>_candidates.jsonl.gz files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.release_records is None) == (args.scores_root is None):
        parser.error("give either --release-records or --scores-root with --gt")

    gt_paths: dict[str, list[str]] = {}
    for item in args.gt:
        name, path = item.split("=", 1)
        gt_paths.setdefault(name, []).append(path)

    registry = load_registry()["systems"]["System A"]
    report: dict[str, object] = {
        "note": "Post hoc diagnostic on records whose labels were read before; not a certificate.",
        "declared_family": list(DECLARED),
        "dense_family": list(DENSE),
        "alpha": 0.05,
        "delta": 0.05,
        "tie_tolerance": TIE_TOLERANCE,
        "inputs": {},
    }
    for name, entry in registry.items():
        truth: dict | None = None
        if args.release_records is not None:
            release = args.release_records / f"{name.replace('+', 'plus')}_candidates.jsonl.gz"
            scores = load_release(release)
            report["inputs"][name] = {"release_records": {release.name: sha256(release)}}
        else:
            scores_dir = args.scores_root / f"cs2_scaleup_{TAGS[name]}" / "scores"
            scores = load_scores(scores_dir)
            report["inputs"][name] = {
                "score_logs": {Path(p).name: sha256(p) for p in score_paths(scores_dir)},
                "ground_truth": {Path(p).name: sha256(p) for p in gt_paths[name]},
            }
            truth = {}
            for path in gt_paths[name]:
                for key, box in load_ground_truth(path).items():
                    truth.setdefault(key, box)
        seeds = {"20270822": entry["primary"], **entry.get("reductions", {})}
        per_seed = {}
        for seed, rel in seeds.items():
            records = pd.read_csv(ROOT / rel, dtype={"expression_id": str})
            tau = float(entry["threshold"])
            matched = {s: int((records.loc[records["split"] == s, "score"] >= tau).sum()) for s in ("select", "certify", "confirm")}
            per_seed[seed] = benchmark(records, scores, truth, matched)
            bad = per_seed[seed]["check"]["pairwise_mismatches_vs_registered"]
            if any(bad.values()):
                raise SystemExit(f"{name} seed {seed}: rebuilt pairwise records differ from the registry: {bad}")
        report[name] = per_seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
