"""Fit the real train-only improvement discriminator and audit it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit import Policy, run_protocol
from .risk_controls import add_improvement_score, fit_improvement_discriminator


DEFAULT_FEATURES = (
    "candidate_confidence",
    "candidate_box_area",
    "candidate_aspect_ratio",
    "query_token_count",
    "candidate_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(frame: pd.DataFrame, feature_columns: tuple[str, ...], seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    model = fit_improvement_discriminator(frame, feature_columns, seed=seed)
    scored = add_improvement_score(frame, model)
    # Keep the raw candidate confidence for diagnostics, but route on the
    # learned improvement score in this arm.
    scored["raw_candidate_score"] = scored["score"]
    scored["score"] = scored["improvement_score"]
    policies = [Policy(name=f"improvement_score_ge_{threshold:.2f}", threshold=threshold)
                for threshold in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)]
    result = run_protocol(scored, policies, alpha=0.05, delta=0.05)
    result["discriminator"] = model.to_dict()
    result["feature_columns"] = list(feature_columns)
    result["score_semantics"] = "sigmoid(logistic improvement-vs-harm score fitted on train labels only)"
    return scored, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--seed", type=int, default=20270822)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_columns = tuple(value.strip() for value in args.features.split(",") if value.strip())
    frame = pd.read_csv(args.input_csv)
    scored, result = run(frame, feature_columns, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "predictions_with_improvement_score.csv"
    summary_path = args.output_dir / "audit_summary.json"
    model_path = args.output_dir / "improvement_discriminator.json"
    scored.to_csv(scored_path, index=False)
    result["input_sha256"] = _sha256(args.input_csv)
    result["scored_manifest_sha256"] = _sha256(scored_path)
    result["model_sha256"] = hashlib.sha256(
        json.dumps(result["discriminator"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    model_path.write_text(json.dumps(result["discriminator"], indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
