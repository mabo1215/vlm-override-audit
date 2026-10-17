"""CLI for the revised CS^2 audit protocol and GPU smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audit import Policy, run_protocol
from .baselines import (
    matched_random_reference,
    no_switch_reference,
    oracle_reference,
    raw_score_reference,
)
from .report import build_latex_table
from .cost_report import read_jsonl, sha256_file, summarize_events
from .select_qualitative import select_examples
from .risk_controls import rcps_binary_harm_reference
from .sensitivity import auroc_help_harm, image_block_permutation_placebo, image_cluster_bootstrap


REQUIRED_COLUMNS = [
    "split",
    "image_id",
    "score",
    "baseline_iou",
    "challenger_iou",
]


def gpu_smoke_test(device_name: str) -> dict[str, object]:
    """Exercise the selected CUDA device without loading a model checkpoint."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    generator = torch.Generator(device=device).manual_seed(20270822)
    left = torch.randn((256, 256), generator=generator, device=device)
    right = torch.randn((256, 256), generator=generator, device=device)
    product = left @ right
    checksum = float(product.mean().detach().cpu())
    return {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "matmul_shape": [256, 256],
        "checksum": checksum,
    }


def make_smoke_manifest(seed: int = 20270822) -> pd.DataFrame:
    """Create deterministic, image-disjoint synthetic records for CI/smoke."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    # The train split is present to make the data-flow explicit.  The smoke
    # test does not train a scorer; real vGPU runs will replace this with model
    # outputs produced from the train split.
    split_sizes = {"train": 48, "select": 96, "certify": 128, "confirm": 128}
    image_index = 0
    for split, size in split_sizes.items():
        latent = rng.normal(size=size)
        score = 0.5 + 0.22 * latent + rng.normal(scale=0.12, size=size)
        score = np.clip(score, 0.0, 1.0)
        baseline_iou = np.clip(0.55 + 0.20 * latent + rng.normal(scale=0.22, size=size), 0, 1)
        challenger_iou = np.clip(0.56 + 0.25 * latent + rng.normal(scale=0.20, size=size), 0, 1)
        for row in range(size):
            rows.append(
                {
                    "split": split,
                    "image_id": f"smoke-image-{image_index:05d}",
                    "expression_id": f"smoke-expression-{image_index:05d}",
                    "score": float(score[row]),
                    "baseline_iou": float(baseline_iou[row]),
                    "challenger_iou": float(challenger_iou[row]),
                }
            )
            image_index += 1
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke-test", action="store_true")
    source.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cs2_audit/outputs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--thresholds", default="0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help=(
            "Also compute no-switch/raw-score/matched-random/oracle baselines "
            "and AUROC/placebo diagnostics on the certify split, and emit "
            "methods_summary.json, diagnostics.json, and main_table.tex."
        ),
    )
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--placebo-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--baseline-seed", type=int, default=0)
    parser.add_argument(
        "--cost-events",
        type=Path,
        help="Optional JSONL inference event log to summarize into cost_summary.json.",
    )
    parser.add_argument("--cost-output", type=Path)
    parser.add_argument(
        "--qualitative-output",
        type=Path,
        help="Optional JSON output for one frozen help/harm/no-route confirm example.",
    )
    return parser.parse_args()


def run_baselines_and_diagnostics(
    frame: pd.DataFrame,
    result: dict[str, object],
    *,
    random_repeats: int,
    placebo_repeats: int,
    bootstrap_repeats: int,
    baseline_seed: int,
    alpha: float,
    delta: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute P0-6 baselines and P0-4/P0-8 diagnostics on the certify split.

    Returns ``(methods_summary, diagnostics)``; both are JSON-serializable and
    every number traces back to this one manifest and its frozen policy.
    """

    certify_frame = frame[frame["split"] == "certify"].reset_index(drop=True)
    selected_policy = Policy(**result["selected_policy"])
    certify_result = result["certify"]

    methods_summary: dict[str, object] = {
        "No-switch": no_switch_reference(certify_frame, "certify").to_dict(),
        "Raw-score (uncertified)": raw_score_reference(
            certify_frame,
            "certify",
            selected_policy.threshold,
            alpha=alpha,
            delta=delta,
        ).to_dict(),
        f"CS$^2$ (threshold {selected_policy.threshold:.2f})": certify_result,
        "Fixed-candidate hindsight oracle": oracle_reference(
            certify_frame, "certify"
        ).to_dict(),
        "RCPS-binary-harm (equivalent bound)": rcps_binary_harm_reference(
            certify_frame,
            selected_policy,
            split="certify",
            alpha=alpha,
            delta=delta,
        ),
    }
    random_summary = matched_random_reference(
        certify_frame,
        "certify",
        route_count=certify_result["route_count"],
        n_repeats=random_repeats,
        seed=baseline_seed,
    )
    methods_summary["Matched-random"] = random_summary.to_audit_row()
    if "relation_verifier_score" in certify_frame.columns:
        from .baselines import relation_verifier_reference

        methods_summary["Relation-verifier (uncertified)"] = relation_verifier_reference(
            certify_frame,
            "certify",
            "relation_verifier_score",
            selected_policy.threshold,
            alpha=alpha,
            delta=delta,
        ).to_dict()

    def net_gain_statistic(sub_frame: pd.DataFrame) -> float:
        selected = Policy(**result["selected_policy"])
        routed = selected.route(sub_frame["score"].to_numpy())
        baseline_correct = sub_frame["baseline_iou"].to_numpy(dtype=float) >= 0.5
        challenger_correct = sub_frame["challenger_iou"].to_numpy(dtype=float) >= 0.5
        help_count = int((routed & ~baseline_correct & challenger_correct).sum())
        harm_count = int((routed & baseline_correct & ~challenger_correct).sum())
        return (help_count - harm_count) / len(sub_frame)

    diagnostics: dict[str, object] = {
        "auroc_help_harm": auroc_help_harm(certify_frame),
        "net_accuracy_gain_bootstrap": image_cluster_bootstrap(
            certify_frame,
            net_gain_statistic,
            n_bootstrap=bootstrap_repeats,
            seed=baseline_seed,
        ).to_dict(),
        "placebo": image_block_permutation_placebo(
            certify_frame,
            "certify",
            selected_policy,
            n_repeats=placebo_repeats,
            seed=baseline_seed,
        ),
    }
    return methods_summary, diagnostics


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    if not thresholds:
        raise SystemExit("--thresholds must contain at least one value")
    policies = [Policy(name=f"score_ge_{threshold:.2f}", threshold=threshold) for threshold in thresholds]

    if args.smoke_test:
        frame = make_smoke_manifest()
        device_report = gpu_smoke_test(args.device)
    else:
        frame = pd.read_csv(args.input_csv)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise SystemExit(f"input CSV is missing columns: {missing}")
        device_report = {"device": "not-used", "note": "input manifest mode"}

    result = run_protocol(frame, policies, alpha=args.alpha, delta=args.delta)
    result["device"] = device_report
    result["protocol_version"] = "cs2-image-level-v2"
    result["sampling_unit"] = "one preselected expression per image"
    result["policy_freeze"] = {
        "selection_split": "select",
        "certification_split": "certify",
        "confirmation_split": "confirm",
        "frozen_before_certify_labels": True,
    }
    result["threshold_family"] = thresholds
    result["alpha"] = args.alpha
    result["delta"] = args.delta
    result["input_manifest_sha256"] = (
        sha256_file(args.input_csv) if args.input_csv is not None else None
    )
    result["rows"] = {split: int((frame["split"] == split).sum()) for split in sorted(frame["split"].unique())}
    result["unique_images"] = {
        split: int(frame.loc[frame["split"] == split, "image_id"].nunique())
        for split in sorted(frame["split"].unique())
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / ("smoke_summary.json" if args.smoke_test else "audit_summary.json")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")

    if args.qualitative_output:
        qualitative = select_examples(frame, Policy(**result["selected_policy"]))
        args.qualitative_output.parent.mkdir(parents=True, exist_ok=True)
        args.qualitative_output.write_text(
            json.dumps(qualitative, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {args.qualitative_output}")

    if args.cost_events:
        cost_output = args.cost_output or args.output_dir / "cost_summary.json"
        summary = summarize_events(
            read_jsonl(args.cost_events),
            input_sha256=(sha256_file(args.input_csv) if args.input_csv else None),
            output_sha256=sha256_file(output_path),
        )
        cost_output.parent.mkdir(parents=True, exist_ok=True)
        cost_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {cost_output}")

    if args.with_baselines:
        methods_summary, diagnostics = run_baselines_and_diagnostics(
            frame,
            result,
            random_repeats=args.random_repeats,
            placebo_repeats=args.placebo_repeats,
            bootstrap_repeats=args.bootstrap_repeats,
            baseline_seed=args.baseline_seed,
            alpha=args.alpha,
            delta=args.delta,
        )
        methods_path = args.output_dir / "methods_summary.json"
        methods_path.write_text(json.dumps(methods_summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {methods_path}")

        diagnostics_path = args.output_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote {diagnostics_path}")

        table = build_latex_table(
            list(methods_summary.items()),
            caption=(
                "Selective-switching audit results on the certify split "
                "(smoke manifest; not a paper result)."
                if args.smoke_test
                else "Selective-switching audit results (exact counts from the archived manifest)."
            ),
            label="tab:cs2_main",
        )
        table_path = args.output_dir / "main_table.tex"
        table_path.write_text(table, encoding="utf-8")
        print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
