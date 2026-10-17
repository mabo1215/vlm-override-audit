"""Every number in the paper's main result tables, computed from the registry.

For each system and benchmark in ``configs/audit_records.json`` and for the
certification and confirmation sets, this command reports:

* the selected (frozen) rule: switches m, helps h, harms k, harm among
  switches k/m, exact one-sided 95% upper and lower bounds, accuracy change;
* always switch (every record with s > 0), with a descriptive bound;
* proposal-matched random: m images drawn uniformly among records with s > 0
  (expected values, exact; review item J10);
* the hindsight oracle (switches exactly the helpful records);
* the help-versus-harm AUROC of the switch score;
* an image-level bootstrap of the selected rule minus proposal-matched random
  (accuracy change and harm among switches).

Output: one JSON file; the manuscript tables are written from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from .audit import clopper_pearson_upper
from .records import ROOT, load_registry
from .sensitivity import auroc_help_harm

IOU = 0.5
DELTA = 0.05


def cp_lower(k: int, m: int, delta: float = DELTA) -> float | None:
    if m == 0:
        return None
    return 0.0 if k == 0 else float(beta.ppf(delta, k, m - k + 1))


def outcome(frame: pd.DataFrame, switched: np.ndarray) -> dict[str, object]:
    base = frame["baseline_iou"].to_numpy() >= IOU
    alt = frame["challenger_iou"].to_numpy() >= IOU
    m = int(switched.sum())
    h = int((switched & ~base & alt).sum())
    k = int((switched & base & ~alt).sum())
    n = int(len(frame))
    return {
        "n": n,
        "m": m,
        "h": h,
        "k": k,
        "harm_among_switches": k / m if m else None,
        "upper": clopper_pearson_upper(k, m, DELTA) if m else None,
        "lower": cp_lower(k, m),
        "accuracy_change_pp": 100.0 * (h - k) / n,
    }


def proposal_matched_random(frame: pd.DataFrame, m: int) -> dict[str, object]:
    proposals = frame["score"].to_numpy() > 0
    base = frame["baseline_iou"].to_numpy() >= IOU
    alt = frame["challenger_iou"].to_numpy() >= IOU
    big_n = int(proposals.sum())
    helps = int((proposals & ~base & alt).sum())
    harms = int((proposals & base & ~alt).sum())
    n = int(len(frame))
    draws = min(m, big_n)
    return {
        "m": draws,
        "pool": big_n,
        "expected_h": draws * helps / big_n if big_n else 0.0,
        "expected_k": draws * harms / big_n if big_n else 0.0,
        "harm_among_switches": harms / big_n if big_n else None,
        "accuracy_change_pp": 100.0 * draws * (helps - harms) / big_n / n if big_n else 0.0,
    }


def bootstrap_vs_random(frame: pd.DataFrame, tau: float, repeats: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    rows = frame.reset_index(drop=True)
    gain_diff, harm_diff = [], []
    for _ in range(repeats):
        sample = rows.iloc[rng.integers(0, len(rows), len(rows))]
        selected = outcome(sample, sample["score"].to_numpy() >= tau)
        random = proposal_matched_random(sample, selected["m"])
        gain_diff.append(selected["accuracy_change_pp"] - random["accuracy_change_pp"])
        if selected["m"] and random["harm_among_switches"] is not None:
            harm_diff.append(selected["harm_among_switches"] - random["harm_among_switches"])
    q = lambda values: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    return {"repeats": repeats, "gain_diff_pp_95": q(gain_diff), "harm_diff_95": q(harm_diff) if harm_diff else None}


def benchmark_numbers(entry: dict, repeats: int, seed: int) -> dict[str, object]:
    frame = pd.read_csv(ROOT / entry["primary"])
    tau = float(entry["threshold"])
    out: dict[str, object] = {"threshold": tau}
    for split in ("certify", "confirm", "select"):
        part = frame[frame["split"] == split].reset_index(drop=True)
        selected = outcome(part, part["score"].to_numpy() >= tau)
        always = outcome(part, part["score"].to_numpy() > 0)
        oracle_switch = (part["baseline_iou"].to_numpy() < IOU) & (part["challenger_iou"].to_numpy() >= IOU)
        out[split] = {
            "selected": selected,
            "always_switch": always,
            "proposal_matched_random": proposal_matched_random(part, selected["m"]),
            "oracle": outcome(part, oracle_switch),
            "auroc": auroc_help_harm(part),
        }
        if split in ("certify", "confirm"):
            out[split]["bootstrap_vs_random"] = bootstrap_vs_random(part, tau, repeats, seed)
    reductions = {}
    for name, path in entry.get("reductions", {}).items():
        other = pd.read_csv(ROOT / path)
        reductions[name] = auroc_help_harm(other[other["split"] == "certify"])["auroc"]
    reductions["primary"] = out["certify"]["auroc"]["auroc"]
    values = [v for v in reductions.values() if v is not None]
    out["certify_auroc_reductions"] = {"values": reductions, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", action="append", help="system name(s) in the registry; default all")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/cs2_audit/results/paper_numbers_20261016.json")
    args = parser.parse_args()

    registry = load_registry()["systems"]
    systems = args.system or list(registry)
    result = {
        system: {name: benchmark_numbers(entry, args.bootstrap, args.seed) for name, entry in registry[system].items()}
        for system in systems
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
