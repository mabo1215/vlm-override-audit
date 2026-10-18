"""Secondary analyses of the System B audits (Design.md section 15.4).

Runs after an arm's audits are assembled and never changes a frozen rule:

* Learn-then-Test comparison: every rule of the declared family is tested on
  the pooled selection and certification sets with a Bonferroni-corrected
  Clopper--Pearson bound; the passing rule with the largest net gain is
  reported on the confirmation set.
* V1 diagnostics at the selected gated rule: recall of harmful switches and
  false-veto rate on the selection set with exact 95% intervals, and the
  help-versus-harm AUROC of V1's harm probability and of the switch score.
* V1 applied to System A: System A's own threshold family gated by the
  frozen V1, run through the same select -> certify -> confirm protocol.
  System A's labels were read before V1 existed, so this is reported as a
  post hoc check, not as a result fixed in advance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

from .audit import IOU_THRESHOLD, Policy, evaluate_policy, run_protocol
from .records import primary_manifests
from .verifier import SYSTEM_A_PREFIX, attach_verifier, gated_family

SYSTEM_A_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)  # System A's declared family


def exact_interval(k: int, n: int, level: float = 0.95) -> list[float] | None:
    """Two-sided Clopper--Pearson interval for k successes in n trials."""

    if n == 0:
        return None
    tail = (1 - level) / 2
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return [lower, upper]


def ltt_bonferroni(frame: pd.DataFrame, policies: Sequence[Policy], *, alpha: float = 0.05, delta: float = 0.05) -> dict:
    """Learn then Test over the family on the pooled selection and certification sets."""

    pooled = frame[frame["split"].isin(["select", "certify"])].assign(split="pooled")
    tested = [
        evaluate_policy(pooled, p, split="pooled", alpha=alpha, delta=delta, certification_family_size=len(policies))
        for p in policies
    ]
    valid = [r for r in tested if r.route_count > 0 and r.cp_upper is not None and r.cp_upper <= alpha]
    out: dict[str, object] = {
        "family_size": len(policies),
        "pooled_records": int(len(pooled)),
        "passing": [r.policy for r in valid],
    }
    if not valid:
        out["deployed"] = False
        return out
    best = max(valid, key=lambda r: (r.net_accuracy_gain, r.route_count))
    policy = next(p for p in policies if p.name == best.policy)
    confirm = frame[frame["split"] == "confirm"]
    out.update(deployed=True, policy=best.policy, confirm=evaluate_policy(confirm, policy, split="confirm", alpha=alpha, delta=delta).to_dict())
    return out


def _auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Probability that a positive scores above a negative (ties count half)."""

    if len(positive) == 0 or len(negative) == 0:
        return None
    diff = positive[:, None] - negative[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def verifier_diagnostics(frame: pd.DataFrame, tau: float, nu: float, split: str) -> dict:
    """Veto recall and false-veto rate among switches with s >= tau; help-vs-harm AUROC of v and s."""

    part = frame[frame["split"] == split]
    switched = part[part["score"] >= tau]
    base_ok = switched["baseline_iou"] >= IOU_THRESHOLD
    alt_ok = switched["challenger_iou"] >= IOU_THRESHOLD
    harmful = (base_ok & ~alt_ok).to_numpy()
    vetoed = (switched["verifier_harm"] > nu).to_numpy()
    k_harm, n_harm = int((vetoed & harmful).sum()), int(harmful.sum())
    k_false, n_false = int((vetoed & ~harmful).sum()), int((~harmful).sum())
    decisive = part[part["score"] > 0]
    help_mask = ((decisive["baseline_iou"] < IOU_THRESHOLD) & (decisive["challenger_iou"] >= IOU_THRESHOLD)).to_numpy()
    harm_mask = ((decisive["baseline_iou"] >= IOU_THRESHOLD) & (decisive["challenger_iou"] < IOU_THRESHOLD)).to_numpy()
    v = decisive["verifier_harm"].to_numpy(dtype=float)
    s = decisive["score"].to_numpy(dtype=float)
    return {
        "split": split,
        "tau": tau,
        "nu": nu,
        "switches_at_tau": int(len(switched)),
        "harmful_vetoed": [k_harm, n_harm],
        "harm_recall": k_harm / n_harm if n_harm else None,
        "harm_recall_ci95": exact_interval(k_harm, n_harm),
        "non_harmful_vetoed": [k_false, n_false],
        "false_veto_rate": k_false / n_false if n_false else None,
        "false_veto_ci95": exact_interval(k_false, n_false),
        "decisive_switches": {"help": int(help_mask.sum()), "harm": int(harm_mask.sum())},
        # harm probability should be higher on harmful switches; the score higher on helpful ones
        "auroc_v_harm_over_help": _auroc(v[harm_mask], v[help_mask]),
        "auroc_s_help_over_harm": _auroc(s[help_mask], s[harm_mask]),
    }


def g1_analysis(protocol: dict, joined: Path, arm: str = "G1", alpha: float = 0.05, delta: float = 0.05) -> dict:
    """LTT comparison and V1 diagnostics from an arm's assembled manifests (the gated one carries V1)."""

    out: dict[str, object] = {}
    for benchmark in sorted(protocol["frozen_rules"][arm]):
        manifest = pd.read_csv(joined / f"{arm}_{benchmark}_manifest.csv", dtype={"expression_id": str})
        listwise = pd.read_csv(joined / f"{arm}-listwise_{benchmark}_manifest.csv", dtype={"expression_id": str})
        gated = pd.read_csv(joined / f"{arm}-gated_{benchmark}_manifest.csv", dtype={"expression_id": str})
        family = protocol["threshold_families"][arm][benchmark]
        spec = protocol["gated_families"][arm + "-gated"][benchmark]
        rule = protocol["frozen_rules"][arm + "-gated"][benchmark]
        plain = [Policy(name=f"score_ge_{t:.6g}", threshold=float(t)) for t in family]
        lw = [Policy(name=f"score_ge_{t:.6g}", threshold=float(t)) for t in protocol["threshold_families"][arm + "-listwise"][benchmark]]
        out[benchmark] = {
            "ltt_pairwise": ltt_bonferroni(manifest, plain, alpha=alpha, delta=delta),
            "ltt_listwise": ltt_bonferroni(listwise, lw, alpha=alpha, delta=delta),
            "ltt_gated": ltt_bonferroni(gated, gated_family(spec["taus"], spec["nus"]), alpha=alpha, delta=delta),
            **{f"verifier_{split}": verifier_diagnostics(gated, rule["threshold"], rule["nu"], split) for split in ("select", "certify", "confirm")},
        }
    return out


def system_a_analysis(predictions: Path, alpha: float = 0.05, delta: float = 0.05) -> dict:
    preds = pd.read_csv(predictions)
    out: dict[str, object] = {}
    for benchmark, (path, threshold) in primary_manifests("System A").items():
        manifest = pd.read_csv(path, dtype={"expression_id": str})
        gated = attach_verifier(manifest, preds, benchmark, key_prefix=SYSTEM_A_PREFIX)
        family = gated_family(SYSTEM_A_THRESHOLDS)
        result = run_protocol(gated, family, alpha=alpha, delta=delta)
        selected = next(p for p in family if p.name == result["certify"]["policy"])
        out[benchmark] = {
            "ungated_frozen_threshold": threshold,
            "gated": {k: result[k] for k in ("certify", "confirm")},
            "gated_policy": selected.name,
            "ltt_gated": ltt_bonferroni(gated, family, alpha=alpha, delta=delta),
            **{f"verifier_{split}": verifier_diagnostics(gated, selected.threshold, selected.nu, split) for split in ("select", "certify", "confirm")},
        }
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", default="G1")
    parser.add_argument("--joined", type=Path, required=True, help="directory with the assembled manifests")
    parser.add_argument("--system-a-predictions", type=Path, help="V1 predictions for System A (optional)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = {"arm": args.arm, "alpha": protocol["alpha"], "delta": protocol["delta"],
              "system_b": g1_analysis(protocol, args.joined, args.arm, protocol["alpha"], protocol["delta"])}
    if args.system_a_predictions:
        result["system_a_with_v1"] = system_a_analysis(args.system_a_predictions, protocol["alpha"], protocol["delta"])
        result["system_a_note"] = "post hoc: System A's labels were read before V1 was trained"
    args.output.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
