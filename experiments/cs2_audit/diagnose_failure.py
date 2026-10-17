"""Explain why a frozen switching policy fails certification.

All analyses run on CPU from existing one-expression-per-image manifests
(columns ``split, image_id, score, baseline_iou, challenger_iou``).  No model
is rerun and no certification label is used to choose anything that is later
certified.

1. Accuracy view versus audit view: what an accuracy-only evaluation reports
   for the same switch, next to the routed help/harm counts.
2. Risk-coverage curves on the selection split only.
3. Exact certification power as a function of the number of routed examples
   and the true harm rate among routed examples.
4. A controlled verifier study.  The real images, boxes, help/harm outcomes,
   routing scores and split sizes are kept; a simulated verifier vetoes
   switches before routing: it catches a set
   fraction of harmful switches and wrongly vetoes a set fraction of the
   others.  Because the veto rates are known, the true conditional harm of
   every threshold is known, so each decision procedure can be checked
   against the truth.
5. Alternative decision procedures on the real scores at the same risk target
   and data budget (Learn then Test with a Bonferroni family correction, and
   two shortcuts that skip parts of the protocol).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

from .records import check_thresholds, primary_manifests

IOU_THRESHOLD = 0.5
ALPHA = 0.05
DELTA = 0.05
REAL_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
CONTROLLED_RECALLS = (0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)
CONTROLLED_FALSE_VETOES = (0.0, 0.1, 0.3)
METHODS = ("cs2", "ltt_bonferroni", "same_split", "empirical_screen")


def cp_upper(k: np.ndarray | int, m: np.ndarray | int, delta: float) -> np.ndarray:
    """Vectorized one-sided Clopper--Pearson upper bound; NaN when ``m == 0``."""

    k_arr = np.asarray(k, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    out = np.full(np.broadcast(k_arr, m_arr).shape, np.nan)
    k_b, m_b = np.broadcast_arrays(k_arr, m_arr)
    full = (m_b > 0) & (k_b >= m_b)
    inner = (m_b > 0) & (k_b < m_b)
    out[full] = 1.0
    out[inner] = beta.ppf(1.0 - delta, k_b[inner] + 1.0, m_b[inner] - k_b[inner])
    return out


def outcome_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (help, harm, proposed) masks for each row."""

    baseline = frame["baseline_iou"].to_numpy(dtype=float) >= IOU_THRESHOLD
    challenger = frame["challenger_iou"].to_numpy(dtype=float) >= IOU_THRESHOLD
    proposed = frame["score"].to_numpy(dtype=float) > 0.0
    return (~baseline & challenger), (baseline & ~challenger), proposed


def _counts(routed: np.ndarray, help_: np.ndarray, harm: np.ndarray) -> dict[str, float]:
    n = int(routed.size)
    m = int(routed.sum())
    h = int((routed & help_).sum())
    k = int((routed & harm).sum())
    upper = float(cp_upper(k, m, DELTA)) if m else None
    return {
        "n": n,
        "routed": m,
        "help": h,
        "harm": k,
        "route_rate": m / n,
        "harm_among_routed": (k / m) if m else None,
        "cp_upper": upper,
        "net_gain_pp": 100.0 * (h - k) / n,
    }


# ---------------------------------------------------------------------------
# 1. accuracy view versus audit view
# ---------------------------------------------------------------------------


def accuracy_versus_audit(frame: pd.DataFrame, threshold: float) -> dict[str, dict]:
    """Compare 'always take the challenger' with the frozen threshold policy."""

    out: dict[str, dict] = {}
    for split in ("certify", "confirm"):
        part = frame[frame["split"] == split]
        help_, harm, proposed = outcome_masks(part)
        baseline_acc = float((part["baseline_iou"] >= IOU_THRESHOLD).mean())
        always = _counts(proposed, help_, harm)
        frozen = _counts(part["score"].to_numpy() >= threshold, help_, harm)
        out[split] = {
            "incumbent_accuracy": baseline_acc,
            "always_switch": always,
            "frozen_policy": frozen,
        }
    return out


# ---------------------------------------------------------------------------
# 2. risk-coverage on the selection split
# ---------------------------------------------------------------------------


def risk_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Harm among routed versus route rate for every distinct positive score."""

    part = frame[frame["split"] == "select"]
    help_, harm, _ = outcome_masks(part)
    scores = part["score"].to_numpy(dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    cum_route = np.arange(1, scores.size + 1)
    cum_help = np.cumsum(help_[order])
    cum_harm = np.cumsum(harm[order])
    # Keep the last index of each distinct score so ties are routed together.
    distinct_last = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    keep = distinct_last & (sorted_scores > 0.0)
    m = cum_route[keep]
    k = cum_harm[keep]
    h = cum_help[keep]
    n = scores.size
    return pd.DataFrame(
        {
            "threshold": sorted_scores[keep],
            "routed": m,
            "help": h,
            "harm": k,
            "route_rate": m / n,
            "harm_among_routed": k / m,
            "cp_upper": cp_upper(k, m, DELTA),
            "net_gain_pp": 100.0 * (h - k) / n,
        }
    )


def risk_coverage_summary(curve: pd.DataFrame, min_routed: int = 20) -> dict[str, float]:
    sized = curve[curve["routed"] >= min_routed]
    best = sized.loc[sized["harm_among_routed"].idxmin()]
    best_upper = curve.loc[curve["cp_upper"].idxmin()]
    return {
        "min_routed_for_summary": min_routed,
        "lowest_harm_rate_with_min_routed": float(best["harm_among_routed"]),
        "routed_at_lowest_harm_rate": int(best["routed"]),
        "lowest_cp_upper_any_threshold": float(best_upper["cp_upper"]),
        "routed_at_lowest_cp_upper": int(best_upper["routed"]),
        "max_route_rate": float(curve["route_rate"].max()),
    }


# ---------------------------------------------------------------------------
# 3. exact certification power
# ---------------------------------------------------------------------------


def max_certifiable_harms(m: int, alpha: float = ALPHA, delta: float = DELTA) -> int:
    """Largest harm count ``k`` with ``U_delta(k, m) <= alpha``; -1 if none."""

    if m <= 0:
        return -1
    ks = np.arange(0, m + 1)
    ok = np.nan_to_num(cp_upper(ks, m, delta), nan=1.0) <= alpha
    return int(ks[ok].max()) if ok.any() else -1


def certification_power(m: int, true_rate: float) -> float:
    """Probability that ``m`` routed examples with harm rate ``p`` certify."""

    k_max = max_certifiable_harms(m)
    if k_max < 0:
        return 0.0
    return float(binom.cdf(k_max, m, true_rate))


def power_table(
    routed_sizes: Sequence[int], rates: Sequence[float]
) -> pd.DataFrame:
    rows = []
    for m in routed_sizes:
        k_max = max_certifiable_harms(int(m))
        for p in rates:
            rows.append(
                {
                    "routed": int(m),
                    "true_harm_rate": float(p),
                    "max_certifiable_harms": k_max,
                    "power": certification_power(int(m), float(p)),
                }
            )
    return pd.DataFrame(rows)


def routed_needed(true_rate: float, target_power: float = 0.8, limit: int = 20000) -> int | None:
    """Smallest routed sample size whose certification power reaches the target."""

    for m in range(1, limit + 1):
        if certification_power(m, true_rate) >= target_power:
            return m
    return None


# ---------------------------------------------------------------------------
# 4/5. decision procedures shared by the real and controlled analyses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    deployed: bool
    threshold: float | None


def _threshold_counts(
    scores: np.ndarray, help_: np.ndarray, harm: np.ndarray, thresholds: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    routed = scores[None, :] >= np.asarray(thresholds)[:, None]
    m = routed.sum(axis=1)
    h = (routed & help_[None, :]).sum(axis=1)
    k = (routed & harm[None, :]).sum(axis=1)
    return m, h, k


def _cs2_select(m: np.ndarray, h: np.ndarray, k: np.ndarray, thresholds: Sequence[float]) -> int:
    """CS^2 selection rule: harm screen, then max net gain; explicit fallback."""

    net = h - k
    rate = np.where(m > 0, k / np.maximum(m, 1), np.inf)
    admissible = (m >= 1) & (rate <= ALPHA)
    pool = np.flatnonzero(admissible) if admissible.any() else np.arange(len(thresholds))
    # Ties: larger net gain, then larger route count, then lower threshold.
    return int(max(pool, key=lambda i: (net[i], m[i], -thresholds[i])))


def decide(
    method: str,
    scores: np.ndarray,
    help_: np.ndarray,
    harm: np.ndarray,
    select: np.ndarray,
    certify: np.ndarray,
    thresholds: Sequence[float],
) -> Decision:
    """Apply one decision procedure; return whether and which policy deploys."""

    thresholds = tuple(float(t) for t in thresholds)
    if method == "cs2":
        m, h, k = _threshold_counts(scores[select], help_[select], harm[select], thresholds)
        idx = _cs2_select(m, h, k, thresholds)
        mc, _, kc = _threshold_counts(
            scores[certify], help_[certify], harm[certify], (thresholds[idx],)
        )
        upper = cp_upper(kc[0], mc[0], DELTA)
        ok = bool(mc[0] > 0 and upper <= ALPHA)
        return Decision(ok, thresholds[idx] if ok else None)
    pooled = select | certify
    if method in ("ltt_bonferroni", "same_split"):
        m, h, k = _threshold_counts(scores[pooled], help_[pooled], harm[pooled], thresholds)
        level = DELTA / len(thresholds) if method == "ltt_bonferroni" else DELTA
        upper = np.nan_to_num(cp_upper(k, m, level), nan=1.0)
        valid = np.flatnonzero((m > 0) & (upper <= ALPHA))
        if valid.size == 0:
            return Decision(False, None)
        net = h - k
        idx = int(max(valid, key=lambda i: (net[i], m[i], -thresholds[i])))
        return Decision(True, thresholds[idx])
    if method == "empirical_screen":
        m, h, k = _threshold_counts(scores[select], help_[select], harm[select], thresholds)
        rate = np.where(m > 0, k / np.maximum(m, 1), np.inf)
        valid = np.flatnonzero((m >= 1) & (rate <= ALPHA))
        if valid.size == 0:
            return Decision(False, None)
        net = h - k
        idx = int(max(valid, key=lambda i: (net[i], m[i], -thresholds[i])))
        return Decision(True, thresholds[idx])
    raise ValueError(f"unknown method {method!r}")


def real_score_decisions(frame: pd.DataFrame) -> dict[str, dict]:
    help_, harm, _ = outcome_masks(frame)
    scores = frame["score"].to_numpy(dtype=float)
    select = (frame["split"] == "select").to_numpy()
    certify = (frame["split"] == "certify").to_numpy()
    confirm = (frame["split"] == "confirm").to_numpy()
    out: dict[str, dict] = {}
    for method in METHODS:
        decision = decide(method, scores, help_, harm, select, certify, REAL_THRESHOLDS)
        row: dict[str, object] = {"deployed": decision.deployed, "threshold": decision.threshold}
        if decision.deployed:
            routed = scores[confirm] >= decision.threshold
            row["confirm"] = _counts(routed, help_[confirm], harm[confirm])
        out[method] = row
    # Calibration data used by each method, for a matched-budget statement.
    out["calibration_images"] = {
        "cs2": {"select": int(select.sum()), "certify": int(certify.sum())},
        "ltt_bonferroni": {"pooled": int((select | certify).sum())},
    }
    return out


# ---------------------------------------------------------------------------
# 4. controlled verifier study with known true harm
# ---------------------------------------------------------------------------


def decomposition(frame: pd.DataFrame, split: str, thresholds: Sequence[float]) -> pd.DataFrame:
    """Split harm among routed into (decisive share) x (harmful share of decisive).

    A routed switch is decisive when it changes correctness (help or harm).
    Conditional harm equals decisive share times harmful share of decisive.
    """

    part = frame[frame["split"] == split]
    help_, harm, _ = outcome_masks(part)
    scores = part["score"].to_numpy(dtype=float)
    rows = []
    for t in thresholds:
        routed = scores >= t
        m = int(routed.sum())
        h = int((routed & help_).sum())
        k = int((routed & harm).sum())
        rows.append(
            {
                "split": split,
                "threshold": float(t),
                "routed": m,
                "help": h,
                "harm": k,
                "decisive_share": (h + k) / m if m else np.nan,
                "harmful_share_of_decisive": k / (h + k) if (h + k) else np.nan,
                "harm_among_routed": k / m if m else np.nan,
            }
        )
    return pd.DataFrame(rows)


def verifier_truth(
    scores: np.ndarray,
    help_: np.ndarray,
    harm: np.ndarray,
    thresholds: Sequence[float],
    recall: float,
    false_veto: float,
) -> dict[str, np.ndarray]:
    """Population values when a verifier vetoes switches before routing.

    The verifier vetoes a harmful switch with probability ``recall`` and any
    other switch with probability ``false_veto``.  Values are expectations
    over the verifier's randomness on the full manifest.
    """

    t = np.asarray(thresholds, dtype=float)[:, None]
    above = scores[None, :] >= t
    k = (above & harm[None, :]).sum(axis=1).astype(float)
    h = (above & help_[None, :]).sum(axis=1).astype(float)
    other = above.sum(axis=1).astype(float) - k
    routed = (1.0 - recall) * k + (1.0 - false_veto) * other
    harm_routed = (1.0 - recall) * k
    n = float(scores.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = np.where(routed > 0, harm_routed / routed, np.nan)
    return {
        "route_rate": routed / n,
        "conditional_harm": conditional,
        "net_gain_pp": 100.0 * ((1.0 - false_veto) * h - harm_routed) / n,
    }


def min_recall_for_safe_policy(
    frame: pd.DataFrame, thresholds: Sequence[float], false_veto: float
) -> float | None:
    """Smallest verifier recall that makes at least one threshold truly safe."""

    help_, harm, _ = outcome_masks(frame)
    scores = frame["score"].to_numpy(dtype=float)
    for recall in np.round(np.arange(0.0, 1.0001, 0.005), 3):
        truth = verifier_truth(scores, help_, harm, thresholds, float(recall), false_veto)
        if np.nanmin(truth["conditional_harm"]) <= ALPHA:
            return float(recall)
    return None


def controlled_study(
    frame: pd.DataFrame,
    *,
    recalls: Sequence[float] = CONTROLLED_RECALLS,
    false_vetoes: Sequence[float] = CONTROLLED_FALSE_VETOES,
    repeats: int = 2000,
    seed: int = 20261015,
) -> pd.DataFrame:
    """Rerun every decision procedure with a simulated verifier of known quality.

    Real scores, boxes, and help/harm outcomes are kept.  Each repeat draws
    the verifier's vetoes and a fresh image-level partition of the whole
    manifest into selection and certification sets of the real sizes; the
    true values are computed over all rows of the manifest.
    """

    help_, harm, _ = outcome_masks(frame)
    scores_real = frame["score"].to_numpy(dtype=float)
    n_rows = len(frame)
    n_select = int((frame["split"] == "select").sum())
    n_certify = int((frame["split"] == "certify").sum())
    rng = np.random.default_rng(seed)
    records = []
    for false_veto in false_vetoes:
        for recall in recalls:
            truth = verifier_truth(
                scores_real, help_, harm, REAL_THRESHOLDS, recall, false_veto
            )
            veto_prob = np.where(harm, recall, false_veto)
            for rep in range(repeats):
                vetoed = rng.random(n_rows) < veto_prob
                scores = np.where(vetoed, -np.inf, scores_real)
                perm = rng.permutation(n_rows)
                select = np.zeros(n_rows, dtype=bool)
                certify = np.zeros(n_rows, dtype=bool)
                select[perm[:n_select]] = True
                certify[perm[n_select : n_select + n_certify]] = True
                for method in METHODS:
                    decision = decide(
                        method, scores, help_, harm, select, certify, REAL_THRESHOLDS
                    )
                    record = {
                        "false_veto": false_veto,
                        "recall": recall,
                        "repeat": rep,
                        "method": method,
                        "deployed": decision.deployed,
                        "threshold": decision.threshold,
                        "true_route_rate": 0.0,
                        "true_conditional_harm": np.nan,
                        "true_net_gain_pp": 0.0,
                        "unsafe": False,
                    }
                    if decision.deployed:
                        i = REAL_THRESHOLDS.index(decision.threshold)
                        record["true_route_rate"] = float(truth["route_rate"][i])
                        record["true_conditional_harm"] = float(truth["conditional_harm"][i])
                        record["true_net_gain_pp"] = float(truth["net_gain_pp"][i])
                        record["unsafe"] = bool(truth["conditional_harm"][i] > ALPHA)
                    records.append(record)
    return pd.DataFrame.from_records(records)


def summarize_controlled(runs: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    return (
        runs.groupby(list(keys), sort=True)
        .agg(
            deploy_rate=("deployed", "mean"),
            unsafe_deploy_rate=("unsafe", "mean"),
            mean_true_route_rate=("true_route_rate", "mean"),
            mean_true_net_gain_pp=("true_net_gain_pp", "mean"),
        )
        .reset_index()
    )


def truth_table(
    frame: pd.DataFrame,
    recalls: Sequence[float] = CONTROLLED_RECALLS,
    false_vetoes: Sequence[float] = CONTROLLED_FALSE_VETOES,
) -> pd.DataFrame:
    help_, harm, _ = outcome_masks(frame)
    scores = frame["score"].to_numpy(dtype=float)
    rows = []
    for false_veto in false_vetoes:
        for recall in recalls:
            truth = verifier_truth(scores, help_, harm, REAL_THRESHOLDS, recall, false_veto)
            for i, t in enumerate(REAL_THRESHOLDS):
                rows.append(
                    {
                        "false_veto": false_veto,
                        "recall": recall,
                        "threshold": t,
                        "true_route_rate": float(truth["route_rate"][i]),
                        "true_conditional_harm": float(truth["conditional_harm"][i]),
                        "true_net_gain_pp": float(truth["net_gain_pp"][i]),
                        "safe": bool(truth["conditional_harm"][i] <= ALPHA),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

PRIMARY_MANIFESTS = {
    "RefCOCO+": (
        "refcocoplus_scaleup_20260822/predictions_one_expression_per_image.csv",
        0.40,
    ),
    "RefCOCOg": (
        "refcocog_scaleup_20260822/results_refcocog/predictions_reduced_seed20270822.csv",
        0.60,
    ),
    "RefCOCO": (
        "refcoco_scaleup_20260823/results_refcoco/predictions_reduced_seed20270822.csv",
        0.40,
    ),
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest in a canonical row order.

    Rows are sorted by their values so that the simulation's random draws do
    not depend on how the file was ordered or how image identifiers were
    hashed; rows that tie on every value are interchangeable.
    """

    frame = pd.read_csv(path)
    columns = ["split", "score", "baseline_iou", "challenger_iou"]
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def manifest_paths(results_root: Path, manifest_dir: Path | None, system: str = "System A") -> dict[str, tuple[Path, float]]:
    """Locate the primary manifest of each benchmark and its frozen threshold.

    By default the records come from the registry (``records.py``); with
    ``manifest_dir`` they are read from released '<benchmark>__primary.csv'
    files.  ``PRIMARY_MANIFESTS`` keeps the archived 2026-08 paths.
    """

    if manifest_dir is None:
        check_thresholds(system)
        return primary_manifests(system)
    return {
        name: (manifest_dir / f"{name}__primary.csv", threshold)
        for name, (_, threshold) in PRIMARY_MANIFESTS.items()
    }


def run(
    results_root: Path, output_dir: Path, repeats: int, manifest_dir: Path | None = None
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "alpha": ALPHA,
        "delta": DELTA,
        "real_thresholds": list(REAL_THRESHOLDS),
        "controlled_recalls": list(CONTROLLED_RECALLS),
        "controlled_false_vetoes": list(CONTROLLED_FALSE_VETOES),
        "controlled_repeats": repeats,
        "benchmarks": {},
    }
    rates = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
    sizes = sorted(set(np.unique(np.round(np.logspace(0, 4, 160)).astype(int)).tolist()))
    power = power_table(sizes, rates)
    power.to_csv(output_dir / "certification_power.csv", index=False)
    summary["power"] = {
        "min_routed_for_any_certificate": next(
            m for m in range(1, 1000) if max_certifiable_harms(m) >= 0
        ),
        "routed_needed_for_80pct_power": {
            str(p): routed_needed(p) for p in (0.0, 0.01, 0.02, 0.03)
        },
    }
    controlled_frames = []
    for name, (path, threshold) in manifest_paths(results_root, manifest_dir).items():
        frame = load_manifest(path)
        help_, harm, proposed = outcome_masks(frame)
        bench: dict[str, object] = {
            "frozen_threshold": threshold,
            "rows": int(len(frame)),
            "proposed_rows": int(proposed.sum()),
            "help_rows": int(help_.sum()),
            "harm_rows": int(harm.sum()),
        }
        bench["accuracy_versus_audit"] = accuracy_versus_audit(frame, threshold)
        curve = risk_coverage(frame)
        curve.to_csv(output_dir / f"risk_coverage_{name.replace('+', 'plus')}.csv", index=False)
        bench["risk_coverage_select"] = risk_coverage_summary(curve)
        observed = bench["accuracy_versus_audit"]["certify"]["frozen_policy"]
        bench["observed_power"] = {
            "routed": observed["routed"],
            "harm_rate": observed["harm_among_routed"],
            "power_at_observed_rate_and_size": certification_power(
                observed["routed"], observed["harm_among_routed"]
            ),
            "power_at_observed_rate_and_10000_routed": certification_power(
                10000, observed["harm_among_routed"]
            ),
        }
        bench["real_score_decisions"] = real_score_decisions(frame)
        decomp = pd.concat(
            [
                decomposition(frame, "select", REAL_THRESHOLDS),
                decomposition(frame, "certify", (threshold,)),
            ],
            ignore_index=True,
        )
        bench["decomposition"] = decomp.to_dict(orient="records")
        bench["min_recall_for_a_truly_safe_policy"] = {
            str(fv): min_recall_for_safe_policy(frame, REAL_THRESHOLDS, fv)
            for fv in CONTROLLED_FALSE_VETOES
        }
        runs = controlled_study(frame, repeats=repeats)
        runs.insert(0, "benchmark", name)
        controlled_frames.append(runs)
        truth = truth_table(frame)
        truth.insert(0, "benchmark", name)
        truth.to_csv(
            output_dir / f"controlled_truth_{name.replace('+', 'plus')}.csv", index=False
        )
        summary["benchmarks"][name] = bench
    all_runs = pd.concat(controlled_frames, ignore_index=True)
    all_runs.to_csv(output_dir / "controlled_runs.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    controlled = summarize_controlled(all_runs, ("benchmark", "false_veto", "recall", "method"))
    controlled.to_csv(output_dir / "controlled_summary.csv", index=False)
    summary["controlled_max_unsafe_deploy_rate"] = (
        controlled.groupby("method")["unsafe_deploy_rate"].max().to_dict()
    )
    summary = _jsonable(summary)
    (output_dir / "failure_diagnosis.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--results-root", type=Path, default=here / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=here / "results" / "failure_diagnosis_20261016"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="directory of released '<benchmark>__primary.csv' audit records",
    )
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    summary = run(args.results_root, args.output_dir, args.repeats, args.manifest_dir)
    print(json.dumps(summary["power"], indent=2))
    for name, bench in summary["benchmarks"].items():
        print(name, json.dumps(bench["risk_coverage_select"]))
        print(name, json.dumps(bench["real_score_decisions"]))


if __name__ == "__main__":
    main()
