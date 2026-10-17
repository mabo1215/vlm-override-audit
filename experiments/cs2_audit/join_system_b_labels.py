"""Join ground truth to System B generation records in the preregistered order.

Generation (``system_b.py``) never reads labels.  This command adds them one
set at a time and refuses to run out of order (Design.md section 15.2):

1. ``families``  -- label-free threshold family from training switch scores,
                    written into the protocol file;
2. ``join --stage train``    -- needs the protocol file committed;
3. ``join --stage select``   -- needs the threshold family committed;
4. ``select``    -- freezes the selected rule into the protocol file;
5. ``join --stage certify``  -- needs the frozen rule committed; once only;
6. ``join --stage confirm``  -- needs the certification join logged;
7. ``assemble``  -- concatenates the four joined sets into an audit manifest
                    and checks that the audit reproduces the frozen rule.

Records that carry the secondary listwise score (Design.md section 15.3) are
handled in the same commands: each stage's single join writes both scores'
columns, ``families`` and ``select`` write the listwise family and rule under
the arm name ``<ARM>-listwise``, the selection join needs both families and
the certification join needs both frozen rules, and ``assemble`` audits each
score separately.  Neither rule can be chosen after labels it would be
certified on have been read.

The V1-gated family (Design.md section 15.4; arm ``<ARM>-gated``) switches
iff the pairwise score is at least tau and V1's harm probability is at most
nu.  If the protocol sets ``require_verifier_gate``, ``freeze-verifier``
must record V1's SHA-256, the SHA-256 of its predictions for the arm and the
gated family (tau from the arm's pairwise family, nu from the declared grid)
before the selection join; the certification join then also needs the frozen
gated rule.  V1 cannot be replaced once frozen.

A transfer test (G3b, protocol key ``transfer``) applies rules frozen on a
source benchmark to a target benchmark's confirmation set.  ``transfer-apply``
writes the label-free routing decisions and must be logged and committed
before any label join on the target benchmark; ``transfer-evaluate`` scores
them once the target's confirmation labels are joined, after checking that
the joined scores reproduce the committed decisions.

Every join appends input and output SHA-256 and the UTC time to
``label_join_log.jsonl``.  Records follow two consistency rules by
construction: a record without a switch has alternative IoU equal to default
IoU, and switch scores below the tie tolerance are stored as zero.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import Policy, evaluate_policy, run_protocol, select_policy
from .system_b import TIE_TOLERANCE, box_iou

STAGES = ("train", "select", "certify", "confirm")
MANIFEST_COLUMNS = ["split", "image_id", "expression_id", "score", "baseline_iou", "challenger_iou"]
LISTWISE_COLUMNS = ["score_listwise", "challenger_iou_listwise"]
# score name -> (record score field, record chosen-index field, arm suffix)
SCORES = {
    "pairwise": ("switch_score", "chosen_index", ""),
    "listwise": ("listwise_switch_score", "listwise_chosen_index", "-listwise"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_committed(path: Path) -> bool:
    """True if ``path`` is tracked by git and has no uncommitted change."""

    cwd = path.resolve().parent
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path.name], cwd=cwd, capture_output=True)
    if tracked.returncode != 0:
        return False
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path.name], cwd=cwd)
    return clean.returncode == 0


def load_generation(paths: list[Path]) -> pd.DataFrame:
    """Last successful record per key across one or more JSONL files."""

    latest: dict[str, dict] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in record:
                    continue
                latest[str(record["record_key"])] = record
    if not latest:
        raise SystemExit("no successful generation records found")
    return pd.DataFrame(latest.values())


def load_truth(gt_csv: Path) -> dict[str, tuple[float, float, float, float]]:
    frame = pd.read_csv(gt_csv, dtype={"expression_id": str})
    return {
        f"{row.benchmark}:{row.expression_id}": (row.x1, row.y1, row.x2, row.y2)
        for row in frame.itertuples(index=False)
    }


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def has_listwise(records: pd.DataFrame) -> bool:
    """True if the records carry the listwise score (all or none must)."""

    if "listwise_switch_score" not in records.columns:
        return False
    present = ~records["listwise_switch_score"].isna()
    if not present.all():
        raise SystemExit("some records carry the listwise score and some do not")
    return True


def _scored(record: dict, gt, baseline: float, score_field: str, chosen_field: str) -> tuple[float, float]:
    chosen = record.get(chosen_field)
    score = float(record.get(score_field) or 0.0)
    if _missing(chosen) or chosen == record.get("default_index") or score < TIE_TOLERANCE:
        return 0.0, baseline
    return score, box_iou(record["candidates"][int(chosen)]["box"], gt)


def joined_rows(records: pd.DataFrame, truth: dict) -> pd.DataFrame:
    listwise = has_listwise(records)
    rows = []
    for record in records.to_dict("records"):
        key = str(record["record_key"])
        if key not in truth:
            raise SystemExit(f"no ground truth for {key}")
        gt = truth[key]
        default = record.get("default_box")
        baseline = box_iou(default, gt) if not _missing(default) and default else 0.0
        score, challenger = _scored(record, gt, baseline, "switch_score", "chosen_index")
        row = {
            "benchmark": record["benchmark"],
            "split": record["split"],
            "image_id": record["image_id"],
            "expression_id": str(record["expression_id"]),
            "score": score,
            "baseline_iou": baseline,
            "challenger_iou": challenger,
        }
        if listwise:
            row["score_listwise"], row["challenger_iou_listwise"] = _scored(
                record, gt, baseline, "listwise_switch_score", "listwise_chosen_index"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def score_views(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One audit manifest per score present in a joined frame, in the standard columns."""

    views = {"pairwise": frame[MANIFEST_COLUMNS]}
    if set(LISTWISE_COLUMNS) <= set(frame.columns):
        views["listwise"] = frame.drop(columns=["score", "challenger_iou"]).rename(
            columns={"score_listwise": "score", "challenger_iou_listwise": "challenger_iou"}
        )[MANIFEST_COLUMNS]
    return views


def score_arms(arm: str, listwise: bool) -> dict[str, str]:
    names = ["pairwise", "listwise"] if listwise else ["pairwise"]
    return {name: arm + SCORES[name][2] for name in names}


def read_protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_protocol(path: Path, protocol: dict) -> None:
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def logged_stages(log: Path, arm: str, benchmark: str) -> set[str]:
    if not log.exists():
        return set()
    stages = set()
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("arm") == arm and entry.get("benchmark") == benchmark:
                stages.add(entry["stage"])
    return stages


def policies_for(protocol: dict, arm: str, benchmark: str) -> list[Policy]:
    family = protocol["threshold_families"][arm][benchmark]
    return [Policy(name=f"score_ge_{t:.6g}", threshold=float(t)) for t in family]


GATED = "-gated"
NUS = (0.02, 0.05, 0.10, 0.20, 0.30)


def gate_required(protocol: dict) -> bool:
    return bool(protocol.get("require_verifier_gate"))


def gated_policies_for(protocol: dict, arm: str, benchmark: str) -> list[Policy]:
    from .verifier import gated_family

    spec = protocol["gated_families"][arm + GATED][benchmark]
    return gated_family(spec["taus"], spec["nus"])


def with_verifier(frame: pd.DataFrame, protocol: dict, arm: str, benchmark: str, predictions: Path | None) -> pd.DataFrame:
    """Attach V1's frozen predictions to a pairwise manifest after checking their hash."""

    from .verifier import attach_verifier

    if predictions is None:
        raise SystemExit("--verifier-predictions is required for the gated rule")
    expected = protocol.get("verifier_predictions", {}).get(arm, {}).get("sha256")
    if expected != sha256_file(predictions):
        raise SystemExit("the verifier predictions do not match the hash frozen in the protocol")
    return attach_verifier(frame, pd.read_csv(predictions), benchmark)


def cmd_freeze_verifier(args: argparse.Namespace) -> None:
    protocol = read_protocol(args.protocol)
    families = protocol.get("threshold_families", {}).get(args.arm, {})
    if not families:
        raise SystemExit(f"declare the {args.arm} threshold families first")
    for benchmark in families:
        if "select" in logged_stages(args.log, args.arm, benchmark):
            raise SystemExit(f"{args.arm} {benchmark}: selection labels were already read; V1 can no longer be frozen")
    digest = sha256_file(args.verifier)
    frozen = protocol.get("verifier")
    if frozen and frozen["sha256"] != digest:
        raise SystemExit("a different V1 is already frozen in the protocol")
    sidecar = args.verifier.with_suffix(".json")
    protocol["verifier"] = {
        "file": args.verifier.name,
        "sha256": digest,
        "metadata": json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {},
    }
    protocol.setdefault("verifier_predictions", {})[args.arm] = {"file": args.predictions.name, "sha256": sha256_file(args.predictions)}
    gated = protocol.setdefault("gated_families", {}).setdefault(args.arm + GATED, {})
    for benchmark, taus in families.items():
        gated[benchmark] = {"taus": list(taus), "nus": list(NUS)}
    write_protocol(args.protocol, protocol)
    print(json.dumps({"verifier_sha256": digest, "arm": args.arm + GATED, "benchmarks": sorted(gated)}))
    print(f"Wrote V1 and the gated family to {args.protocol}; commit it before the selection join.")


def cmd_families(args: argparse.Namespace) -> None:
    records = load_generation(args.generation)
    protocol = read_protocol(args.protocol)
    for name, arm in score_arms(args.arm, has_listwise(records)).items():
        field = SCORES[name][0]
        families = protocol.setdefault("threshold_families", {}).setdefault(arm, {})
        for benchmark, group in records[records["split"] == "train"].groupby("benchmark"):
            scores = group[field].astype(float)
            positive = scores[scores >= TIE_TOLERANCE].to_numpy()
            if positive.size == 0:
                raise SystemExit(f"{benchmark} ({name}): no training switches; the family cannot be declared")
            quantiles = [float(np.quantile(positive, q)) for q in (0.25, 0.5, 0.75, 0.9)]
            families[benchmark] = sorted({TIE_TOLERANCE, *quantiles})
            print(json.dumps({"arm": arm, "benchmark": benchmark, "training_switches": int(positive.size), "family": families[benchmark]}))
    write_protocol(args.protocol, protocol)
    print(f"Wrote families to {args.protocol}; commit it before the selection join.")


def transfer_routes(records: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    """Label-free routing of the target's confirmation records under the source's frozen rules."""

    transfer = protocol["transfer"]
    part = records[(records["benchmark"] == transfer["target_benchmark"]) & (records["split"] == transfer["stage"])]
    if part.empty:
        raise SystemExit("no target confirmation records")
    out = pd.DataFrame({"record_key": part["record_key"].astype(str), "expression_id": part["expression_id"].astype(str)})
    for name, arm in score_arms(transfer["source_arm"], has_listwise(part)).items():
        threshold = protocol["frozen_rules"][arm][transfer["source_benchmark"]]["threshold"]
        scores = part[SCORES[name][0]].astype(float).where(lambda v: v >= TIE_TOLERANCE, 0.0)
        out[f"routed_{name}"] = (scores >= float(threshold)).to_numpy()
    return out.sort_values("record_key").reset_index(drop=True)


def cmd_transfer_apply(args: argparse.Namespace) -> None:
    protocol = read_protocol(args.protocol)
    if not is_committed(args.protocol):
        raise SystemExit("the protocol file must be committed before the transfer is applied")
    transfer = protocol.get("transfer")
    if not transfer:
        raise SystemExit("the protocol declares no transfer")
    target = transfer["target_benchmark"]
    if args.log.exists():
        for line in args.log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line) if line.strip() else {}
            if entry.get("benchmark") == target:
                raise SystemExit(f"{target} already has a logged {entry.get('stage')} entry; the transfer must come first")
    routes = transfer_routes(load_generation(args.generation), protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{transfer['arm']}_{target}_routes.csv"
    routes.to_csv(out, index=False)
    entry = {
        "arm": transfer["arm"],
        "benchmark": target,
        "stage": "transfer-apply",
        "rows": int(len(routes)),
        "routed": {c: int(routes[c].sum()) for c in routes.columns if c.startswith("routed_")},
        "protocol_sha256": sha256_file(args.protocol),
        "generation_sha256": [sha256_file(p) for p in args.generation],
        "output": out.name,
        "output_sha256": sha256_file(out),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, indent=2))
    print(f"Commit {out.name} and the log before any {target} label join.")


def cmd_transfer_evaluate(args: argparse.Namespace) -> None:
    protocol = read_protocol(args.protocol)
    transfer = protocol["transfer"]
    target = transfer["target_benchmark"]
    routes_path = args.out_dir / f"{transfer['arm']}_{target}_routes.csv"
    if not is_committed(routes_path):
        raise SystemExit("the transfer routes must be committed before they are evaluated")
    applied = [json.loads(l) for l in args.log.read_text(encoding="utf-8").splitlines() if l.strip()]
    applied = [e for e in applied if e.get("stage") == "transfer-apply" and e.get("arm") == transfer["arm"]]
    if not applied or applied[-1]["output_sha256"] != sha256_file(routes_path):
        raise SystemExit("the transfer routes do not match the logged application")
    routes = pd.read_csv(routes_path, dtype={"expression_id": str})
    joined = pd.read_csv(args.out_dir / f"{transfer['target_arm']}_{target}_{transfer['stage']}.csv", dtype={"expression_id": str})
    views = score_views(joined)
    result = {"arm": transfer["arm"], "source": f"{transfer['source_arm']} {transfer['source_benchmark']}", "target": target}
    for name, arm in score_arms(transfer["source_arm"], "listwise" in views).items():
        rule = protocol["frozen_rules"][arm][transfer["source_benchmark"]]
        frame = views[name]
        recomputed = (frame.set_index("expression_id")["score"] >= float(rule["threshold"])).sort_index()
        committed = routes.set_index("expression_id")[f"routed_{name}"].astype(bool).reindex(recomputed.index)
        if not recomputed.equals(committed):
            raise SystemExit(f"{name}: the joined scores do not reproduce the committed transfer routes")
        policy = Policy(name=rule["policy"], threshold=float(rule["threshold"]))
        audit = evaluate_policy(frame, policy, split=transfer["stage"], alpha=args.alpha, delta=args.delta)
        result[name] = audit.to_dict()
    summary = args.out_dir / f"{transfer['arm']}_{target}_transfer_summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


def cmd_join(args: argparse.Namespace) -> None:
    protocol = read_protocol(args.protocol)
    if not is_committed(args.protocol):
        raise SystemExit("the protocol file must be committed before any label join")
    records = load_generation(args.generation)
    records = records[(records["benchmark"] == args.benchmark) & (records["split"] == args.stage)]
    if records.empty:
        raise SystemExit("no generation records for this benchmark and set")
    done = logged_stages(args.log, args.arm, args.benchmark)
    if args.stage in done:
        raise SystemExit(f"{args.stage} labels were already joined for {args.benchmark}; joins happen once")
    transfer = protocol.get("transfer")
    if transfer and args.benchmark == transfer["target_benchmark"]:
        if "transfer-apply" not in logged_stages(args.log, transfer["arm"], args.benchmark):
            raise SystemExit(f"apply the {transfer['arm']} transfer before any {args.benchmark} label join")
    arms = score_arms(args.arm, has_listwise(records)).values()
    if gate_required(protocol):
        arms = [*arms, args.arm + GATED]
    if args.stage == "select":
        if any(args.benchmark not in protocol.get("threshold_families", {}).get(a, {}) for a in arms if not a.endswith(GATED)):
            raise SystemExit("declare and commit every threshold family before the selection join")
        if gate_required(protocol) and args.benchmark not in protocol.get("gated_families", {}).get(args.arm + GATED, {}):
            raise SystemExit("freeze V1 and commit the gated family before the selection join")
    if args.stage == "certify":
        if any(args.benchmark not in protocol.get("frozen_rules", {}).get(a, {}) for a in arms):
            raise SystemExit("freeze and commit every selected rule before the certification join")
    if args.stage == "confirm" and "certify" not in done:
        raise SystemExit("join certification labels before confirmation labels")
    frame = joined_rows(records, load_truth(args.gt_csv))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.arm}_{args.benchmark}_{args.stage}.csv"
    columns = MANIFEST_COLUMNS + (LISTWISE_COLUMNS if set(LISTWISE_COLUMNS) <= set(frame.columns) else [])
    frame[columns].to_csv(out, index=False)
    entry = {
        "arm": args.arm,
        "benchmark": args.benchmark,
        "stage": args.stage,
        "rows": int(len(frame)),
        "protocol_sha256": sha256_file(args.protocol),
        "generation_sha256": [sha256_file(p) for p in args.generation],
        "gt_sha256": sha256_file(args.gt_csv),
        "output": out.name,
        "output_sha256": sha256_file(out),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    print(json.dumps(entry, indent=2))


def cmd_select(args: argparse.Namespace) -> None:
    protocol = read_protocol(args.protocol)
    frame = pd.read_csv(args.out_dir / f"{args.arm}_{args.benchmark}_select.csv", dtype={"expression_id": str})
    views = score_views(frame)
    for name, arm in score_arms(args.arm, "listwise" in views).items():
        policy = select_policy(views[name], policies_for(protocol, arm, args.benchmark), alpha=args.alpha)
        protocol.setdefault("frozen_rules", {}).setdefault(arm, {})[args.benchmark] = {
            "policy": policy.name,
            "threshold": policy.threshold,
        }
        print(json.dumps({"arm": arm, "benchmark": args.benchmark, "frozen": policy.name}))
    if args.benchmark in protocol.get("gated_families", {}).get(args.arm + GATED, {}):
        frame = with_verifier(views["pairwise"], protocol, args.arm, args.benchmark, args.verifier_predictions)
        policy = select_policy(frame, gated_policies_for(protocol, args.arm, args.benchmark), alpha=args.alpha)
        protocol.setdefault("frozen_rules", {}).setdefault(args.arm + GATED, {})[args.benchmark] = {
            "policy": policy.name,
            "threshold": policy.threshold,
            "nu": policy.nu,
        }
        print(json.dumps({"arm": args.arm + GATED, "benchmark": args.benchmark, "frozen": policy.name}))
    write_protocol(args.protocol, protocol)
    print(f"Commit {args.protocol} before the certification join.")


def cmd_assemble(args: argparse.Namespace) -> None:
    protocol = read_protocol(args.protocol)
    parts = [pd.read_csv(args.out_dir / f"{args.arm}_{args.benchmark}_{s}.csv", dtype={"expression_id": str}) for s in STAGES]
    views = score_views(pd.concat(parts, ignore_index=True))
    audits = [(arm, views[name], policies_for(protocol, arm, args.benchmark)) for name, arm in score_arms(args.arm, "listwise" in views).items()]
    if args.benchmark in protocol.get("gated_families", {}).get(args.arm + GATED, {}):
        gated = with_verifier(views["pairwise"], protocol, args.arm, args.benchmark, args.verifier_predictions)
        audits.append((args.arm + GATED, gated, gated_policies_for(protocol, args.arm, args.benchmark)))
    for arm, manifest, policies in audits:
        result = run_protocol(manifest, policies, alpha=args.alpha, delta=args.delta)
        frozen = protocol["frozen_rules"][arm][args.benchmark]["policy"]
        if result["certify"]["policy"] != frozen:
            raise SystemExit(f"{arm}: audit selected {result['certify']['policy']} but {frozen} was frozen")
        manifest.to_csv(args.out_dir / f"{arm}_{args.benchmark}_manifest.csv", index=False)
        summary = args.out_dir / f"{arm}_{args.benchmark}_audit_summary.json"
        summary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"arm": arm, **{k: result[k] for k in ("certify", "confirm")}}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, generation: bool = True) -> None:
        p.add_argument("--protocol", type=Path, required=True)
        p.add_argument("--arm", required=True, help="e.g. G1 or G3")
        if generation:
            p.add_argument("--generation", type=Path, nargs="+", required=True)

    p = sub.add_parser("families")
    common(p)
    p.set_defaults(func=cmd_families)

    p = sub.add_parser("join")
    common(p)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--gt-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.set_defaults(func=cmd_join)

    for name, func in (("select", cmd_select), ("assemble", cmd_assemble)):
        p = sub.add_parser(name)
        common(p, generation=False)
        p.add_argument("--benchmark", required=True)
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--delta", type=float, default=0.05)
        p.add_argument("--verifier-predictions", type=Path, help="V1's frozen predictions (gated rule)")
        p.set_defaults(func=func)

    p = sub.add_parser("transfer-apply", help="label-free routing of the transfer target under the source's frozen rules")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--generation", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.set_defaults(func=cmd_transfer_apply)

    p = sub.add_parser("transfer-evaluate", help="score the committed transfer routes on the joined confirmation labels")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.set_defaults(func=cmd_transfer_evaluate)

    p = sub.add_parser("freeze-verifier", help="record V1, its predictions and the gated family before the selection join")
    common(p, generation=False)
    p.add_argument("--verifier", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.set_defaults(func=cmd_freeze_verifier)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
