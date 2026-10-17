"""Write the manuscript's result tables from computed outputs (no hand copying).

Inputs: ``paper_numbers.py`` output, the audit summaries named in the registry,
``audit_revision_cpu.py`` output and ``listwise_check.py`` output.  Each table is written to its own file
under ``paper/figs/`` and included by the manuscript with ``\\input``.

Example:
    python -m experiments.cs2_audit.paper_numbers
    python -m experiments.cs2_audit.write_paper_tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .records import ROOT, load_registry

FIGS = ROOT / "paper" / "figs"


def ordered(numbers: dict, system: str):
    """Benchmarks in registry order (the order used throughout the paper)."""

    order = list(load_registry()["systems"][system])
    return [(name, numbers[system][name]) for name in order if name in numbers[system]]


def pct(value, digits: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{digits}f}\\%"


def pp(value, digits: int = 2) -> str:
    return f"${value:+.{digits}f}$\\pp"


def num(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def short(name: str) -> str:
    return name


def main_table(numbers: dict, system: str, label: str = "tab:main", intro: str = "", tau_digits: int = 1) -> str:
    rows = []
    for bench, data in ordered(numbers, system):
        c = data["certify"]
        sel, alw, rnd, ora = c["selected"], c["always_switch"], c["proposal_matched_random"], c["oracle"]
        tau = data["threshold"]
        rows.append(
            f"{short(bench)} & always switch$^\\dagger$ & {num(alw['m'])} & {num(alw['h'])} & {num(alw['k'])} & {pct(alw['harm_among_switches'])} & {pct(alw['upper'])} & {pp(alw['accuracy_change_pp'])} & -- \\\\"
        )
        rows.append(
            f"$n={num(sel['n'])}$ & selected, $\\tau={tau:.{tau_digits}f}$ & {sel['m']} & {sel['h']} & {sel['k']} & \\textbf{{{pct(sel['harm_among_switches'])}}} & \\textbf{{{pct(sel['upper'])}}} & {pp(sel['accuracy_change_pp'])} & not certified \\\\"
            if sel["upper"] is not None and sel["upper"] > 0.05
            else f"$n={num(sel['n'])}$ & selected, $\\tau={tau:.{tau_digits}f}$ & {sel['m']} & {sel['h']} & {sel['k']} & {pct(sel['harm_among_switches'])} & {pct(sel['upper'])} & {pp(sel['accuracy_change_pp'])} & certified \\\\"
        )
        rows.append(
            f" & random proposals$^\\ddagger$ & {rnd['m']} & {rnd['expected_h']:.1f} & {rnd['expected_k']:.1f} & {pct(rnd['harm_among_switches'])} & -- & {pp(rnd['accuracy_change_pp'])} & -- \\\\"
        )
        rows.append(
            f" & hindsight oracle & {ora['m']} & {ora['h']} & 0 & 0.0\\% & -- & {pp(ora['accuracy_change_pp'])} & uses labels \\\\"
        )
        rows.append("\\midrule")
    rows = rows[:-1]
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{" + intro + "Accuracy view and audit view on the certification sets.  ``Harm among\n"
        "switches'' is $k/m$; ``Upper bound'' is the exact one-sided 95\\% bound $U$; the\n"
        "limit is 5\\%.  Accuracy change is in percentage points (pp) over all $n$\n"
        "images.  The selected rule is the only rule tested formally.  $^\\dagger$Always\n"
        "switch takes every proposed switch ($s>0$); it was not in the declared rule\n"
        "family, so its bound is descriptive.  $^\\ddagger$Random proposals switch the\n"
        "same number of images, drawn uniformly among those with a proposed switch\n"
        "(expected values).\\label{" + label + "}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llrrrrrrl@{}}\n\\toprule\n"
        "& & Switches & Helps & Harms & Harm among & Upper & Accuracy & \\\\\n"
        "Benchmark & Rule & $m$ & $h$ & $k$ & switches $k/m$ & bound $U$ & change & Verdict \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def confirm_table(numbers: dict, system: str, label: str = "tab:confirm", tau_digits: int = 1) -> str:
    rows = []
    for bench, data in ordered(numbers, system):
        c, f = data["certify"]["selected"], data["confirm"]["selected"]
        rows.append(
            f"{bench} ($\\tau={data['threshold']:.{tau_digits}f}$) & {num(c['n'])} & {c['m']} & {c['h']} & {c['k']} & {pct(c['lower'])} & {pct(c['upper'])} & "
            f"{num(f['n'])} & {f['m']} & {f['h']} & {f['k']} & {pct(f['harm_among_switches'])} & {pct(f['lower'])} \\\\"
        )
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{Selected rules on the certification and confirmation sets.  $L$ and\n"
        "$U$ are exact one-sided 95\\% lower and upper bounds on harm among switches;\n"
        "$U$ on the certification set decides certification, and the confirmation\n"
        "values are descriptive.  A lower bound above 5\\% means that the rule's true\n"
        "harm rate exceeds the limit, so more test data would not lead to\n"
        "certification.\\label{" + label + "}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}\n\\toprule\n"
        "& \\multicolumn{6}{c}{Certification set} & \\multicolumn{6}{c}{Confirmation set} \\\\\n"
        "\\cmidrule(lr){2-7}\\cmidrule(l){8-13}\n"
        "Benchmark & $n$ & $m$ & $h$ & $k$ & $L$ & $U$ & $n$ & $m$ & $h$ & $k$ & $k/m$ & $L$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def auroc_table(numbers: dict, system: str) -> str:
    rows = []
    for bench, data in ordered(numbers, system):
        c, s = data["certify"]["auroc"], data["select"]["auroc"]
        red = data["certify_auroc_reductions"]
        rows.append(
            f"{bench} & {c['auroc']:.2f} ({c['n_help']}/{c['n_harm']}) & {s['auroc']:.2f} ({s['n_help']}/{s['n_harm']}) & "
            f"${red['mean']:.3f}\\pm{red['sd']:.3f}$ \\\\"
        )
    return (
        "\\begin{table}[H]\n"
        "\\caption{Help-versus-harm AUROC of the switch score.  Counts in parentheses\n"
        "are the helpful and harmful opportunities in each set.  The last column is the\n"
        "mean $\\pm$ standard deviation of the certification-set value over the three\n"
        "reductions of \\Cref{sec:data}.\\label{tab:auroc}}\n"
        "\\centering\n\\small\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
        "Benchmark & Certification & Selection & Three reductions \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def selection_trace_table(system: str) -> str:
    rows = []
    for bench, entry in load_registry()["systems"][system].items():
        summary = json.loads((ROOT / entry["audit_summary"]).read_text(encoding="utf-8"))
        trace = sorted(summary["selection_trace"], key=lambda r: float(r["policy"].rsplit("_", 1)[1]))
        for i, r in enumerate(trace):
            tau = float(r["policy"].rsplit("_", 1)[1])
            cells = [f"{tau:.1f}", str(r["route_count"]), str(r["help_count"]), str(r["harm_count"]), pp(100 * r["net_accuracy_gain"])]
            if r["selected"]:
                cells = [f"\\textbf{{{c}}}" if "$" not in c else c.replace("$", "$\\mathbf{", 1).replace("$\\pp", "}$\\pp") for c in cells]
            rows.append(f"{bench if i == 0 else ''} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    return (
        "\\begin{table}[H]\n"
        "\\caption{All six declared thresholds on the selection sets.  None has observed\n"
        "harm among switches of at most 5\\%; bold rows are the selected rules (largest\n"
        "accuracy gain).\\label{tab:selection-trace}}\n"
        "\\centering\n\\small\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Benchmark & $\\tau$ & Switches & Helps & Harms & Accuracy change \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def random_table(numbers: dict, system: str) -> str:
    rows = []
    for bench, data in ordered(numbers, system):
        for split, label in (("certify", "certification"), ("confirm", "confirmation")):
            c = data[split]
            sel, rnd, boot = c["selected"], c["proposal_matched_random"], c["bootstrap_vs_random"]
            g, h = boot["gain_diff_pp_95"], boot["harm_diff_95"]
            rows.append(
                f"{bench if split == 'certify' else ''} & {label} & {rnd['pool']} & {pct(sel['harm_among_switches'])} & {pct(rnd['harm_among_switches'])} & "
                f"$[{100 * h[0]:+.1f},{100 * h[1]:+.1f}]$ & {pp(sel['accuracy_change_pp'])} & {pp(rnd['accuracy_change_pp'])} & $[{g[0]:+.2f},{g[1]:+.2f}]$\\pp \\\\"
            )
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{Selected rule versus random proposals at the same switch count.\n"
        "``Proposals'': images with a proposed switch ($s>0$), from which the random\n"
        "rule draws.  Intervals are 95\\% image-level bootstrap intervals (500\n"
        "resamples, rule frozen) of the selected-minus-random difference, in\n"
        "percentage points.  These intervals are descriptive and play no role in the\n"
        "certificate.\\label{tab:matched-random}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llrrrcrrc@{}}\n\\toprule\n"
        "& & & \\multicolumn{3}{c}{Harm among switches} & \\multicolumn{3}{c}{Accuracy change} \\\\\n"
        "\\cmidrule(lr){4-6}\\cmidrule(l){7-9}\n"
        "Benchmark & Set & Proposals & Selected & Random & Difference & Selected & Random & Difference \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def continuous_iou_table(revision: dict) -> str:
    rows = []
    order = [name for name in load_registry()["systems"]["System A"] if name in revision["benchmarks"]]
    for bench, data in ((name, revision["benchmarks"][name]) for name in order):
        for split, label in (("certify", "certification"), ("confirm", "confirmation")):
            d = data[split]
            rows.append(
                f"{bench if split == 'certify' else ''} & {label} & {num(d['rows'])} & {d['route_count']} & {d['mean_baseline_iou']:.3f} & "
                f"{d['mean_challenger_iou']:.3f} & ${d['mean_delta_iou']:+.3f}$ / ${d['routed_mean_delta_iou']:+.3f}$ \\\\"
            )
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{Mean IoU of the default and alternative boxes on the held-out sets.\n"
        "``All $\\Delta$'' averages the alternative-minus-default IoU over all images;\n"
        "``switched $\\Delta$'' averages it over the images that the selected rule\n"
        "switches.\\label{tab:continuous-iou}}\n"
        "\\centering\n\\small\n\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\n"
        "Benchmark & Set & $n$ & Switched & Mean IoU($b$) & Mean IoU($c$) & All $\\Delta$ / switched $\\Delta$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


PROCEDURES = (
    ("empirical_screen", "No bound (selection set)"),
    ("same_split", "Same data, no correction"),
    ("cs2", "Split audit (ours)"),
    ("ltt_bonferroni", "LTT, pooled sets"),
)


def procedures_table(diagnosis_dir: Path) -> str:
    """Decision procedures: real-score verdict, worst unsafe rate, deployment and realized gain."""

    import pandas as pd

    summary = json.loads((diagnosis_dir / "failure_diagnosis.json").read_text(encoding="utf-8"))
    controlled = pd.read_csv(diagnosis_dir / "controlled_summary.csv")
    order = [b for b in load_registry()["systems"]["System A"]]
    rows = []
    for method, label in PROCEDURES:
        real = [summary["benchmarks"][b]["real_score_decisions"][method]["deployed"] for b in order]
        verdict = "no switch" if not any(real) else "switches on " + ", ".join(b for b, d in zip(order, real) if d)
        worst = controlled.loc[controlled["method"] == method, "unsafe_deploy_rate"].max()
        cells = [label, verdict, pct(worst)]
        for recall, column in ((0.95, "deploy_rate"), (0.90, "mean_true_net_gain_pp"), (0.95, "mean_true_net_gain_pp")):
            part = controlled[(controlled["method"] == method) & (controlled["false_veto"] == 0.1) & (controlled["recall"] == recall)].set_index("benchmark")
            values = [part.loc[b, column] for b in order]
            cells.append("/".join(f"{100 * v:.0f}" for v in values) + "\\%" if column == "deploy_rate" else "/".join(f"{v:+.2f}" for v in values))
        rows.append(" & ".join(cells) + " \\\\")
    return (
        "\\begin{table}[htbp]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{Four ways to decide whether to deploy a threshold, all with\n"
        "$\\alpha=\\delta=5\\%$.  ``Real scores'': decision on the actual switch scores of\n"
        "all three benchmarks.  ``Unsafe'': worst share of runs, over all 99 controlled\n"
        "settings, that deploy a rule whose true harm exceeds 5\\%.  The last three\n"
        "columns use a verifier with 10\\% false vetoes that catches 90\\% or 95\\% of\n"
        "harmful switches, where safe rules exist: the share of runs that deploy a rule,\n"
        "and the mean accuracy change actually realized in percentage points, averaged\n"
        "over all runs (RefCOCO+/RefCOCOg/RefCOCO).\\label{tab:procedures}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "& & & Deploys & \\multicolumn{2}{c}{Realized gain (pp)} \\\\\n"
        "\\cmidrule(l){5-6}\n"
        "Procedure & Real scores & Unsafe & at 95\\% & at 90\\% & at 95\\% \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def listwise_table(check: dict, seed: str = "20270822") -> str:
    """Pairwise (audited) versus listwise score, from ``listwise_check.py``."""

    def counts(o: dict) -> str:
        return f"{o['m']}/{o['h']}/{o['k']}"

    rows = []
    for bench in ("RefCOCO+", "RefCOCOg", "RefCOCO"):
        result = check[bench][seed]
        passes = {"pairwise": result["check"]["mean_pairwise_passes"], "listwise": result["check"]["mean_listwise_passes"]}
        for i, (score, label) in enumerate((("pairwise", "Pairwise"), ("listwise", "Listwise"))):
            r = result[score]
            declared, fine = r["declared_family"], r["dense_family"]
            rows.append(
                f"{bench if i == 0 else ''} & {label} & {passes[score]:.1f} & {declared['threshold']:.1f} & "
                f"{counts(declared['certify'])} & {pct(declared['certify']['upper'])} & {counts(declared['confirm'])} & "
                f"{r['certify']['auroc']:.2f} & {pct(r['certify']['top_m_matched_to_selected_pairwise']['harm_among_switches'])} & "
                f"{fine['threshold']:.2f} & {pct(fine['certify']['upper'])} \\\\"
            )
        rows.append("\\midrule")
    rows = rows[:-1]
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{The audited pairwise score and the listwise score recorded by the same\n"
        "generator, on the same records (post hoc for the listwise score; not a\n"
        "certificate).  Passes: scoring forward passes per record.  $\\tau$: threshold\n"
        "selected from the declared family; $m/h/k$: switches, helps and harms;\n"
        "$U$: one-sided 95\\% upper bound.  AUROC: help versus harm on the certification\n"
        "set.  Matched: harm among the top-scoring switches on the certification set, as\n"
        "many as the selected pairwise rule makes.  Fine family: thresholds 0.05 to 0.95\n"
        "in steps of 0.05.  No rule is certified.\\label{tab:listwise}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llrrrrrrrrr@{}}\n\\toprule\n"
        "& & & \\multicolumn{4}{c}{Declared family} & & & \\multicolumn{2}{c}{Fine family} \\\\\n"
        "\\cmidrule(lr){4-7}\\cmidrule(l){10-11}\n"
        "Benchmark & Score & Passes & $\\tau$ & Certify $m/h/k$ & $U$ & Confirm $m/h/k$ & AUROC & Matched & $\\tau$ & $U$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# System B (preregistered second system, trained verifier V1, RefCLEF)
# ---------------------------------------------------------------------------

SYSTEM_B_DIR = ROOT / "experiments/cs2_audit/results/system_b_20261016"
B_ARMS = {"RefCOCO+": "G1", "RefCOCOg": "G1", "RefCOCO": "G1", "RefCLEF": "G3"}


def _audit(arm: str, bench: str) -> dict:
    return json.loads((SYSTEM_B_DIR / "joined" / f"{arm}_{bench}_audit_summary.json").read_text(encoding="utf-8"))


def _mhk(d: dict) -> str:
    return f"{d['route_count']}/{d['help_count']}/{d['harm_count']}"


def _tau(policy: str) -> float:
    return float(policy.split("_")[2])


def _bold(cell: str) -> str:
    """Bold a table cell: text, a math value, or a math value followed by the pp unit."""

    if cell.endswith("$\\pp"):
        return "$\\mathbf{" + cell[1:-len("$\\pp")] + "}$\\pp"
    if cell.startswith("$") and cell.endswith("$"):
        return "$\\mathbf{" + cell[1:-1] + "}$"
    return f"\\textbf{{{cell}}}"


def _tau_tex(value: float, digits: int = 2) -> str:
    """A threshold for the text; the tie tolerance prints as 10^{-6}."""

    return "10^{-6}" if value < 1e-3 else f"{value:.{digits}f}"


def _verdict(cert: dict) -> str:
    if cert["route_count"] == 0:
        return "no switches"
    return "certified" if cert["cp_upper"] <= 0.05 else "not certified"


def _ci(pair) -> str:
    return "--" if pair is None else f"[{100 * pair[0]:.0f}, {100 * pair[1]:.0f}]"


def system_b_scores_table(numbers: dict) -> str:
    """Selected pairwise (primary) and listwise (secondary) rules of System B."""

    from .paper_numbers import cp_lower

    rows = []
    for bench, data in ordered(numbers, "System B"):
        arm = B_ARMS[bench]
        n_cert, n_conf = data["certify"]["selected"]["n"], data["confirm"]["selected"]["n"]
        for i, (score, summary) in enumerate((("pairwise", _audit(arm, bench)), ("listwise", _audit(arm + "-listwise", bench)))):
            c, f = summary["certify"], summary["confirm"]
            lower_c = cp_lower(c["harm_count"], c["route_count"])
            lower_f = cp_lower(f["harm_count"], f["route_count"])
            label = f"{bench} ({num(n_cert)}/{num(n_conf)})" if i == 0 else ""
            rows.append(
                f"{label} & {score} & {_tau(c['policy']):.2f} & {c['route_count']} & {c['help_count']} & {c['harm_count']} & "
                f"{pct(lower_c)} & {pct(c['cp_upper'])} & {f['route_count']} & {f['help_count']} & {f['harm_count']} & "
                f"{pct(f['conditional_harm'])} & {pct(lower_f)} & {_verdict(c)} \\\\"
            )
        rows.append("\\midrule")
    rows = rows[:-1]
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{System B: the selected rules of the preregistered pairwise score\n"
        "(primary) and listwise score (secondary) on the certification and\n"
        "confirmation sets.  In parentheses: certification/confirmation images.\n"
        "$m/h/k$: switches, helps, harms; $L$ and $U$: exact one-sided 95\\% lower and\n"
        "upper bounds on harm among switches.  Each benchmark's thresholds are\n"
        "percentiles of its own training switch scores (\\Cref{sec:system-b}).\\label{tab:sysb-scores}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}llrrrrrrrrrrrl@{}}\n\\toprule\n"
        "& & & \\multicolumn{5}{c}{Certification set} & \\multicolumn{5}{c}{Confirmation set} & \\\\\n"
        "\\cmidrule(lr){4-8}\\cmidrule(lr){9-13}\n"
        "Benchmark & Score & $\\tau$ & $m$ & $h$ & $k$ & $L$ & $U$ & $m$ & $h$ & $k$ & $k/m$ & $L$ & Verdict \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def system_b_diagnostics_table(numbers: dict, diagnostics: dict) -> str:
    """Help-versus-harm AUROC, decomposition at the selected rule, and certification power for System B."""

    rows = []
    for bench, data in ordered(numbers, "System B"):
        c, sel = data["certify"]["auroc"], data["select"]["auroc"]
        diag = diagnostics[bench]
        tau = diag["selected_threshold"]
        row = next(r for r in diag["decomposition_select"] if abs(r["threshold"] - tau) < 1e-12)
        always = data["certify"]["always_switch"]
        power = diag["power"]
        rows.append(
            f"{bench} & {c['auroc']:.2f} ({c['n_help']}/{c['n_harm']}) & {sel['auroc']:.2f} & "
            f"{pct(row['decisive_share'], 0)} & {pct(row['harmful_share_of_decisive'], 0)} & "
            f"{pct(always['harm_among_switches'])} ({num(always['m'])}) & {pct(always['upper'])} & "
            f"{power['power_at_observed_rate_and_size']:.2f} \\\\"
        )
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{System B diagnostics.  AUROC: help versus harm of the switch\n"
        "score (helpful/harmful opportunities in parentheses).  Decisive share and\n"
        "harmful share of decisive switches (\\Cref{eq:decomp}) are for the selected\n"
        "rule on the selection set.  Always switch: harm among all proposed switches\n"
        "on the certification set (their number in parentheses) and its descriptive\n"
        "bound $U$.  Power: exact probability that the selected rule is certified at\n"
        "its observed certification size and harm rate.\\label{tab:sysb-diag}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
        "& \\multicolumn{2}{c}{AUROC} & \\multicolumn{2}{c}{Selected rule (selection set)} & \\multicolumn{2}{c}{Always switch (cert.)} & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Benchmark & Certification & Selection & Decisive & Harmful of decisive & $k/m$ & $U$ & Power \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def gated_table(g1: dict, g3: dict) -> str:
    """Ungated and V1-gated rules on the same certification images, System B and (post hoc) System A."""

    from .records import load_registry

    rows = ["\\multicolumn{8}{@{}l}{\\emph{System B}} \\\\"]
    blocks = [("System B", bench, B_ARMS[bench], (g3 if bench == "RefCLEF" else g1)["system_b"][bench]) for bench in B_ARMS]
    for i, (system, bench, arm, analysis) in enumerate(blocks):
        ungated, gated = _audit(arm, bench), _audit(arm + "-gated", bench)
        v = analysis["verifier_select"]
        cert = analysis["verifier_certify"]
        for j, (label, summary) in enumerate((("ungated", ungated), ("V1-gated", gated))):
            policy = summary["certify"]["policy"]
            rule = f"$\\tau={_tau_tex(_tau(policy))}$" + (f", $\\nu={float(policy.rsplit('_', 1)[1]):.2f}$" if j else "")
            extra = (
                f"{v['harmful_vetoed'][0]}/{v['harmful_vetoed'][1]} {_ci(v['harm_recall_ci95'])} & "
                f"{v['non_harmful_vetoed'][0]}/{v['non_harmful_vetoed'][1]} {_ci(v['false_veto_ci95'])}"
                if j else
                f"\\multicolumn{{2}}{{c}}{{AUROC $s$ {cert['auroc_s_help_over_harm']:.2f}, $v$ {cert['auroc_v_harm_over_help']:.2f}}}"
            )
            rows.append(
                f"{bench if j == 0 else ''} & {label} {rule} & {_mhk(summary['certify'])} & {pct(summary['certify']['cp_upper'])} & "
                f"{_verdict(summary['certify'])} & {_mhk(summary['confirm'])} & {extra} \\\\"
            )
        rows.append("\\addlinespace")
    rows[-1] = "\\midrule"
    rows.append("\\multicolumn{8}{@{}l}{\\emph{System A, post hoc}} \\\\")
    for bench in load_registry()["systems"]["System A"]:
        r = g1["system_a_with_v1"][bench]
        c, f = r["gated"]["certify"], r["gated"]["confirm"]
        v, cert = r["verifier_select"], r["verifier_certify"]
        rows.append(
            f"{bench} & V1-gated $\\tau={_tau(r['gated_policy']):.1f}$, $\\nu={float(r['gated_policy'].rsplit('_', 1)[1]):.2f}$ & "
            f"{_mhk(c)} & {pct(c['cp_upper'])} & {_verdict(c)} & {_mhk(f)} & "
            f"{v['harmful_vetoed'][0]}/{v['harmful_vetoed'][1]} {_ci(v['harm_recall_ci95'])} & "
            f"{v['non_harmful_vetoed'][0]}/{v['non_harmful_vetoed'][1]} {_ci(v['false_veto_ci95'])} \\\\"
        )
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{The trained verifier V1.  Ungated: the selected pairwise rule;\n"
        "V1-gated: switch only if $s\\ge\\tau$ and V1's harm probability is at most\n"
        "$\\nu$, selected from the declared $(\\tau,\\nu)$ family before the\n"
        "certification labels were read.  $m/h/k$: switches, helps and harms on the\n"
        "same certification and confirmation images; $U$: exact 95\\% bound.  Caught\n"
        "and false vetoes: harmful and other switches with $s\\ge\\tau$ that V1 vetoes\n"
        "on the selection set, with exact two-sided 95\\% intervals in percent.  On the\n"
        "ungated rows: help-versus-harm AUROC of the switch score $s$ and of V1's harm\n"
        "probability $v$ (higher $v$ on harm) on the certification set.  System A\n"
        "rows are post hoc, because its labels were read before V1 was\n"
        "trained.\\label{tab:gated}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}llrrlrll@{}}\n\\toprule\n"
        "& & \\multicolumn{3}{c}{Certification set} & Confirmation & \\multicolumn{2}{c}{V1 on the selection set} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}\n"
        "Benchmark & Rule & $m/h/k$ & $U$ & Verdict & $m/h/k$ & Harms caught & False vetoes \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def transfer_table() -> str:
    """G3b: G1's frozen RefCOCO rules on the RefCLEF confirmation set, next to RefCLEF's own rules."""

    transfer = json.loads((SYSTEM_B_DIR / "joined" / "G3b_RefCLEF_transfer_summary.json").read_text(encoding="utf-8"))
    rows = []
    for score, own_arm in (("pairwise", "G3"), ("listwise", "G3-listwise")):
        t = transfer[score]
        own = _audit(own_arm, "RefCLEF")["confirm"]
        for label, d in ((f"RefCOCO rule (transfer), $\\tau={_tau(t['policy']):.2f}$", t), (f"RefCLEF rule (in-source), $\\tau={_tau(own['policy']):.2f}$", own)):
            rows.append(
                f"{score if 'transfer' in label else ''} & {label} & {d['route_count']} & {d['help_count']} & {d['harm_count']} & "
                f"{pct(d['conditional_harm'])} & {pct(d['cp_upper'])} & {pp(100 * d['net_accuracy_gain'])} \\\\"
            )
    gated = _audit("G3-gated", "RefCLEF")["confirm"]
    rows.append(
        f"V1-gated & RefCLEF rule (in-source), $\\tau={_tau(gated['policy']):.2f}$, $\\nu={float(gated['policy'].rsplit('_', 1)[1]):.2f}$ & "
        f"{gated['route_count']} & {gated['help_count']} & {gated['harm_count']} & {pct(gated['conditional_harm'])} & "
        f"{pct(gated['cp_upper'])} & {pp(100 * gated['net_accuracy_gain'])} \\\\"
    )
    return (
        "\\begin{table}[H]\n\\begin{adjustwidth}{-\\extralength}{0cm}\n"
        "\\caption{Transfer test on the RefCLEF confirmation set (2,999 images).  The\n"
        "RefCOCO rules were frozen on RefCOCO's selection set and their RefCLEF\n"
        "switches were fixed and committed before any RefCLEF label was read; the\n"
        "in-source rules were selected on RefCLEF's own selection set.  $m/h/k$:\n"
        "switches, helps and harms; $U$: exact one-sided 95\\% upper bound; accuracy\n"
        "change over all 2,999 images.\\label{tab:transfer}}\n"
        "\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n"
        "Score & Rule & $m$ & $h$ & $k$ & $k/m$ & $U$ & Accuracy change \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}\n"
    )


def system_b_trace_table(diagnostics: dict) -> str:
    """All declared System B thresholds (pairwise) on the selection sets."""

    rows = []
    for bench in load_registry()["systems"]["System B"]:
        trace = sorted(diagnostics[bench]["selection_trace"], key=lambda r: _tau(r["policy"]))
        for i, r in enumerate(trace):
            cells = [f"${_tau_tex(_tau(r['policy']))}$", num(r["route_count"]), str(r["help_count"]), str(r["harm_count"]), pp(100 * r["net_accuracy_gain"])]
            if r["selected"]:
                cells = [_bold(c) for c in cells]
            rows.append(f"{bench if i == 0 else ''} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    return (
        "\\begin{table}[H]\n"
        "\\caption{System B: all declared pairwise thresholds on the selection sets\n"
        "(the smallest is the tie tolerance).  Bold: the selected\n"
        "rule.\\label{tab:sysb-trace}}\n"
        "\\centering\n\\small\n\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "Benchmark & $\\tau$ & Switches & Helps & Harms & Accuracy change \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numbers", type=Path, default=ROOT / "experiments/cs2_audit/results/paper_numbers_20261016.json")
    parser.add_argument("--revision", type=Path, default=ROOT / "experiments/cs2_audit/results/revision_cpu_audit_20261016.json")
    parser.add_argument("--diagnosis-dir", type=Path, default=ROOT / "experiments/cs2_audit/results/failure_diagnosis_20261016")
    parser.add_argument("--listwise", type=Path, default=ROOT / "experiments/cs2_audit/results/listwise_check_20261016.json")
    parser.add_argument("--system", default="System A")
    parser.add_argument("--out-dir", type=Path, default=FIGS)
    args = parser.parse_args()

    numbers = json.loads(args.numbers.read_text(encoding="utf-8"))
    revision = json.loads(args.revision.read_text(encoding="utf-8"))
    outputs = {
        "mdpi_tab_main.tex": main_table(numbers, args.system),
        "mdpi_tab_confirm.tex": confirm_table(numbers, args.system),
        "mdpi_tab_auroc.tex": auroc_table(numbers, args.system),
        "mdpi_tab_selection_trace.tex": selection_trace_table(args.system),
        "mdpi_tab_matched_random.tex": random_table(numbers, args.system),
        "mdpi_tab_continuous_iou.tex": continuous_iou_table(revision),
        "mdpi_tab_procedures.tex": procedures_table(args.diagnosis_dir),
        "mdpi_tab_listwise.tex": listwise_table(json.loads(args.listwise.read_text(encoding="utf-8"))),
    }
    if "System B" in numbers:
        diagnostics = json.loads((SYSTEM_B_DIR / "diagnostics" / "system_b_diagnostics.json").read_text(encoding="utf-8"))
        g1 = json.loads((SYSTEM_B_DIR / "g1_secondary_analysis.json").read_text(encoding="utf-8"))
        g3 = json.loads((SYSTEM_B_DIR / "g3_secondary_analysis.json").read_text(encoding="utf-8"))
        outputs.update({
            "mdpi_tab_sysb_main.tex": main_table(numbers, "System B", label="tab:sysb-main", intro="System B. ", tau_digits=2),
            "mdpi_tab_sysb_scores.tex": system_b_scores_table(numbers),
            "mdpi_tab_sysb_diag.tex": system_b_diagnostics_table(numbers, diagnostics),
            "mdpi_tab_gated.tex": gated_table(g1, g3),
            "mdpi_tab_transfer.tex": transfer_table(),
            "mdpi_tab_sysb_trace.tex": system_b_trace_table(diagnostics),
        })
    for name, text in outputs.items():
        (args.out_dir / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
