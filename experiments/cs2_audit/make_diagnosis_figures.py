"""Render the paper figures from the failure-diagnosis outputs.

Reads the CSV/JSON files written by ``diagnose_failure.py`` and writes vector
PDFs.  Benchmarks keep one fixed colour and marker everywhere; methods are
distinguished by line style.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _font in Path("/usr/share/fonts/opentype/urw-base35").glob("NimbusRoman-*.otf"):
    font_manager.fontManager.addfont(str(_font))

BENCHMARKS = ("RefCOCO+", "RefCOCOg", "RefCOCO")
COLOURS = {"RefCOCO+": "#2a78d6", "RefCOCOg": "#eb6834", "RefCOCO": "#1baf7a", "RefCLEF": "#8a3fb5"}
MARKERS = {"RefCOCO+": "o", "RefCOCOg": "s", "RefCOCO": "^", "RefCLEF": "D"}
DECLARED = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
    }
)


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _slug(name: str) -> str:
    return name.replace("+", "plus")


def risk_coverage_figure(diag_dir: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    top = 0.0
    for index, name in enumerate(BENCHMARKS):
        curve = pd.read_csv(diag_dir / f"risk_coverage_{_slug(name)}.csv")
        curve = curve[curve["routed"] >= 15].sort_values("route_rate")
        top = max(top, float(100 * curve["cp_upper"].max()))
        x = 100 * curve["route_rate"]
        ax.plot(x, 100 * curve["harm_among_routed"], color=COLOURS[name], label=name)
        ax.plot(
            x,
            100 * curve["cp_upper"],
            color=COLOURS[name],
            linestyle=(0, (2, 1.5)),
            linewidth=0.9,
        )
        for t in DECLARED:
            row = curve[curve["threshold"] >= t - 1e-12].sort_values("threshold").head(1)
            if row.empty:
                continue
            ax.plot(
                100 * row["route_rate"],
                100 * row["harm_among_routed"],
                marker=MARKERS[name],
                markersize=4.2,
                color=COLOURS[name],
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
            )
            if index == 0 and t in (DECLARED[0], DECLARED[-1]):
                ax.annotate(
                    rf"$\tau={t:.1f}$",
                    (float(100 * row["route_rate"].iloc[0]), float(100 * row["harm_among_routed"].iloc[0])),
                    xytext=(12, 30) if t == DECLARED[0] else (7, 40),
                    textcoords="data",
                    fontsize=6,
                    color=COLOURS[name],
                    arrowprops={"arrowstyle": "-", "color": COLOURS[name], "linewidth": 0.6},
                )
    ax.axhline(5.0, color=INK, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(37.5, 4.2, r"target $\alpha=5\%$", ha="right", va="top", color=INK, fontsize=6.5)
    ax.set_xlim(0, 38)
    ax.set_ylim(0, max(45.0, 5.0 * np.ceil(top / 5.0)))
    ax.set_xlabel("images switched (% of selection set)")
    ax.set_ylabel("harm among switched (%)")
    handles, labels = ax.get_legend_handles_labels()
    solid = plt.Line2D([], [], color=MUTED, label="observed")
    dashed = plt.Line2D([], [], color=MUTED, linestyle=(0, (2, 1.5)), linewidth=0.9, label="95% upper bound")
    ax.legend(
        handles + [solid, dashed],
        labels + ["observed", "95% upper bound"],
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.0,
    )
    _style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, metadata={"CreationDate": None})  # reproducible bytes
    plt.close(fig)


def risk_coverage_system_b(diag_dir: Path, out: Path) -> None:
    """System B's risk--coverage on the selection sets; markers are its declared thresholds."""

    diagnostics = json.loads((diag_dir / "system_b_diagnostics.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    for name in (*BENCHMARKS, "RefCLEF"):
        curve = pd.read_csv(diag_dir / f"risk_coverage_{_slug(name)}.csv")
        curve = curve[curve["routed"] >= 15].sort_values("route_rate")
        ax.plot(100 * curve["route_rate"], 100 * curve["harm_among_routed"], color=COLOURS[name], label=name)
        for t in diagnostics[name]["family"]:
            row = curve[curve["threshold"] >= t - 1e-12].sort_values("threshold").head(1)
            if row.empty:
                continue
            selected = abs(t - diagnostics[name]["selected_threshold"]) < 1e-12
            ax.plot(
                100 * row["route_rate"], 100 * row["harm_among_routed"], marker=MARKERS[name],
                markersize=5.2 if selected else 3.8, color=COLOURS[name],
                markeredgecolor=INK if selected else "white", markeredgewidth=0.8 if selected else 0.6, linestyle="none",
            )
    ax.axhline(5.0, color=INK, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(54, 6.0, r"target $\alpha=5\%$", ha="right", va="bottom", color=INK, fontsize=6.5)
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 80)
    ax.set_xlabel("images switched (% of selection set)")
    ax.set_ylabel("harm among switched (%)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, ncol=2, handlelength=1.8, columnspacing=1.0)
    _style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, metadata={"CreationDate": None})  # reproducible bytes
    plt.close(fig)


def diagnosis_figure(diag_dir: Path, out: Path, false_veto: float, verifier: dict | None = None) -> None:
    summary = json.loads((diag_dir / "failure_diagnosis.json").read_text())
    power = pd.read_csv(diag_dir / "certification_power.csv")
    controlled = pd.read_csv(diag_dir / "controlled_summary.csv")
    controlled = controlled[controlled["false_veto"] == false_veto]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.05))

    # (a) certification power versus routed sample size.
    ax = axes[0]
    rates = (0.0, 0.01, 0.02, 0.03, 0.04)
    shades = ("#0d366b", "#184f95", "#2a78d6", "#6aa3e6", "#a9c9f1")
    for rate, shade in zip(rates, shades):
        part = power[power["true_harm_rate"] == rate]
        ax.plot(part["routed"], part["power"], color=shade, label=f"{100 * rate:.0f}%")
    for name in BENCHMARKS:
        obs = summary["benchmarks"][name]["observed_power"]
        ax.plot(
            obs["routed"],
            obs["power_at_observed_rate_and_size"],
            marker=MARKERS[name],
            color=COLOURS[name],
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            linestyle="none",
        )
    ax.axvline(59, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(10, 10000)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("switched images in certification set")
    ax.set_ylabel("probability of certification")
    ax.set_title("(a) data needed", loc="left")
    rate_legend = ax.legend(
        title="true harm\nrate",
        title_fontsize=6,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.0),
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    ax.add_artist(rate_legend)
    selected_handles = [
        plt.Line2D([], [], color=COLOURS[n], marker=MARKERS[n], linestyle="none", markersize=4, label=n)
        for n in BENCHMARKS
    ]
    ax.legend(
        handles=selected_handles,
        title="selected rules",
        title_fontsize=6,
        frameon=False,
        loc="center right",
        handletextpad=0.3,
        labelspacing=0.25,
    )
    _style(ax)

    # (b) probability of certification versus verifier recall.
    ax = axes[1]
    for name in BENCHMARKS:
        part = controlled[controlled["benchmark"] == name]
        for method, style in (("cs2", "-"), ("ltt_bonferroni", (0, (3, 1.5)))):
            rows = part[part["method"] == method].sort_values("recall")
            ax.plot(
                100 * rows["recall"],
                rows["deploy_rate"],
                color=COLOURS[name],
                linestyle=style,
                linewidth=1.3 if method == "cs2" else 1.0,
                marker=MARKERS[name] if method == "cs2" else None,
                markersize=3.2,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
    if verifier:
        # V1's measured recall on System A's selection sets (exact 95% intervals),
        # drawn at the height of its observed outcome (not certified).
        for offset, name in zip((0.03, 0.08, 0.13), BENCHMARKS):
            point = verifier[name]["verifier_select"]
            low, high = point["harm_recall_ci95"]
            x = 100 * point["harm_recall"]
            ax.errorbar(
                x, offset, xerr=[[x - 100 * low], [100 * high - x]], fmt=MARKERS[name], color=COLOURS[name],
                markerfacecolor="white", markersize=3.6, elinewidth=0.8, capsize=1.5,
            )
        ax.text(3, 0.20, "trained verifier V1\n(measured)", fontsize=5.8, color=INK, ha="left", va="bottom")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("harmful switches caught (%)")
    ax.set_ylabel("probability of certification")
    ax.set_title("(b) verifier quality needed", loc="left")
    bench_handles = [
        plt.Line2D([], [], color=COLOURS[n], marker=MARKERS[n], markersize=3.2, label=n)
        for n in BENCHMARKS
    ]
    method_handles = [
        plt.Line2D([], [], color=MUTED, label="split audit"),
        plt.Line2D([], [], color=MUTED, linestyle=(0, (3, 1.5)), linewidth=1.0, label="LTT, pooled"),
    ]
    ax.legend(handles=bench_handles + method_handles, frameon=False, loc="upper left", handlelength=1.6)
    _style(ax)

    # (c) how often each procedure deploys a truly unsafe policy.
    ax = axes[2]
    for name in BENCHMARKS:
        rows = controlled[
            (controlled["benchmark"] == name) & (controlled["method"] == "empirical_screen")
        ].sort_values("recall")
        ax.plot(
            100 * rows["recall"],
            100 * rows["unsafe_deploy_rate"],
            color=COLOURS[name],
            marker=MARKERS[name],
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
    for method, style in (("cs2", "-"), ("ltt_bonferroni", (0, (3, 1.5)))):
        worst = (
            controlled[controlled["method"] == method]
            .groupby("recall")["unsafe_deploy_rate"]
            .max()
            .sort_index()
        )
        ax.plot(100 * worst.index, 100 * worst.values, color=INK, linestyle=style, linewidth=1.1)
    ax.axhline(5.0, color=MUTED, linewidth=0.7, linestyle=":")
    ax.text(1.5, 6.5, r"error budget $\delta=5\%$", ha="left", fontsize=6, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 80)
    ax.set_xlabel("harmful switches caught (%)")
    ax.set_ylabel("unsafe deployments (% of runs)")
    ax.set_title("(c) deploying an unsafe switch", loc="left")
    proc_handles = [
        plt.Line2D([], [], color=MUTED, marker="o", markersize=3.2, label="no bound"),
        plt.Line2D([], [], color=INK, label="split audit"),
        plt.Line2D([], [], color=INK, linestyle=(0, (3, 1.5)), linewidth=1.1, label="LTT"),
    ]
    ax.legend(handles=proc_handles, frameon=False, loc="upper left", handlelength=1.6)
    _style(ax)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(out, metadata={"CreationDate": None})  # reproducible bytes
    plt.close(fig)


def _pct(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}\\%"


def _table_specs(diag_dir: Path) -> list[dict]:
    """Build the diagnosis tables once, independent of the page layout."""

    summary = json.loads((diag_dir / "failure_diagnosis.json").read_text())
    controlled = pd.read_csv(diag_dir / "controlled_summary.csv")
    specs: list[dict] = []

    rows: list[str] = []
    for b_index, name in enumerate(BENCHMARKS):
        decomposition = [r for r in summary["benchmarks"][name]["decomposition"] if r["split"] == "select"]
        for r_index, row in enumerate(decomposition):
            label = name if r_index == 0 else ""
            rows.append(
                f"{label} & {row['threshold']:.1f} & {row['routed']} & {row['help']}/{row['harm']} & "
                f"{_pct(row['decisive_share'])} & {_pct(row['harmful_share_of_decisive'])} & "
                f"{_pct(row['harm_among_routed'], 1)} \\\\"
            )
        if b_index < len(BENCHMARKS) - 1:
            rows.append("\\midrule")
    specs.append(
        {
            "label": "tab:decomp-full",
            "caption": [
                "Decomposition of harm among switches on the selection sets for all six",
                "declared thresholds.  Decisive switches change correctness; the harm rate is the",
                "decisive share times the harmful share of decisive switches.",
            ],
            "tabcolsep": "3pt",
            "colspec": "@{}lrrrrrr@{}",
            "header": [
                "& & & & Decisive & Harmful & Harm \\\\",
                "Bench. & $\\tau$ & $m$ & $h/k$ & share & of decisive & rate \\\\",
            ],
            "rows": rows,
            "wide": False,
        }
    )

    rows = []
    for name in BENCHMARKS:
        req = summary["benchmarks"][name]["min_recall_for_a_truly_safe_policy"]
        rows.append(f"{name} & {_pct(req['0.0'], 1)} & {_pct(req['0.1'], 1)} & {_pct(req['0.3'], 1)} \\\\")
    specs.append(
        {
            "label": "tab:min-recall",
            "caption": [
                "Smallest verifier recall (share of harmful switches vetoed) at which at",
                "least one declared threshold is truly safe, for three false-veto rates $f$.",
            ],
            "tabcolsep": None,
            "colspec": "@{}lrrr@{}",
            "header": ["Benchmark & $f=0\\%$ & $f=10\\%$ & $f=30\\%$ \\\\"],
            "rows": rows,
            "wide": False,
        }
    )

    methods = (
        ("empirical_screen", "No bound"),
        ("same_split", "Same data, no correction"),
        ("cs2", "Split audit"),
        ("ltt_bonferroni", "LTT, pooled"),
    )
    recalls = (0.8, 0.9, 0.95, 1.0)
    rows = []
    for f_index, false_veto in enumerate((0.0, 0.1, 0.3)):
        for m_index, (method, label) in enumerate(methods):
            cells = []
            for recall in recalls:
                part = controlled[
                    (controlled["false_veto"] == false_veto)
                    & (controlled["method"] == method)
                    & (controlled["recall"] == recall)
                ].set_index("benchmark")
                deploy = "/".join(f"{100 * part.loc[n, 'deploy_rate']:.0f}" for n in BENCHMARKS)
                unsafe = "/".join(
                    f"{100 * part.loc[n, 'unsafe_deploy_rate']:.0f}" for n in BENCHMARKS
                )
                cells.append(f"{deploy} ({unsafe})")
            prefix = _pct(false_veto) if m_index == 0 else ""
            rows.append(f"{prefix} & {label} & " + " & ".join(cells) + " \\\\")
        if f_index < 2:
            rows.append("\\midrule")
    specs.append(
        {
            "label": "tab:controlled",
            "caption": [
                f"Controlled verifier study ({summary['controlled_repeats']:,} runs per cell).  Each cell gives the share of",
                "runs that deploy a rule, followed in parentheses by the share that deploy a truly unsafe",
                "rule (true harm among switches above 5\\%), for RefCOCO+ / RefCOCOg / RefCOCO.",
            ],
            "tabcolsep": "3pt",
            "colspec": "@{}llllll@{}",
            "header": [
                "$f$ & Procedure & " + " & ".join(f"recall {_pct(r)}" for r in recalls) + " \\\\"
            ],
            "rows": rows,
            "wide": True,
        }
    )
    return specs


def _render_table(spec: dict, layout: str) -> list[str]:
    """Render one table for the two-column conference or the MDPI journal layout."""

    caption = list(spec["caption"])
    body = [f"\\begin{{tabular}}{{{spec['colspec']}}}", "\\toprule", *spec["header"], "\\midrule"]
    body += [*spec["rows"], "\\bottomrule", "\\end{tabular}"]
    if layout == "wacv":
        env = "table*" if spec["wide"] else "table"
        lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small"]
        lines += ["\\caption{" + caption[0], *caption[1:]]
        lines[-1] += "}"
        lines.append(f"\\label{{{spec['label']}}}")
        if spec["tabcolsep"]:
            lines.append(f"\\setlength{{\\tabcolsep}}{{{spec['tabcolsep']}}}")
        return lines + body + [f"\\end{{{env}}}", ""]
    if layout == "mdpi":
        lines = ["\\begin{table}[H]"]
        if spec["wide"]:
            lines.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
        lines += ["\\caption{" + caption[0], *caption[1:]]
        lines[-1] += f"\\label{{{spec['label']}}}}}"
        lines += ["\\centering", "\\small"]
        if spec["tabcolsep"]:
            lines.append(f"\\setlength{{\\tabcolsep}}{{{spec['tabcolsep']}}}")
        lines += body
        if spec["wide"]:
            lines.append("\\end{adjustwidth}")
        return lines + ["\\end{table}", ""]
    raise ValueError(f"unknown table layout: {layout}")


def write_tables(diag_dir: Path, out: Path, layout: str = "wacv") -> None:
    """Write the diagnosis tables as one LaTeX include file for one page layout."""

    lines: list[str] = []
    for spec in _table_specs(diag_dir):
        lines += _render_table(spec, layout)
    out.write_text("\n".join(lines))


def write_table_files(diag_dir: Path, out_dir: Path, layout: str, prefix: str) -> list[Path]:
    """Write each diagnosis table to its own include file, named after its label."""

    written = []
    for spec in _table_specs(diag_dir):
        path = out_dir / f"{prefix}{spec['label'].removeprefix('tab:')}.tex"
        path.write_text("\n".join(_render_table(spec, layout)))
        written.append(path)
    return written


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--diagnosis-dir", type=Path, default=here / "results" / "failure_diagnosis_20261016"
    )
    parser.add_argument("--figure-dir", type=Path, default=here.parents[1] / "paper" / "figs")
    parser.add_argument("--false-veto", type=float, default=0.1)
    parser.add_argument(
        "--system-b-dir", type=Path, default=here / "results" / "system_b_20261016",
        help="System B results: diagnostics/ and the V1 analysis (skipped if absent)",
    )
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    verifier = None
    analysis = args.system_b_dir / "g1_secondary_analysis.json"
    if analysis.exists():
        verifier = json.loads(analysis.read_text())["system_a_with_v1"]
    if (args.system_b_dir / "diagnostics" / "system_b_diagnostics.json").exists():
        risk_coverage_system_b(args.system_b_dir / "diagnostics", args.figure_dir / "risk_coverage_system_b.pdf")
    risk_coverage_figure(args.diagnosis_dir, args.figure_dir / "risk_coverage.pdf")
    diagnosis_figure(args.diagnosis_dir, args.figure_dir / "diagnosis.pdf", args.false_veto, verifier)
    write_tables(args.diagnosis_dir, args.figure_dir / "diagnosis_tables.tex", layout="wacv")
    write_table_files(args.diagnosis_dir, args.figure_dir, layout="mdpi", prefix="mdpi_tab_")


if __name__ == "__main__":
    main()
