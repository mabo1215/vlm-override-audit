"""Deterministic report generator: raw audit JSON -> paper-ready LaTeX table.

Matches the main-table schema requested in docs/revision_suggestions.tex
P0-7: Method, n, m, Help, Harm, k/m, U_.05, Net, Status.  Every number in the
emitted table traces back to one ``methods_summary.json`` produced by
``run_experiment.py --with-baselines``; nothing here is hand-typed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"
    return str(value)


def format_row(method_name: str, result: Mapping[str, object]) -> str:
    n = result.get("n")
    m = result.get("route_count")
    help_count = result.get("help_count")
    harm_count = result.get("harm_count")
    k_over_m = None
    if m:
        k_over_m = harm_count / m
    cells = [
        str(method_name),
        _fmt(n, 0),
        _fmt(m, 0),
        _fmt(help_count),
        _fmt(harm_count),
        _fmt(k_over_m),
        _fmt(result.get("cp_upper")),
        _fmt(result.get("net_accuracy_gain")),
        str(result.get("status", "")),
    ]
    return " & ".join(cells) + r" \\"


def build_latex_table(
    rows: Sequence[tuple[str, Mapping[str, object]]], *, caption: str, label: str
) -> str:
    header = (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{caption}}}" "\n"
        rf"\label{{{label}}}" "\n"
        r"\begin{tabular}{lrrrrrrrl}" "\n"
        r"\toprule" "\n"
        r"Method & $n$ & $m$ & Help & Harm & $k/m$ & $U_{.05}$ & Net & Status \\" "\n"
        r"\midrule"
    )
    body = "\n".join(format_row(name, result) for name, result in rows)
    footer = r"\bottomrule" "\n" r"\end{tabular}" "\n" r"\end{table*}"
    return "\n".join([header, body, footer, ""])


def load_methods_summary(path: Path) -> list[tuple[str, Mapping[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must contain a JSON object mapping method name to result"
        )
    return list(data.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--caption",
        default="Selective-switching audit results (exact counts from the archived manifest).",
    )
    parser.add_argument("--label", default="tab:cs2_main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_methods_summary(args.summary_json)
    table = build_latex_table(rows, caption=args.caption, label=args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(table)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
