"""Unit tests for the deterministic LaTeX report generator."""

import json
import tempfile
import unittest
from pathlib import Path

from .report import build_latex_table, format_row, load_methods_summary


class ReportTests(unittest.TestCase):
    def test_format_row_handles_missing_route(self) -> None:
        row = format_row(
            "No-switch",
            {
                "n": 128,
                "route_count": 0,
                "help_count": 0,
                "harm_count": 0,
                "cp_upper": None,
                "net_accuracy_gain": 0.0,
                "status": "zero-route; conditional harm undefined",
            },
        )
        self.assertIn("No-switch", row)
        self.assertIn("N/A", row)
        self.assertTrue(row.endswith(r"\\"))

    def test_format_row_computes_k_over_m(self) -> None:
        row = format_row(
            "CS$^2$",
            {
                "n": 128,
                "route_count": 20,
                "help_count": 5,
                "harm_count": 1,
                "cp_upper": 0.18,
                "net_accuracy_gain": 0.03,
                "status": "certified",
            },
        )
        cells = row.rstrip(r" \\").split(" & ")
        self.assertEqual(cells[5], f"{1 / 20:.3f}")

    def test_build_latex_table_is_deterministic(self) -> None:
        rows = [
            ("No-switch", {"n": 10, "route_count": 0, "help_count": 0, "harm_count": 0, "cp_upper": None, "net_accuracy_gain": 0.0, "status": "reference"}),
            ("Oracle", {"n": 10, "route_count": 4, "help_count": 4, "harm_count": 0, "cp_upper": None, "net_accuracy_gain": 0.4, "status": "oracle"}),
        ]
        first = build_latex_table(rows, caption="c", label="tab:x")
        second = build_latex_table(rows, caption="c", label="tab:x")
        self.assertEqual(first, second)
        self.assertIn(r"\begin{table*}", first)
        self.assertIn(r"\end{table*}", first)
        self.assertIn("No-switch", first)
        self.assertIn("Oracle", first)

    def test_load_methods_summary_round_trips(self) -> None:
        data = {"No-switch": {"n": 5, "route_count": 0, "help_count": 0, "harm_count": 0, "cp_upper": None, "net_accuracy_gain": 0.0, "status": "reference"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "methods_summary.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            rows = load_methods_summary(path)
        self.assertEqual(rows, [("No-switch", data["No-switch"])])


if __name__ == "__main__":
    unittest.main()
