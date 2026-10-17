"""Unit tests for the layout-specific rendering of the diagnosis tables."""

import unittest

from .make_diagnosis_figures import _render_table


def _spec(wide: bool) -> dict:
    return {
        "label": "tab:demo",
        "caption": ["First caption line", "second caption line."],
        "tabcolsep": "3pt",
        "colspec": "@{}lr@{}",
        "header": ["Name & Value \\\\"],
        "rows": ["a & 1 \\\\"],
        "wide": wide,
    }


class RenderTableTest(unittest.TestCase):
    def test_conference_layout_uses_starred_float_for_wide_tables(self) -> None:
        lines = _render_table(_spec(wide=True), "wacv")
        self.assertEqual(lines[0], "\\begin{table*}[t]")
        self.assertIn("second caption line.}", lines)
        self.assertIn("\\label{tab:demo}", lines)
        self.assertEqual(lines[-2], "\\end{table*}")

    def test_journal_layout_keeps_label_in_caption_and_widens_tables(self) -> None:
        lines = _render_table(_spec(wide=True), "mdpi")
        self.assertEqual(lines[:2], ["\\begin{table}[H]", "\\begin{adjustwidth}{-\\extralength}{0cm}"])
        self.assertIn("second caption line.\\label{tab:demo}}", lines)
        self.assertEqual(lines[-3:], ["\\end{adjustwidth}", "\\end{table}", ""])

    def test_journal_layout_keeps_narrow_tables_in_the_text_column(self) -> None:
        lines = _render_table(_spec(wide=False), "mdpi")
        self.assertNotIn("\\begin{adjustwidth}{-\\extralength}{0cm}", lines)
        self.assertIn("a & 1 \\\\", lines)

    def test_unknown_layout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _render_table(_spec(wide=False), "poster")


if __name__ == "__main__":
    unittest.main()
