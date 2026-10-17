"""Tests for the manuscript/artifact count release gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .validate_authoritative_counts import validate_paper_text, validate_summary


class AuthoritativeCountTests(unittest.TestCase):
    def test_summary_matches_exact_four_way_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit_summary.json"
            path.write_text(
                json.dumps({"rows": {"train": 2, "select": 3, "certify": 4, "confirm": 5}}),
                encoding="utf-8",
            )
            report = validate_summary(
                path,
                {"train": 2, "select": 3, "certify": 4, "confirm": 5},
            )
            self.assertEqual(report["rows"]["confirm"], 5)

    def test_paper_rejects_stale_count_and_accepts_authoritative_sentences(self) -> None:
        benchmarks = {
            "RefCOCO+": {"paper_sentence": "train/select/certify/confirm counts $1/2/3/4$"},
            "RefCOCOg": {"paper_sentence": "train/select/certify/confirm counts $5/6/7/8$"},
            "RefCOCO": {"paper_sentence": "train/select/certify/confirm counts $9/10/11/12$"},
        }
        text = " ".join(spec["paper_sentence"] for spec in benchmarks.values())
        validate_paper_text(text, benchmarks, ["317/1451/1430/1500"])
        with self.assertRaises(ValueError):
            validate_paper_text(text + " 317/1451/1430/1500", benchmarks, ["317/1451/1430/1500"])

    def test_paper_accepts_reader_facing_count_sentence(self) -> None:
        benchmarks = {
            "RefCOCO+": {
                "paper_sentence": "train/select/certify/confirm counts $1/2/3/4$",
                "counts": {"train": 1, "select": 2, "certify": 3, "confirm": 4},
            }
        }
        validate_paper_text(
            "The resulting splits contain 1 training, 2 selection, 3 certification, "
            "and 4 confirmation images.",
            benchmarks,
            [],
        )


if __name__ == "__main__":
    unittest.main()
