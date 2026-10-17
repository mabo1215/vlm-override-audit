from __future__ import annotations

import unittest

import pandas as pd

from experiments.cs2_audit import listwise_check as lc


def entry(boxes, pairwise, listwise):
    return {
        "candidate_sources": ["restricted"] + ["detector_expr"] * (len(boxes) - 1),
        "candidate_boxes": boxes,
        "candidate_scores": [{"pairwise_score": p, "listwise_posterior": q} for p, q in zip(pairwise, listwise)],
    }


class SwitchChoiceTests(unittest.TestCase):
    def test_ties_and_tiny_margins_keep_the_default(self) -> None:
        self.assertEqual(lc.switch_choice([0.2, 0.2], 0), (0, 0.0))
        self.assertEqual(lc.switch_choice([0.2, 0.2 + 5e-7], 0), (0, 0.0))
        best, margin = lc.switch_choice([0.1, 0.7, 0.2], 0)
        self.assertEqual(best, 1)
        self.assertAlmostEqual(margin, 0.6)


class RebuildTests(unittest.TestCase):
    def setUp(self) -> None:
        gt = (0.0, 0.0, 0.5, 0.5)
        self.truth = {"1": gt, "2": gt}
        good, bad = [0.0, 0.0, 0.5, 0.5], [0.6, 0.6, 1.0, 1.0]
        # Record 1: pairwise prefers the bad box, listwise the good default.
        # Record 2: both prefer the good box over a bad default.
        self.scores = {
            "1": entry([good, bad], [-0.4, 0.4], [0.9, 0.1]),
            "2": entry([bad, good], [-0.5, 0.5], [0.2, 0.8]),
        }
        self.records = pd.DataFrame(
            {
                "split": ["certify", "certify"],
                "image_id": [10, 20],
                "expression_id": ["1", "2"],
                "score": [0.8, 1.0],
                "baseline_iou": [1.0, 0.0],
                "challenger_iou": [0.0, 1.0],
            }
        )

    def test_rebuilds_both_scores_and_checks_the_registered_pairwise_records(self) -> None:
        frames, check = lc.rebuild(self.records, self.scores, self.truth)
        self.assertEqual(check["pairwise_mismatches_vs_registered"], {"score": 0, "baseline_iou": 0, "challenger_iou": 0})
        listwise = frames["listwise"].set_index("expression_id")
        self.assertEqual(listwise.loc["1", "score"], 0.0)
        self.assertEqual(listwise.loc["1", "challenger_iou"], 1.0)
        self.assertAlmostEqual(listwise.loc["2", "score"], 0.6)
        self.assertEqual(check["both_propose_a_switch"], 1)
        self.assertEqual(check["same_alternative_when_both_propose"], 1)
        self.assertEqual(check["mean_pairwise_passes"], 2.0)
        self.assertEqual(check["mean_listwise_passes"], 2.0)

    def test_a_record_mismatch_is_counted(self) -> None:
        records = self.records.copy()
        records.loc[0, "score"] = 0.5
        _, check = lc.rebuild(records, self.scores, self.truth)
        self.assertEqual(check["pairwise_mismatches_vs_registered"]["score"], 1)


class HindsightTests(unittest.TestCase):
    def test_reports_lowest_upper_bound_over_thresholds(self) -> None:
        part = pd.DataFrame(
            {
                "split": ["certify"] * 60,
                "image_id": range(60),
                "expression_id": [str(i) for i in range(60)],
                "score": [1.0] * 59 + [0.5],
                "baseline_iou": [0.0] * 59 + [1.0],
                "challenger_iou": [1.0] * 59 + [0.0],
            }
        )
        result = lc.hindsight(part)
        self.assertEqual(result["lowest_upper"]["m"], 59)
        self.assertEqual(result["lowest_upper"]["k"], 0)
        self.assertEqual(result["thresholds_with_upper_at_most_alpha"], 1)



class ReleaseInputTests(unittest.TestCase):
    def test_candidate_ious_replace_ground_truth(self) -> None:
        import gzip
        import json
        import tempfile
        from pathlib import Path

        import pandas as pd

        from experiments.cs2_audit import listwise_check as lc

        row = {"ref_id": "7", "candidate_sources": ["restricted", "detector_expr"],
               "candidate_boxes": [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
               "candidate_pairwise_scores": [0.0, 0.6], "candidate_listwise_posteriors": [0.7, 0.3],
               "candidate_ious": [0.9, 0.2]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x_candidates.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            scores = lc.load_release(path)
        records = pd.DataFrame([{"split": "certify", "image_id": 1, "expression_id": "7", "score": 0.6,
                                 "baseline_iou": 0.9, "challenger_iou": 0.2}])
        frames, check = lc.rebuild(records, scores, None)
        self.assertEqual(check["pairwise_mismatches_vs_registered"], {"score": 0, "baseline_iou": 0, "challenger_iou": 0})
        self.assertEqual(float(frames["listwise"]["score"].iloc[0]), 0.0)  # the listwise score keeps the default


if __name__ == "__main__":
    unittest.main()
