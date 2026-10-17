"""Unit tests for the failed-certification diagnostics."""

import unittest

import numpy as np
import pandas as pd

from .diagnose_failure import (
    certification_power,
    controlled_study,
    cp_upper,
    decide,
    decomposition,
    max_certifiable_harms,
    outcome_masks,
    risk_coverage,
    routed_needed,
    verifier_truth,
)


def _frame(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": split,
                "image_id": f"img-{i}",
                "score": score,
                "baseline_iou": base,
                "challenger_iou": chal,
            }
            for i, (split, score, base, chal) in enumerate(rows)
        ]
    )


class CertificationPowerTest(unittest.TestCase):
    def test_fewest_switches_that_can_certify_is_59(self) -> None:
        self.assertEqual(max_certifiable_harms(58), -1)
        self.assertEqual(max_certifiable_harms(59), 0)
        self.assertEqual(routed_needed(0.0), 59)

    def test_power_is_zero_without_a_certifiable_count(self) -> None:
        self.assertEqual(certification_power(30, 0.0), 0.0)

    def test_power_at_the_limit_never_exceeds_delta(self) -> None:
        for m in (59, 120, 500, 2000):
            self.assertLessEqual(certification_power(m, 0.05), 0.05 + 1e-12)

    def test_cp_upper_matches_known_values(self) -> None:
        self.assertTrue(np.isnan(cp_upper(0, 0, 0.05)))
        self.assertEqual(float(cp_upper(3, 3, 0.05)), 1.0)
        self.assertAlmostEqual(float(cp_upper(19, 93, 0.05)), 0.2852, places=3)


class DecompositionTest(unittest.TestCase):
    def test_harm_rate_is_decisive_share_times_harmful_share(self) -> None:
        frame = _frame(
            [
                ("select", 0.9, 0.1, 0.9),  # help
                ("select", 0.8, 0.9, 0.1),  # harm
                ("select", 0.7, 0.9, 0.8),  # neutral
                ("select", 0.6, 0.9, 0.9),  # neutral
                ("select", 0.0, 0.9, 0.9),  # no alternative
            ]
        )
        row = decomposition(frame, "select", (0.5,)).iloc[0]
        self.assertEqual(row["routed"], 4)
        self.assertAlmostEqual(
            row["harm_among_routed"],
            row["decisive_share"] * row["harmful_share_of_decisive"],
        )

    def test_risk_coverage_routes_ties_together_and_skips_zero(self) -> None:
        frame = _frame(
            [
                ("select", 0.5, 0.9, 0.1),
                ("select", 0.5, 0.1, 0.9),
                ("select", 0.2, 0.9, 0.9),
                ("select", 0.0, 0.9, 0.9),
            ]
        )
        curve = risk_coverage(frame)
        self.assertEqual(curve["routed"].tolist(), [2, 3])
        self.assertEqual(curve["harm"].tolist(), [1, 1])


class VerifierTest(unittest.TestCase):
    def test_perfect_verifier_removes_all_harm(self) -> None:
        frame = _frame([("select", 0.9, 0.9, 0.1), ("select", 0.9, 0.1, 0.9)])
        help_, harm, _ = outcome_masks(frame)
        truth = verifier_truth(
            frame["score"].to_numpy(), help_, harm, (0.5,), recall=1.0, false_veto=0.0
        )
        self.assertEqual(truth["conditional_harm"][0], 0.0)

    def test_audit_refuses_when_every_switch_harms(self) -> None:
        rows = [("select", 0.9, 0.9, 0.1)] * 80 + [("certify", 0.9, 0.9, 0.1)] * 80
        frame = _frame(rows)
        help_, harm, _ = outcome_masks(frame)
        scores = frame["score"].to_numpy()
        select = (frame["split"] == "select").to_numpy()
        certify = (frame["split"] == "certify").to_numpy()
        for method in ("cs2", "ltt_bonferroni", "same_split", "empirical_screen"):
            decision = decide(method, scores, help_, harm, select, certify, (0.5,))
            self.assertFalse(decision.deployed, method)

    def test_audit_certifies_a_harmless_rule_with_enough_switches(self) -> None:
        rows = [("select", 0.9, 0.1, 0.9)] * 80 + [("certify", 0.9, 0.1, 0.9)] * 80
        frame = _frame(rows)
        help_, harm, _ = outcome_masks(frame)
        decision = decide(
            "cs2",
            frame["score"].to_numpy(),
            help_,
            harm,
            (frame["split"] == "select").to_numpy(),
            (frame["split"] == "certify").to_numpy(),
            (0.5,),
        )
        self.assertTrue(decision.deployed)

    def test_controlled_study_is_deterministic(self) -> None:
        rows = [("select", 0.9, 0.9, 0.1)] * 30 + [("certify", 0.6, 0.1, 0.9)] * 30
        frame = _frame(rows)
        first = controlled_study(frame, recalls=(0.5,), false_vetoes=(0.0,), repeats=3)
        second = controlled_study(frame, recalls=(0.5,), false_vetoes=(0.0,), repeats=3)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
