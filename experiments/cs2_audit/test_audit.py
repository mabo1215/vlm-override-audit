"""Unit tests for the fixed-policy audit implementation."""

import unittest

import pandas as pd

from .audit import Policy, clopper_pearson_upper, run_protocol
from .run_experiment import make_smoke_manifest


class AuditTests(unittest.TestCase):
    def test_zero_route_has_undefined_conditional_harm(self) -> None:
        frame = make_smoke_manifest()
        result = run_protocol(frame, [Policy("never", 2.0)])
        self.assertEqual(result["certify"]["route_count"], 0)
        self.assertIsNone(result["certify"]["conditional_harm"])
        self.assertIsNone(result["certify"]["cp_upper"])
        self.assertIn("zero-route", result["certify"]["status"])

    def test_cp_handles_extreme_counts(self) -> None:
        self.assertIsNone(clopper_pearson_upper(0, 0, 0.05))
        self.assertEqual(clopper_pearson_upper(5, 5, 0.05), 1.0)
        self.assertGreater(clopper_pearson_upper(0, 10, 0.05), 0.0)

    def test_cp_rejects_invalid_counts_and_confidence_budget(self) -> None:
        for args in ((-1, 3, 0.05), (4, 3, 0.05), (0, 3, 0.0), (0, 3, 1.0)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    clopper_pearson_upper(*args)

    def test_image_overlap_is_rejected(self) -> None:
        frame = make_smoke_manifest()
        frame.loc[frame.index[-1], "image_id"] = frame.loc[0, "image_id"]
        with self.assertRaises(ValueError):
            run_protocol(frame, [Policy("p", 0.5)])

    def test_protocol_freezes_one_policy(self) -> None:
        frame = make_smoke_manifest()
        result = run_protocol(frame, [Policy("low", 0.4), Policy("high", 0.8)])
        self.assertTrue(result["protocol"]["policy_frozen_before_certify"])
        self.assertEqual(result["selected_policy"]["name"], result["certify"]["policy"])
        self.assertEqual(result["selected_policy"]["name"], result["confirm"]["policy"])

    def test_confirmation_is_a_utility_gate_not_a_second_selection(self) -> None:
        frame = make_smoke_manifest()
        result = run_protocol(frame, [Policy("low", 0.4), Policy("high", 0.8)])
        self.assertIn(result["protocol"]["deployment_status"], {"admitted", "not-admitted"})
        self.assertEqual(
            result["protocol"]["confirm_utility_gate"],
            result["confirm"]["net_accuracy_gain"] >= 0.0,
        )


if __name__ == "__main__":
    unittest.main()
