"""Unit tests for the uncertified reference baselines."""

import unittest

from .baselines import (
    matched_random_reference,
    no_switch_reference,
    oracle_reference,
    raw_score_reference,
    relation_verifier_reference,
)
from .run_experiment import make_smoke_manifest


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = make_smoke_manifest()
        self.certify = frame[frame["split"] == "certify"].reset_index(drop=True)

    def test_no_switch_never_routes(self) -> None:
        result = no_switch_reference(self.certify, "certify")
        self.assertEqual(result.route_count, 0)
        self.assertIsNone(result.conditional_harm)
        self.assertEqual(result.net_accuracy_gain, 0.0)

    def test_oracle_never_harms(self) -> None:
        result = oracle_reference(self.certify, "certify")
        self.assertEqual(result.harm_count, 0)
        self.assertEqual(result.help_count, result.route_count)
        self.assertGreater(result.route_count, 0)
        self.assertEqual(result.net_accuracy_gain, result.route_count / result.n)

    def test_raw_score_is_never_certified(self) -> None:
        result = raw_score_reference(self.certify, "certify", threshold=0.5)
        self.assertIsNone(result.cp_upper)
        self.assertIn("descriptive", result.status)

    def test_matched_random_uses_requested_route_count(self) -> None:
        summary = matched_random_reference(
            self.certify, "certify", route_count=20, n_repeats=50, seed=1
        )
        self.assertEqual(summary.route_count, 20)
        self.assertEqual(summary.n_repeats, 50)
        self.assertEqual(len(summary.net_accuracy_gain_samples), 50)

    def test_matched_random_zero_route_is_deterministic(self) -> None:
        summary = matched_random_reference(
            self.certify, "certify", route_count=0, n_repeats=10, seed=1
        )
        self.assertEqual(summary.mean_help, 0.0)
        self.assertEqual(summary.mean_harm, 0.0)
        self.assertIsNone(summary.mean_conditional_harm)

    def test_matched_random_is_reproducible_given_seed(self) -> None:
        first = matched_random_reference(
            self.certify, "certify", route_count=20, n_repeats=30, seed=7
        )
        second = matched_random_reference(
            self.certify, "certify", route_count=20, n_repeats=30, seed=7
        )
        self.assertEqual(
            first.net_accuracy_gain_samples, second.net_accuracy_gain_samples
        )

    def test_relation_verifier_requires_declared_column(self) -> None:
        with self.assertRaises(ValueError):
            relation_verifier_reference(
                self.certify, "certify", "missing_column", threshold=0.5
            )

    def test_relation_verifier_uses_alternate_score_column(self) -> None:
        with_alt = self.certify.copy()
        with_alt["verifier_score"] = 1.0 - with_alt["score"]
        result = relation_verifier_reference(
            with_alt, "certify", "verifier_score", threshold=0.5
        )
        self.assertIsNone(result.cp_upper)
        self.assertIn("uncertified", result.status)


if __name__ == "__main__":
    unittest.main()
