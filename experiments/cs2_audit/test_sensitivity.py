"""Unit tests for cluster-aware bootstrap, AUROC, and placebo."""

import unittest

import numpy as np
import pandas as pd

from .audit import Policy
from .run_experiment import make_smoke_manifest
from .sensitivity import (
    auroc_help_harm,
    image_block_permutation_placebo,
    image_cluster_bootstrap,
)


def _perfectly_separable_frame() -> pd.DataFrame:
    # 5 help-eligible rows scored high, 5 harm-eligible rows scored low.
    rows = []
    for i in range(5):
        rows.append(
            {
                "split": "certify",
                "image_id": f"help-{i}",
                "score": 0.9 + 0.01 * i,
                "baseline_iou": 0.1,
                "challenger_iou": 0.9,
            }
        )
        rows.append(
            {
                "split": "certify",
                "image_id": f"harm-{i}",
                "score": 0.1 + 0.01 * i,
                "baseline_iou": 0.9,
                "challenger_iou": 0.1,
            }
        )
    return pd.DataFrame(rows)


class BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = make_smoke_manifest()
        self.certify = frame[frame["split"] == "certify"].reset_index(drop=True)

    def test_interval_contains_point_estimate(self) -> None:
        def mean_score(frame: pd.DataFrame) -> float:
            return float(frame["score"].mean())

        interval = image_cluster_bootstrap(
            self.certify, mean_score, n_bootstrap=200, seed=3
        )
        self.assertLessEqual(interval.lower, interval.point_estimate)
        self.assertGreaterEqual(interval.upper, interval.point_estimate)
        self.assertEqual(interval.cluster_unit, "image_id")

    def test_deterministic_given_seed(self) -> None:
        def mean_score(frame: pd.DataFrame) -> float:
            return float(frame["score"].mean())

        first = image_cluster_bootstrap(self.certify, mean_score, n_bootstrap=100, seed=5)
        second = image_cluster_bootstrap(self.certify, mean_score, n_bootstrap=100, seed=5)
        self.assertEqual(first.to_dict(), second.to_dict())


class AurocTests(unittest.TestCase):
    def test_perfectly_separable_scores_give_auroc_one(self) -> None:
        result = auroc_help_harm(_perfectly_separable_frame())
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["n_help"], 5)
        self.assertEqual(result["n_harm"], 5)

    def test_missing_one_class_returns_none(self) -> None:
        frame = _perfectly_separable_frame()
        frame = frame[frame["image_id"].str.startswith("help")].reset_index(drop=True)
        result = auroc_help_harm(frame)
        self.assertIsNone(result["auroc"])

    def test_tied_scores_do_not_crash(self) -> None:
        frame = _perfectly_separable_frame()
        frame["score"] = 0.5
        result = auroc_help_harm(frame)
        self.assertAlmostEqual(result["auroc"], 0.5)


class PlaceboTests(unittest.TestCase):
    def setUp(self) -> None:
        frame = make_smoke_manifest()
        self.certify = frame[frame["split"] == "certify"].reset_index(drop=True)

    def test_permutation_unit_is_image_level(self) -> None:
        policy = Policy("p", 0.5)
        result = image_block_permutation_placebo(
            self.certify, "certify", policy, n_repeats=20, seed=2
        )
        self.assertEqual(result["permutation_unit"], "image_id (block-shuffled image-mean score)")
        self.assertEqual(result["n_repeats"], 20)
        self.assertIn("null_hypothesis", result)
        self.assertTrue(0.0 <= result["p_value_one_sided_greater"] <= 1.0)

    def test_deterministic_given_seed(self) -> None:
        policy = Policy("p", 0.5)
        first = image_block_permutation_placebo(
            self.certify, "certify", policy, n_repeats=15, seed=9
        )
        second = image_block_permutation_placebo(
            self.certify, "certify", policy, n_repeats=15, seed=9
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
