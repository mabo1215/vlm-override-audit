"""Tests for fresh-review protocol additions."""

import unittest

import pandas as pd

from .audit import Policy, run_protocol
from .cost_report import summarize_events
from .reduce_manifest import reduce_one_expression_per_image
from .risk_controls import (
    add_improvement_score,
    fit_improvement_discriminator,
    rcps_binary_harm_reference,
)
from .make_independent_splits import resplit_frame
from .register_stress_subset import register_subset
from .select_qualitative import select_examples
from .run_experiment import make_smoke_manifest


class RevisionProtocolTests(unittest.TestCase):
    def test_duplicate_expression_rows_are_rejected_until_reduced(self) -> None:
        frame = make_smoke_manifest()
        duplicate = frame.iloc[[48]].copy()
        duplicate["expression_id"] = "duplicate"
        duplicate["image_id"] = frame.iloc[48]["image_id"]
        repeated = pd.concat([frame, duplicate], ignore_index=True)
        with self.assertRaises(ValueError):
            run_protocol(repeated, [Policy("p", 0.4)])
        reduced, summary = reduce_one_expression_per_image(repeated, seed=7)
        self.assertEqual(len(reduced), len(frame))
        self.assertEqual(summary["output_rows"], len(frame))
        result = run_protocol(reduced, [Policy("p", 0.4)])
        self.assertTrue(result["protocol"]["one_row_per_image"])

    def test_reduction_is_deterministic(self) -> None:
        frame = make_smoke_manifest()
        extra = frame.iloc[[1]].copy()
        extra["expression_id"] = "extra"
        extra["image_id"] = frame.iloc[1]["image_id"]
        repeated = pd.concat([frame, extra], ignore_index=True)
        left, _ = reduce_one_expression_per_image(repeated, seed=11)
        right, _ = reduce_one_expression_per_image(repeated, seed=11)
        pd.testing.assert_frame_equal(left, right)

    def test_reduction_is_invariant_to_input_permutation(self) -> None:
        frame = make_smoke_manifest()
        extra = frame.iloc[[1]].copy()
        extra["expression_id"] = "extra"
        extra["image_id"] = frame.iloc[1]["image_id"]
        repeated = pd.concat([frame, extra], ignore_index=True)
        left, _ = reduce_one_expression_per_image(repeated, seed=11)
        permuted, _ = reduce_one_expression_per_image(
            repeated.sample(frac=1.0, random_state=17).reset_index(drop=True), seed=11
        )
        columns = ["split", "image_id", "expression_id"]
        left_view = left[columns].sort_values(columns).reset_index(drop=True)
        permuted_view = permuted[columns].sort_values(columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(left_view, permuted_view)

    def test_qualitative_selection_is_frozen_and_marks_render_pending(self) -> None:
        frame = make_smoke_manifest()
        result = select_examples(frame, Policy("p", 0.4), seed=9)
        self.assertEqual(result["render_status"], "render_pending")
        self.assertEqual(result["selection_rule"].startswith("minimum SHA-256"), True)

    def test_cost_summary_is_deterministic(self) -> None:
        summary = summarize_events(
            [
                {"duration_ms": 100, "vlm_calls": 2, "candidates": 4, "status": "ok", "batch_size": 2},
                {"duration_ms": 300, "vlm_calls": 1, "candidates": 2, "status": "failed", "retries": 1},
            ]
        )
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["failed_events"], 1)
        self.assertEqual(summary["candidates"], 6)
        self.assertEqual(summary["latency_ms"]["p95"], 290.0)

    def test_train_only_discriminator_adds_scores_and_rcps_is_explicit(self) -> None:
        frame = make_smoke_manifest()
        frame["feature_a"] = frame["score"]
        model = fit_improvement_discriminator(frame, ["feature_a"], steps=25)
        scored = add_improvement_score(frame, model)
        self.assertEqual(len(scored["improvement_score"]), len(frame))
        rcps = rcps_binary_harm_reference(
            frame[frame["split"] == "certify"], Policy("p", 0.4), split="certify"
        )
        self.assertIn("equivalent_to", rcps)

    def test_discriminator_rejects_ground_truth_features(self) -> None:
        frame = make_smoke_manifest()
        with self.assertRaises(ValueError):
            fit_improvement_discriminator(frame, ["challenger_iou"], steps=5)

    def test_independent_resplit_is_image_disjoint_and_deterministic(self) -> None:
        frame = make_smoke_manifest()
        left, summary = resplit_frame(frame, seed=19)
        right, _ = resplit_frame(frame, seed=19)
        pd.testing.assert_frame_equal(left, right)
        self.assertTrue(summary["one_row_per_image"])
        self.assertEqual(left["image_id"].nunique(), len(left))
        self.assertEqual(set(left["split"]), {"train", "select", "certify", "confirm"})

    def test_stress_subset_uses_metadata_only(self) -> None:
        frame = make_smoke_manifest()
        frame["expression_text"] = [
            "a person left of the bicycle with a long sleeve" if i % 2 == 0 else "short object"
            for i in range(len(frame))
        ]
        subset, summary = register_subset(frame, min_tokens=8)
        self.assertGreater(len(subset), 0)
        self.assertTrue(summary["selection_rule"]["uses_labels"] is False)
        self.assertTrue((subset["split"] == "confirm").all())


if __name__ == "__main__":
    unittest.main()
