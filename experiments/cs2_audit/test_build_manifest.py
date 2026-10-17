"""Unit tests for the raw-log -> audit-manifest conversion."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from .build_manifest import assign_splits, box_iou, build_rows, load_ground_truth


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class BoxIouTests(unittest.TestCase):
    def test_perfect_overlap_is_one(self) -> None:
        box = (0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_disjoint_boxes_are_zero(self) -> None:
        self.assertEqual(box_iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)), 0.0)


class GroundTruthTests(unittest.TestCase):
    def test_loads_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.parquet"
            frame = pd.DataFrame(
                {
                    "ref_id": [1, 2],
                    "bbox": [[10.0, 20.0, 60.0, 70.0], [0.0, 0.0, 50.0, 25.0]],
                    "raw_image_info": [
                        json.dumps({"width": 100, "height": 100}),
                        json.dumps({"width": 200, "height": 50}),
                    ],
                }
            )
            frame.to_parquet(path)
            truth = load_ground_truth(str(path))
        self.assertEqual(truth["1"], (0.1, 0.2, 0.6, 0.7))
        self.assertEqual(truth["2"], (0.0, 0.0, 0.25, 0.5))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_ground_truth("/nonexistent/path/*.parquet")


class BuildRowsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ground_truth = {"1": (0.0, 0.0, 1.0, 1.0)}
        self.generation = {
            "1": {"example_id": "1", "image_id": 42, "restricted_iou": 0.3},
        }

    def test_challenger_beats_incumbent(self) -> None:
        scores = {
            "1": {
                "example_id": "1",
                "candidate_boxes": [(0.0, 0.0, 0.4, 0.4), (0.0, 0.0, 1.0, 1.0)],
                "candidate_sources": ["restricted", "detector"],
                "candidate_scores": [{"pairwise_score": 0.1}, {"pairwise_score": 0.9}],
            }
        }
        rows, stats = build_rows(self.generation, scores, self.ground_truth, "pairwise_score")
        self.assertEqual(stats["ok"], 1)
        row = rows[0]
        self.assertEqual(row["image_id"], 42)
        self.assertAlmostEqual(row["score"], 0.8)
        self.assertAlmostEqual(row["baseline_iou"], 0.3)
        self.assertAlmostEqual(row["challenger_iou"], 1.0)

    def test_tie_resolves_to_incumbent(self) -> None:
        scores = {
            "1": {
                "example_id": "1",
                "candidate_boxes": [(0.0, 0.0, 0.4, 0.4), (0.0, 0.0, 1.0, 1.0)],
                "candidate_sources": ["restricted", "detector"],
                "candidate_scores": [{"pairwise_score": 0.5}, {"pairwise_score": 0.5}],
            }
        }
        rows, _ = build_rows(self.generation, scores, self.ground_truth, "pairwise_score")
        row = rows[0]
        self.assertEqual(row["score"], 0.0)
        self.assertAlmostEqual(row["challenger_iou"], box_iou((0.0, 0.0, 0.4, 0.4), (0.0, 0.0, 1.0, 1.0)))

    def test_missing_gt_is_counted_and_skipped(self) -> None:
        rows, stats = build_rows(
            {"2": {"example_id": "2", "image_id": 1, "restricted_iou": 0.0}},
            {},
            {},
            "pairwise_score",
        )
        self.assertEqual(rows, [])
        self.assertEqual(stats["no_gt"], 1)

    def test_missing_restricted_source_is_counted_and_skipped(self) -> None:
        scores = {
            "1": {
                "example_id": "1",
                "candidate_boxes": [(0.0, 0.0, 1.0, 1.0)],
                "candidate_sources": ["detector"],
                "candidate_scores": [{"pairwise_score": 0.5}],
            }
        }
        rows, stats = build_rows(self.generation, scores, self.ground_truth, "pairwise_score")
        self.assertEqual(rows, [])
        self.assertEqual(stats["no_restricted"], 1)


class AssignSplitsTests(unittest.TestCase):
    def test_confirmation_pool_is_never_repartitioned(self) -> None:
        cal_rows = [{"image_id": i} for i in range(20)]
        conf_rows = [{"image_id": 1000 + i} for i in range(5)]
        assign_splits(cal_rows, conf_rows, seed=0, select_frac=0.4, certify_frac=0.4)
        self.assertTrue(all(row["split"] == "confirm" for row in conf_rows))

    def test_calibration_pool_is_fully_partitioned_and_disjoint_by_image(self) -> None:
        cal_rows = [{"image_id": i} for i in range(20) for _ in range(2)]  # 2 rows/image
        conf_rows: list[dict[str, object]] = []
        assign_splits(cal_rows, conf_rows, seed=1, select_frac=0.4, certify_frac=0.4)
        by_image: dict[int, set[str]] = {}
        for row in cal_rows:
            by_image.setdefault(row["image_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in by_image.values()))
        self.assertEqual({row["split"] for row in cal_rows}, {"train", "select", "certify"})

    def test_rejects_fractions_that_leave_no_room_for_train(self) -> None:
        with self.assertRaises(ValueError):
            assign_splits([], [], seed=0, select_frac=0.6, certify_frac=0.5)

    def test_deterministic_given_seed(self) -> None:
        cal_rows_a = [{"image_id": i} for i in range(30)]
        cal_rows_b = [{"image_id": i} for i in range(30)]
        assign_splits(cal_rows_a, [], seed=7, select_frac=0.4, certify_frac=0.4)
        assign_splits(cal_rows_b, [], seed=7, select_frac=0.4, certify_frac=0.4)
        self.assertEqual(
            [row["split"] for row in cal_rows_a], [row["split"] for row in cal_rows_b]
        )


if __name__ == "__main__":
    unittest.main()
