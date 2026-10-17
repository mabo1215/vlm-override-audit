from __future__ import annotations

import unittest

import pandas as pd

from experiments.cs2_audit.check_default_frame import smart_resize
from experiments.cs2_audit.default_frame_impact import corrected_manifest


class SmartResizeTests(unittest.TestCase):
    def test_rounds_sides_to_multiples_of_28(self) -> None:
        self.assertEqual(smart_resize(480, 640, 28, 3136, 1_003_520), (476, 644))

    def test_caps_large_images_at_max_pixels(self) -> None:
        height, width = smart_resize(2000, 3000, 28, 3136, 1_003_520)
        self.assertLessEqual(height * width, 1_003_520)
        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)


class CorrectedManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pd.DataFrame(
            {
                "split": ["select", "certify", "confirm"],
                "image_id": [1, 2, 3],
                "expression_id": ["a", "b", "c"],
                "score": [0.0, 0.7, 0.0],
                "baseline_iou": [0.45, 0.6, 0.2],
                "challenger_iou": [0.45, 0.1, 0.3],
            }
        )
        self.per_row = pd.DataFrame(
            {
                "example_id": ["a", "b", "c"],
                "status": ["pixel", "pixel", "parse_failure"],
                "fixed_iou": [0.55, 0.62, None],
                "out_of_range": [True, False, None],
                "score_restricted_matches_gen": [True, True, False],
            }
        )

    def test_replaces_default_and_no_switch_alternative(self) -> None:
        fixed, stats = corrected_manifest(self.manifest, self.per_row)
        self.assertAlmostEqual(fixed.loc[0, "baseline_iou"], 0.55)
        self.assertAlmostEqual(fixed.loc[0, "challenger_iou"], 0.55)
        self.assertAlmostEqual(fixed.loc[1, "baseline_iou"], 0.62)
        self.assertAlmostEqual(fixed.loc[1, "challenger_iou"], 0.1)
        self.assertEqual(stats["default_correct_flips"]["select"], 1)
        self.assertEqual(stats["no_switch_rows_corrected"], 1)

    def test_keeps_parse_failures_unchanged(self) -> None:
        fixed, _ = corrected_manifest(self.manifest, self.per_row)
        self.assertAlmostEqual(fixed.loc[2, "baseline_iou"], 0.2)
        self.assertAlmostEqual(fixed.loc[2, "challenger_iou"], 0.3)

    def test_rejects_rows_missing_from_the_check(self) -> None:
        with self.assertRaises(ValueError):
            corrected_manifest(self.manifest, self.per_row.iloc[:2])


if __name__ == "__main__":
    unittest.main()
