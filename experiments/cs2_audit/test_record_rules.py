from __future__ import annotations

import unittest

import pandas as pd

from experiments.cs2_audit.audit import validate_record_consistency
from experiments.cs2_audit.rebuild_system_a_records import rebuild
from experiments.cs2_audit.reduce_manifest import reduce_one_expression_per_image


def manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "select", "certify", "confirm"],
            "image_id": [1, 2, 3, 4],
            "expression_id": ["a", "b", "c", "d"],
            "score": [0.0, 5e-7, 0.8, 0.0],
            "baseline_iou": [0.0, 0.7, 0.6, 0.3],
            "challenger_iou": [0.65, 0.2, 0.1, 0.3],
        }
    )


class RecordRuleTests(unittest.TestCase):
    def test_validator_rejects_ties_and_inconsistent_no_switch(self) -> None:
        with self.assertRaises(ValueError):
            validate_record_consistency(manifest())
        clean = manifest().assign(score=[0.0, 0.0, 0.8, 0.0], challenger_iou=[0.0, 0.7, 0.1, 0.3])
        validate_record_consistency(clean)

    def test_rebuild_uses_scored_default_and_tie_tolerance(self) -> None:
        per_row = pd.DataFrame({"example_id": ["a", "b", "c", "d"], "score_restricted_iou": [0.65, 0.7, 0.6, 0.3]})
        out, changes = rebuild(manifest(), per_row)
        self.assertEqual(out.loc[0, "baseline_iou"], 0.65)  # parse failure: the scored default was a detector box
        self.assertEqual(out.loc[1, "score"], 0.0)
        self.assertEqual(out.loc[1, "challenger_iou"], 0.7)
        self.assertEqual(out.loc[2, "challenger_iou"], 0.1)
        self.assertEqual(changes["ties_set_to_no_switch"], 1)
        self.assertEqual(changes["default_correctness_changed"], 1)


class ReductionRuleTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        rows = []
        for image in range(20):
            for expr in range(3):
                rows.append({"split": "select", "image_id": image, "expression_id": f"{image}-{expr}", "score": 0.0, "baseline_iou": 0.5, "challenger_iou": 0.5})
        return pd.DataFrame(rows)

    def test_identity_rule_ignores_order_but_legacy_rule_does_not(self) -> None:
        frame = self.frame()
        shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
        ident = lambda f: set(reduce_one_expression_per_image(f, rule="identity")[0]["expression_id"])
        legacy = lambda f: set(reduce_one_expression_per_image(f, rule="legacy-source-row")[0]["expression_id"])
        self.assertEqual(ident(frame), ident(shuffled))
        self.assertNotEqual(legacy(frame), legacy(shuffled))
        self.assertEqual(len(legacy(frame)), 20)


if __name__ == "__main__":
    unittest.main()
