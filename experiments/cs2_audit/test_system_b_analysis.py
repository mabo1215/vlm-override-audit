from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.cs2_audit import system_b_analysis as sa
from experiments.cs2_audit.audit import Policy


def frame(rows_per_split: int = 40, harmful_every: int = 4) -> pd.DataFrame:
    rows = []
    for split in ("train", "select", "certify", "confirm"):
        for i in range(rows_per_split):
            switch = i % 2 == 0
            harmful = switch and i % harmful_every == 0
            helpful = switch and not harmful and i % 3 == 0
            base, alt = (0.9, 0.1) if harmful else (0.2, 0.8) if helpful else (0.9, 0.9)
            rows.append({"split": split, "image_id": f"{split}{i}", "expression_id": f"{split}{i}",
                         "score": (0.8 if i % 8 else 0.3) if switch else 0.0, "baseline_iou": base,
                         "challenger_iou": alt if switch else base, "verifier_harm": 0.9 if harmful else 0.01})
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_exact_interval(self) -> None:
        self.assertEqual(sa.exact_interval(0, 0), None)
        lo, hi = sa.exact_interval(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.3084971, places=5)
        lo, hi = sa.exact_interval(10, 10)
        self.assertEqual(hi, 1.0)

    def test_ltt_uses_the_pooled_sets_and_bonferroni(self) -> None:
        f = frame()
        policies = [Policy(name="score_ge_0.2", threshold=0.2), Policy(name="score_ge_0.5", threshold=0.5)]
        out = sa.ltt_bonferroni(f, policies)
        self.assertEqual(out["family_size"], 2)
        self.assertEqual(out["pooled_records"], 80)
        self.assertFalse(out["deployed"])  # a quarter of the switches are harmful
        safe = f.assign(challenger_iou=np.where(f["score"] > 0, np.maximum(f["challenger_iou"], f["baseline_iou"]), f["challenger_iou"]))
        many = pd.concat([safe.assign(image_id=safe["image_id"] + f"_{r}", expression_id=safe["expression_id"] + f"_{r}") for r in range(5)])
        out = sa.ltt_bonferroni(many, policies)
        self.assertTrue(out["deployed"])
        self.assertEqual(out["confirm"]["harm_count"], 0)

    def test_verifier_diagnostics(self) -> None:
        d = sa.verifier_diagnostics(frame(), tau=0.2, nu=0.5, split="select")
        self.assertEqual(d["harm_recall"], 1.0)
        self.assertEqual(d["false_veto_rate"], 0.0)
        self.assertEqual(d["auroc_v_harm_over_help"], 1.0)
        self.assertIsNotNone(d["harm_recall_ci95"])


if __name__ == "__main__":
    unittest.main()
