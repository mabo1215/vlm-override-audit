from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.cs2_audit import verifier as v
from experiments.cs2_audit.audit import run_protocol


class GeometryTests(unittest.TestCase):
    def test_features_and_crop(self) -> None:
        f = v.geometry_features((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
        self.assertAlmostEqual(f["iou_bc"], 1 / 7)
        self.assertAlmostEqual(f["log_area_ratio"], 0.0)
        self.assertAlmostEqual(f["centre_dx"], 0.25)
        self.assertEqual(v.padded_crop_box((0.0, 0.0, 0.5, 0.5), 100, 100), (0, 0, 55, 55))

    def test_feature_names_are_label_free(self) -> None:
        v.assert_label_free(v.feature_columns())
        with self.assertRaises(ValueError):
            v.assert_label_free(v.feature_columns() + ["candidate_gt_iou"])


def sample_record():
    return {
        "record_key": "RefCOCO:1", "image_id": 1, "default_index": 0, "chosen_index": 2,
        "candidates": [{"box": [0.0, 0.0, 0.5, 0.5]}, {"box": [0.5, 0.5, 1.0, 1.0]}, {"box": [0.1, 0.1, 0.5, 0.5]}],
    }


class PairTests(unittest.TestCase):
    def record(self):
        return sample_record()

    def test_pairs_and_labels(self) -> None:
        sims = {"siglip": [0.1, 0.2, 0.3], "clip": [0.4, 0.5, 0.6]}
        pairs = pd.DataFrame(v.pair_rows(self.record(), sims))
        self.assertEqual(list(pairs["candidate_index"]), [1, 2])
        self.assertEqual(list(pairs["is_chosen"]), [False, True])
        self.assertAlmostEqual(pairs.loc[0, "siglip_sim_diff"], 0.1)
        labels = v.harm_labels(pairs, {"RefCOCO:1": self.record()}, {"RefCOCO:1": (0.0, 0.0, 0.5, 0.5)})
        self.assertEqual(list(labels), [1, 0])


class TrainingTests(unittest.TestCase):
    def test_trains_calibrates_and_predicts(self) -> None:
        rng = np.random.default_rng(0)
        n = 600
        frame = pd.DataFrame({c: rng.normal(size=n) for c in v.feature_columns()})
        frame["image_id"] = rng.integers(0, 150, size=n)
        labels = (frame["siglip_sim_diff"] + 0.3 * rng.normal(size=n) < -0.5).astype(int).to_numpy()
        model = v.train_verifier(frame, labels, seed=1)
        pred = model.predict(frame)
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())
        self.assertGreater(np.corrcoef(pred, labels)[0, 1], 0.3)
        self.assertIn(model.metadata["selected_model"], model.metadata["cv_auroc"])


class GatedPolicyTests(unittest.TestCase):
    def test_gate_vetoes_and_runs_through_protocol(self) -> None:
        rows = []
        for split in ("train", "select", "certify", "confirm"):
            for i in range(40):
                harmful = i % 4 == 0
                rows.append({"split": split, "image_id": f"{split}{i}", "expression_id": f"{split}{i}",
                             "score": 0.8 if i % 2 == 0 else 0.0,
                             "baseline_iou": 0.9 if harmful else 0.2, "challenger_iou": 0.1 if harmful else 0.8,
                             "verifier_harm": 0.9 if harmful else 0.01})
        frame = pd.DataFrame(rows)
        frame.loc[frame["score"] == 0, "challenger_iou"] = frame.loc[frame["score"] == 0, "baseline_iou"]
        family = v.gated_family([1e-6, 0.5], nus=(0.05, 0.3))
        result = run_protocol(frame, family)
        self.assertEqual(result["certify"]["harm_count"], 0)
        self.assertGreater(result["certify"]["route_count"], 0)

    def test_attach_verifier_requires_predictions_for_switches(self) -> None:
        manifest = pd.DataFrame({"expression_id": ["1", "2"], "score": [0.5, 0.0]})
        preds = pd.DataFrame({"record_key": ["RefCOCO:1"], "verifier_harm": [0.2]})
        out = v.attach_verifier(manifest, preds, "RefCOCO")
        self.assertEqual(list(out["verifier_harm"]), [0.2, 0.0])
        with self.assertRaises(ValueError):
            v.attach_verifier(manifest, preds.iloc[:0], "RefCOCO")
        prefixed = preds.assign(record_key=v.SYSTEM_A_PREFIX + preds["record_key"])
        out = v.attach_verifier(manifest, prefixed, "RefCOCO", key_prefix=v.SYSTEM_A_PREFIX)
        self.assertEqual(list(out["verifier_harm"]), [0.2, 0.0])


class FrameIndexTests(unittest.TestCase):
    def test_records_without_default_do_not_break_the_frame(self) -> None:
        records = pd.DataFrame(
            [
                {**sample_record(), "record_key": "RefCOCO:1"},
                {**sample_record(), "record_key": "RefCOCO:2", "default_index": None, "chosen_index": None},
            ]
        ).to_dict("records")
        sims = {"siglip": [0.1, 0.2, 0.3], "clip": [0.4, 0.5, 0.6]}
        self.assertIsInstance(records[0]["default_index"], float)  # what a mixed frame gives back
        rows = v.pair_rows(records[0], sims)
        self.assertEqual([r["is_chosen"] for r in rows], [False, True])
        self.assertEqual(v.pair_rows(records[1], sims), [])
        labels = v.harm_labels(pd.DataFrame(rows), {"RefCOCO:1": records[0]}, {"RefCOCO:1": (0.0, 0.0, 0.5, 0.5)})
        self.assertEqual(list(labels), [1, 0])


class SystemATests(unittest.TestCase):
    def entry(self, example_id: str, boxes, scores, sources=None):
        return {
            "example_id": example_id, "file_name": f"img{example_id}.jpg", "referring_expression": f"text {example_id}",
            "candidate_boxes": boxes, "candidate_sources": sources or ["restricted"] + ["detector"] * (len(boxes) - 1),
            "candidate_scores": [{"pairwise_score": s} for s in scores],
        }

    def test_export_keys_sets_and_duplicates(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        a, b = [0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / v.SYSTEM_A_DIRS["RefCOCO"] / "scores"
            scores.mkdir(parents=True)
            (scores / "cal_shard0_scores.json").write_text(json.dumps({"scores": [
                self.entry("1", [a, b], [0.0, 0.4]),
                self.entry("2", [a, b], [0.0, 5e-7]),
                self.entry("9", [a], [0.0]),  # not a System B record
            ]}))
            (scores / "conf_shard0_scores.json").write_text(json.dumps({"scores": [self.entry("1", [a, b], [0.0, 0.4])]}))
            table = root / "records.csv"
            pd.DataFrame([
                {"benchmark": "RefCOCO", "split": "train", "expression_id": "1", "image_id": 11},
                {"benchmark": "RefCOCO", "split": "select", "expression_id": "2", "image_id": 12},
                {"benchmark": "RefCOCO", "split": "confirm", "expression_id": "3", "image_id": 13},
            ]).to_csv(table, index=False)
            records, coverage = v.export_system_a(root, table)
            by_key = {r["record_key"]: r for r in records}
            self.assertEqual(sorted(by_key), ["SystemA:RefCOCO:1", "SystemA:RefCOCO:2"])
            self.assertEqual(by_key["SystemA:RefCOCO:1"]["chosen_index"], 1)
            self.assertEqual(by_key["SystemA:RefCOCO:2"]["chosen_index"], 0)  # below the tie tolerance
            self.assertEqual(by_key["SystemA:RefCOCO:1"]["image_id"], 11)
            self.assertEqual(by_key["SystemA:RefCOCO:1"]["truth_key"], "RefCOCO:1")
            self.assertEqual(coverage["RefCOCO"]["missing"], 1)
            labels = v.harm_labels(
                pd.DataFrame(v.pair_rows(by_key["SystemA:RefCOCO:1"], {"siglip": [0, 0], "clip": [0, 0]})),
                by_key, {"RefCOCO:1": tuple(a)},
            )
            self.assertEqual(list(labels), [1])
            (scores / "conf_shard0_scores.json").write_text(json.dumps({"scores": [self.entry("1", [a], [0.0])]}))
            with self.assertRaises(ValueError):
                v.export_system_a(root, table)


if __name__ == "__main__":
    unittest.main()
