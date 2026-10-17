from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.cs2_audit import prepare_system_b_inputs as prep
from experiments.cs2_audit.system_b import load_image


def jpeg_bytes(width: int = 40, height: int = 30) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def refclef_row(image_id: int, refs: list[tuple[int, list[float], str]]) -> dict:
    return {
        "ref_list": [
            {
                "ann_info": {"bbox": bbox, "image_id": image_id},
                "ref_info": {"ref_id": ref_id, "image_id": image_id, "sentences": [{"raw": text, "sent": text.lower()}]},
            }
            for ref_id, bbox, text in refs
        ],
        "image_info": {"file_name": f"{image_id % 100:02d}/images/{image_id}.jpg", "height": 30, "id": image_id, "width": 40},
        "image": {"bytes": jpeg_bytes(), "path": None},
    }


class RefclefTests(unittest.TestCase):
    def test_one_ref_per_image_split_and_box_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [refclef_row(i, [(10 * i, [4.0, 3.0, 20.0, 15.0], "Left Dog"), (10 * i + 1, [0.0, 0.0, 40.0, 30.0], "sky")]) for i in range(1, 41)]
            parquet = root / "refclef.parquet"
            pd.DataFrame(rows).to_parquet(parquet)
            records, truth, summary = prep.refclef_inputs([str(parquet)], root / "images")
            again, _, _ = prep.refclef_inputs([str(parquet)], root / "images")
            self.assertEqual(len(records), 40)
            self.assertTrue(records.equals(again))
            self.assertEqual(records["image_id"].nunique(), 40)
            self.assertEqual(summary["split_counts"], {"select": 12, "certify": 12, "confirm": 12, "train": 4})
            self.assertFalse(any(c in records.columns for c in ("x1", "bbox", "iou")))
            merged = records.merge(truth, on=["benchmark", "expression_id"])
            dog = merged[merged["text"] == "Left Dog"]
            if not dog.empty:
                row = dog.iloc[0]
                self.assertAlmostEqual(row["x1"], 0.1)
                self.assertAlmostEqual(row["y2"], 0.6)
            self.assertTrue(Path(records["image_path"].iloc[0]).exists())


class TextRuleTests(unittest.TestCase):
    def test_captions_take_precedence(self) -> None:
        self.assertEqual(prep._first_text({"captions": ["a man"], "sentences": [{"raw": "other"}]}), "a man")
        self.assertEqual(prep._first_text({"sentences": [{"raw": "Raw text", "sent": "raw text"}]}), "Raw text")


class ZipImageTests(unittest.TestCase):
    def test_loads_by_basename_from_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "train2014.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train2014/COCO_train2014_000000000001.jpg", jpeg_bytes(12, 8))
            image = load_image({"image_path": "COCO_train2014_000000000001.jpg"}, archive)
            self.assertEqual(image.size, (12, 8))



class VerifierPoolTests(unittest.TestCase):
    def test_draws_disjoint_images_with_all_expressions(self) -> None:
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = _json.dumps({"width": 100, "height": 50})
            rows = [{"ref_id": 10 * i + j, "image_id": i, "file_name": f"x_{i}_{j}.jpg", "image_path": f"train2014/COCO_train2014_{i:012d}.jpg",
                     "split": "train", "captions": [f"thing {i}-{j}"], "sentences": [{"raw": "r"}], "bbox": [10.0, 5.0, 60.0, 25.0], "raw_image_info": info}
                     for i in range(30) for j in range(2)]
            pd.DataFrame(rows).to_parquet(root / "train.parquet")
            pd.DataFrame({"image_id": list(range(10))}).to_parquet(root / "pool.parquet")
            records, truth, summary = prep.verifier_pool_inputs({"RefCOCO": root / "train.parquet"}, [str(root / "pool.parquet")], 8)
            self.assertEqual(summary["eligible_images"], 20)
            self.assertEqual(records["image_id"].nunique(), 8)
            self.assertTrue((records["image_id"] >= 10).all())
            self.assertEqual(len(records), 16)
            self.assertTrue((records["split"] == "train").all())
            self.assertTrue(records["file_name"].str.startswith("COCO_train2014_").all())
            self.assertAlmostEqual(truth["x2"].iloc[0], 0.6)
            with self.assertRaises(ValueError):
                prep.verifier_pool_inputs({"RefCOCO": root / "train.parquet"}, [str(root / "pool.parquet")], 50)


def write_logs(root: Path) -> None:
    import json as _json

    scores = {"scores": [
        {"example_id": "11", "file_name": "COCO_train2014_000000000001.jpg", "referring_expression": "left cat",
         "candidate_boxes": [[0.5, 0.5, 1.0, 1.0], [0.1, 0.2, 0.6, 0.6]]},
        {"example_id": "12", "file_name": "COCO_train2014_000000000002.jpg", "referring_expression": "red car",
         "candidate_boxes": [[0.0, 0.0, 0.5, 0.5]]},
    ]}
    candidates = {"logs": [
        {"example_id": "11", "image_id": 1, "file_name": "COCO_train2014_000000000001.jpg"},
        {"example_id": "12", "image_id": 2, "file_name": "COCO_train2014_000000000002.jpg"},
        {"example_id": "13", "image_id": 3, "file_name": "COCO_train2014_000000000003.jpg"},
        {"task": "grounding", "error": "ValueError: no candidates"},
    ]}
    (root / "a_scores.json").write_text(_json.dumps(scores))
    (root / "a.json").write_text(_json.dumps(candidates))


class RebuildPoolTests(unittest.TestCase):
    def public(self, root: Path, shift: float = 0.0) -> Path:
        import json as _json

        rows = [
            {"ref_id": ref, "image_id": image, "file_name": f"COCO_train2014_{image:012d}_9.jpg",
             "raw_image_info": _json.dumps({"file_name": f"COCO_train2014_{image:012d}.jpg", "width": 100, "height": 50}),
             "captions": [text], "sentences": [{"raw": text}], "bbox": [10.0 + shift, 10.0, 60.0, 30.0], "split": "train"}
            for ref, image, text in ((11, 1, "left cat"), (12, 2, "red car"), (13, 3, "tree"), (14, 4, "unused"))
        ]
        path = root / "public.parquet"
        pd.DataFrame(rows).to_parquet(path)
        return path

    def per_row(self, iou: float) -> pd.DataFrame:
        return pd.DataFrame([{"benchmark": "RefCOCOg", "example_id": 11.0, "score_has_restricted": True,
                              "score_restricted_index": 1, "score_restricted_iou": iou}])

    def test_rebuild_checks_files_texts_and_ious(self) -> None:
        from experiments.cs2_audit.system_b import box_iou

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_logs(root)
            public = self.public(root)
            expected = box_iou((0.1, 0.2, 0.6, 0.6), (0.1, 0.2, 0.6, 0.6))
            rows, summary = prep.rebuild_pool([str(root / "*.json")], [str(public)], self.per_row(expected), "RefCOCOg")
            self.assertEqual(rows["ref_id"].tolist(), ["11", "12", "13"])
            self.assertEqual(summary["log_errors_without_id"], 1)
            self.assertEqual(summary["iou_checked_refs"], 1)
            with self.assertRaises(ValueError):
                prep.rebuild_pool([str(root / "*.json")], [str(public)], self.per_row(0.5), "RefCOCOg")
            with self.assertRaises(ValueError):
                prep.rebuild_pool([str(root / "*.json")], [str(self.public(root, shift=5.0))], self.per_row(expected), "RefCOCOg")

    def test_rebuild_rejects_text_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_logs(root)
            public = self.public(root)
            frame = pd.read_parquet(public)
            frame.loc[frame["ref_id"] == 12, "captions"] = pd.Series([["blue car"]], index=frame.index[frame["ref_id"] == 12])
            frame.to_parquet(public)
            with self.assertRaises(ValueError):
                prep.rebuild_pool([str(root / "*.json")], [str(public)])


class LoggedImageTests(unittest.TestCase):
    def test_ids_from_both_log_layouts_and_pool_exclusion(self) -> None:
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_logs(root)
            ids, files = prep.logged_image_ids([str(root / "*.json")])
            self.assertEqual(ids, {1, 2, 3})
            self.assertEqual(files, 2)
            info = _json.dumps({"width": 100, "height": 50})
            rows = [{"ref_id": i, "image_id": i, "file_name": f"x_{i}.jpg", "image_path": f"COCO_train2014_{i:012d}.jpg", "split": "train",
                     "captions": [f"t{i}"], "sentences": [{"raw": "r"}], "bbox": [1.0, 1.0, 5.0, 5.0], "raw_image_info": info} for i in range(1, 9)]
            pd.DataFrame(rows).to_parquet(root / "train.parquet")
            records, _, summary = prep.verifier_pool_inputs({"RefCOCOg": root / "train.parquet"}, [], 5, exclude_logs=[str(root / "*.json")])
            self.assertEqual(summary["eligible_images"], 5)
            self.assertTrue((records["image_id"] > 3).all())
            with self.assertRaises(ValueError):
                prep.verifier_pool_inputs({"RefCOCOg": root / "train.parquet"}, [], 5, exclude_logs=[str(root / "none_*.json")])


if __name__ == "__main__":
    unittest.main()
