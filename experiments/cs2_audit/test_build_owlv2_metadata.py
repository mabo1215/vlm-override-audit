import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .build_owlv2_metadata import build


class BuildOwlv2MetadataTests(unittest.TestCase):
    def test_replaces_filename_and_adds_image_path(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_parquet = tmp_path / "jxu124_sample.parquet"
            frame = pd.DataFrame(
                {
                    "ref_id": [0, 1],
                    "image_id": [581857, 581839],
                    "file_name": [
                        "COCO_train2014_000000581857_16.jpg",
                        "COCO_train2014_000000581839_3.jpg",
                    ],
                    "sentences": [
                        [{"sent": "a lady", "raw": "A LADY"}],
                        [{"sent": "a dog", "raw": "A DOG"}],
                    ],
                    "bbox": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                    "raw_image_info": [
                        json.dumps({"file_name": "COCO_train2014_000000581857.jpg", "width": 427, "height": 640}),
                        json.dumps({"file_name": "COCO_train2014_000000581839.jpg", "width": 500, "height": 375}),
                    ],
                }
            )
            frame.to_parquet(input_parquet, index=False)

            output = build(input_parquet, tmp_path / "coco", "train2014")

            self.assertEqual(
                output["file_name"].tolist(),
                ["COCO_train2014_000000581857.jpg", "COCO_train2014_000000581839.jpg"],
            )
            self.assertEqual(
                output["image_path"].tolist(),
                [
                    str(tmp_path / "coco" / "train2014" / "COCO_train2014_000000581857.jpg"),
                    str(tmp_path / "coco" / "train2014" / "COCO_train2014_000000581839.jpg"),
                ],
            )
            self.assertEqual(output["ref_id"].tolist(), [0, 1])
            self.assertEqual(output.loc[0, "sentences"][0]["sent"], "a lady")


if __name__ == "__main__":
    unittest.main()
