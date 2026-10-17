from __future__ import annotations

import unittest

import pandas as pd

from experiments.cs2_audit import qualitative_figure as qf
from experiments.cs2_audit.select_qualitative import _key


def records() -> pd.DataFrame:
    rows = []
    for i in range(1, 13):
        kind = i % 3  # 0: help, 1: harm, 2: no switch
        base, alt, score = ((0.1, 0.9, 0.8), (0.9, 0.1, 0.8), (0.9, 0.9, 0.0))[kind]
        rows.append({"split": "confirm", "image_id": i, "expression_id": str(100 + i), "score": score,
                     "baseline_iou": base, "challenger_iou": alt})
    return pd.DataFrame(rows)


class SelectionTests(unittest.TestCase):
    def test_skips_follow_the_hash_order_and_both_licence_checks(self) -> None:
        frame = records()
        licences = pd.DataFrame({"image_id": range(1, 13), "file_name": [f"{i}.jpg" for i in range(1, 13)],
                                 "license": 4, "license_name": "Attribution License", "license_url": "u", "flickr_url": "f"})
        harm = frame[frame["image_id"] % 3 == 1].copy()
        order = sorted(harm["image_id"], key=lambda v: _key(qf.SEED, "harm", v))
        licences.loc[licences["image_id"] == order[0], "license"] = 3  # NonCommercial: skipped
        checks = {str(i): {"flickr_photo_id": str(i), "status": "ok", "licence": "CC BY 2.0", "owner": "o", "title": "t"} for i in range(1, 13)}
        checks[str(order[1])]["status"] = "unavailable"
        out = qf.select(frame, 0.4, licences, checks)
        harm_out = out["categories"]["harm"]
        self.assertEqual(harm_out["selected"]["image_id"], order[2])
        self.assertEqual([s["reason"] for s in harm_out["skipped_for_license"]], ["COCO licence", "Flickr page unavailable"])
        self.assertEqual(out["categories"]["help"]["candidates"], 4)
        del checks[str(order[2])]
        with self.assertRaises(SystemExit):
            qf.select(frame, 0.4, licences, checks)  # an unchecked eligible image stops the selection


if __name__ == "__main__":
    unittest.main()
