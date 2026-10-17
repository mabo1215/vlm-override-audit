"""Adapt a cached jxu124/refcoco(+/g) parquet snapshot into the metadata
schema used by ``generate_owlv2_candidates.py`` and
``build_alternative_manifest.py``.

The jxu124 mirror stores a ref-specific ``file_name`` (e.g. with a numeric
crop suffix) that does not match the real COCO archive filename.  This
command replaces it with the true filename from ``raw_image_info`` and
attaches an absolute ``image_path`` under the extracted COCO image root,
leaving every other column (``ref_id``, ``bbox``, ``sentences``,
``raw_image_info``) untouched so ground truth is unaffected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build(input_parquet: Path, image_root: Path, coco_subdir: str) -> pd.DataFrame:
    frame = pd.read_parquet(
        input_parquet,
        columns=["ref_id", "image_id", "sentences", "bbox", "raw_image_info"],
    )
    file_names = frame["raw_image_info"].map(lambda raw: json.loads(raw)["file_name"])
    frame = frame.assign(
        file_name=file_names,
        image_path=[str(image_root / coco_subdir / name) for name in file_names],
    )
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--coco-subdir", default="train2014")
    parser.add_argument("--output-parquet", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build(args.input_parquet, args.image_root, args.coco_subdir)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_parquet, index=False)
    print(f"wrote {len(frame)} rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
