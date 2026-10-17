"""Consistency of the released records with the audited records (runs where ``data/`` exists)."""

from __future__ import annotations

import gzip
import json
import unittest

import numpy as np
import pandas as pd

from experiments.cs2_audit.records import ROOT, load_registry
from experiments.cs2_audit.reduce_manifest import _reduce_legacy_source_row

DATA = ROOT / "data"
TIE = 1e-6


def slug(name: str) -> str:
    return name.replace("+", "plus")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@unittest.skipUnless((DATA / "system_a").exists(), "release data not present")
class SystemAReleaseTests(unittest.TestCase):
    def test_legacy_reduction_reproduces_every_registered_record(self) -> None:
        for name, entry in load_registry()["systems"]["System A"].items():
            expressions = pd.read_csv(DATA / "system_a" / f"{slug(name)}_expressions.csv.gz")
            for seed, rel in {"20270822": entry["primary"], **entry["reductions"]}.items():
                reduced, _ = _reduce_legacy_source_row(expressions, seed=int(seed))
                registered = pd.read_csv(ROOT / rel)
                reduced = reduced[registered.columns].reset_index(drop=True)
                self.assertEqual(len(reduced), len(registered), f"{name} {seed}")
                for column in registered.columns:
                    if pd.api.types.is_float_dtype(registered[column]):
                        self.assertTrue(np.allclose(reduced[column], registered[column], rtol=0, atol=1e-12), f"{name} {seed} {column}")
                    else:
                        self.assertTrue((reduced[column].astype(str).values == registered[column].astype(str).values).all(), f"{name} {seed} {column}")

    def test_candidate_sets_reproduce_the_expression_records(self) -> None:
        for name in load_registry()["systems"]["System A"]:
            expressions = pd.read_csv(DATA / "system_a" / f"{slug(name)}_expressions.csv.gz", dtype={"expression_id": str}).set_index("expression_id")
            for row in read_jsonl_gz(DATA / "system_a" / f"{slug(name)}_candidates.jsonl.gz"):
                record = expressions.loc[row["ref_id"]]
                ious = row["candidate_ious"]
                self.assertAlmostEqual(ious[row["default_index"]], record["baseline_iou"], places=9)
                self.assertAlmostEqual(ious[row["alternative_index"]], record["challenger_iou"], places=9)
                self.assertAlmostEqual(row["switch_score"], record["score"], places=9)


@unittest.skipUnless((DATA / "system_b").exists(), "release data not present")
class SystemBReleaseTests(unittest.TestCase):
    def test_candidate_ious_reproduce_the_audited_manifests(self) -> None:
        arms = {"RefCOCO+": "g1", "RefCOCOg": "g1", "RefCOCO": "g1", "RefCLEF": "g3"}
        records = {arm: {r["record_key"]: r for r in read_jsonl_gz(DATA / "system_b" / f"{arm}_records.jsonl.gz")} for arm in set(arms.values())}
        for name, entry in load_registry()["systems"]["System B"].items():
            manifest = pd.read_csv(ROOT / entry["primary"], dtype={"expression_id": str})
            for row in manifest.itertuples(index=False):
                record = records[arms[name]][f"{name}:{row.expression_id}"]
                ious = record["candidate_ious"]
                default = record.get("default_index")
                baseline = ious[default] if default is not None else 0.0
                self.assertAlmostEqual(baseline, row.baseline_iou, places=9)
                chosen = record.get("chosen_index")
                switched = chosen is not None and chosen != default and float(record["switch_score"]) >= TIE
                self.assertAlmostEqual(ious[chosen] if switched else baseline, row.challenger_iou, places=9)


if __name__ == "__main__":
    unittest.main()
