from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.cs2_audit import join_system_b_labels as jl


def record(key: int, split: str, score: float, chosen: int | None, default=(0.0, 0.0, 0.5, 0.5), other=(0.5, 0.5, 1.0, 1.0)):
    return {
        "record_key": f"RefCOCO:{key}",
        "benchmark": "RefCOCO",
        "split": split,
        "image_id": key,
        "expression_id": str(key),
        "default_box": list(default) if default else None,
        "default_index": 0 if default else None,
        "candidates": [{"box": list(default or other), "source": "default"}, {"box": list(other), "source": "sample"}],
        "chosen_index": chosen,
        "switch_score": score,
    }


def with_listwise(row: dict, score: float, chosen: int | None) -> dict:
    return {**row, "listwise_switch_score": score, "listwise_chosen_index": chosen}


class JoinedRowsTests(unittest.TestCase):
    def test_consistency_rules(self) -> None:
        records = pd.DataFrame(
            [
                record(1, "train", 0.0, 0),
                record(2, "train", 5e-7, 1),
                record(3, "train", 0.8, 1),
                record(4, "train", 0.0, None, default=None),
            ]
        )
        truth = {f"RefCOCO:{k}": (0.0, 0.0, 0.5, 0.5) for k in (1, 2, 3, 4)}
        rows = jl.joined_rows(records, truth).set_index("expression_id")
        self.assertEqual(rows.loc["1", "challenger_iou"], rows.loc["1", "baseline_iou"])
        self.assertEqual(rows.loc["2", "score"], 0.0)
        self.assertEqual(rows.loc["2", "challenger_iou"], rows.loc["2", "baseline_iou"])
        self.assertAlmostEqual(rows.loc["3", "score"], 0.8)
        self.assertEqual(rows.loc["3", "challenger_iou"], 0.0)
        self.assertEqual(rows.loc["4", "baseline_iou"], 0.0)


    def test_listwise_columns_follow_the_same_rules(self) -> None:
        records = pd.DataFrame(
            [
                with_listwise(record(1, "train", 0.8, 1), 0.0, 0),
                with_listwise(record(2, "train", 0.0, 0), 0.6, 1),
                with_listwise(record(3, "train", 0.0, None, default=None), 0.0, None),
            ]
        )
        truth = {f"RefCOCO:{k}": (0.0, 0.0, 0.5, 0.5) for k in (1, 2, 3)}
        rows = jl.joined_rows(records, truth).set_index("expression_id")
        self.assertEqual(rows.loc["1", "score_listwise"], 0.0)
        self.assertEqual(rows.loc["1", "challenger_iou_listwise"], rows.loc["1", "baseline_iou"])
        self.assertAlmostEqual(rows.loc["2", "score_listwise"], 0.6)
        self.assertEqual(rows.loc["2", "challenger_iou_listwise"], 0.0)
        self.assertEqual(rows.loc["2", "challenger_iou"], rows.loc["2", "baseline_iou"])
        views = jl.score_views(rows.reset_index())
        self.assertEqual(sorted(views), ["listwise", "pairwise"])
        self.assertEqual(list(views["listwise"].columns), jl.MANIFEST_COLUMNS)

    def test_mixed_listwise_presence_is_rejected(self) -> None:
        records = pd.DataFrame([with_listwise(record(1, "train", 0.0, 0), 0.0, 0), record(2, "train", 0.0, 0)])
        with self.assertRaises(SystemExit):
            jl.joined_rows(records, {f"RefCOCO:{k}": (0.0, 0.0, 0.5, 0.5) for k in (1, 2)})


class OrderEnforcementTests(unittest.TestCase):
    listwise = False

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
        self.root = root
        self.protocol = root / "protocol.json"
        jl.write_protocol(self.protocol, {"protocol_version": "test"})
        self.gen = root / "gen.jsonl"
        rows = [record(k, s, 0.8 if k % 2 else 0.0, 1 if k % 2 else 0) for k, s in enumerate(["train"] * 4 + ["select"] * 4 + ["certify"] * 4 + ["confirm"] * 4)]
        if self.listwise:
            rows = [with_listwise(r, 0.5 if k % 4 == 1 else 0.0, 1 if k % 4 == 1 else 0) for k, r in enumerate(rows)]
        self.gen.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        self.gt = root / "gt.csv"
        pd.DataFrame(
            [{"benchmark": "RefCOCO", "expression_id": str(k), "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5} for k in range(16)]
        ).to_csv(self.gt, index=False)
        self.out = root / "joined"
        self.log = root / "label_join_log.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def commit(self) -> None:
        subprocess.run(["git", "add", "protocol.json"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "p"], cwd=self.root, check=True)

    def join(self, stage: str) -> None:
        jl.cmd_join(
            argparse.Namespace(
                protocol=self.protocol, arm="G1", generation=[self.gen], benchmark="RefCOCO",
                stage=stage, gt_csv=self.gt, out_dir=self.out, log=self.log,
            )
        )

    def test_full_order(self) -> None:
        with self.assertRaises(SystemExit):
            self.join("train")  # protocol not committed yet
        self.commit()
        self.join("train")
        with self.assertRaises(SystemExit):
            self.join("train")  # joins happen once
        with self.assertRaises(SystemExit):
            self.join("select")  # no family yet
        jl.cmd_families(argparse.Namespace(protocol=self.protocol, arm="G1", generation=[self.gen]))
        with self.assertRaises(SystemExit):
            self.join("select")  # family written but not committed
        self.commit()
        self.join("select")
        with self.assertRaises(SystemExit):
            self.join("certify")  # rule not frozen
        jl.cmd_select(argparse.Namespace(protocol=self.protocol, arm="G1", benchmark="RefCOCO", out_dir=self.out, alpha=0.05, delta=0.05))
        self.commit()
        with self.assertRaises(SystemExit):
            self.join("confirm")  # certification first
        self.join("certify")
        self.join("confirm")
        jl.cmd_assemble(argparse.Namespace(protocol=self.protocol, arm="G1", benchmark="RefCOCO", out_dir=self.out, alpha=0.05, delta=0.05))
        stages = [json.loads(line)["stage"] for line in self.log.read_text().splitlines()]
        self.assertEqual(stages, ["train", "select", "certify", "confirm"])
        self.assertTrue((self.out / "G1_RefCOCO_manifest.csv").exists())
        self.assertEqual((self.out / "G1-listwise_RefCOCO_manifest.csv").exists(), self.listwise)


class GatedOrderEnforcementTests(OrderEnforcementTests):
    listwise = True

    def setUp(self) -> None:
        super().setUp()
        jl.write_protocol(self.protocol, {"protocol_version": "test", "require_verifier_gate": True})
        self.verifier = self.root / "v1.pkl"
        self.verifier.write_bytes(b"frozen verifier")
        self.predictions = self.root / "v1_predictions.csv"
        rows = [json.loads(line) for line in self.gen.read_text().splitlines()]
        pd.DataFrame(
            [{"record_key": r["record_key"], "verifier_harm": 0.01} for r in rows if r["chosen_index"] not in (None, 0)]
        ).to_csv(self.predictions, index=False)

    def freeze(self, verifier: Path | None = None) -> None:
        jl.cmd_freeze_verifier(argparse.Namespace(
            protocol=self.protocol, arm="G1", verifier=verifier or self.verifier, predictions=self.predictions, log=self.log,
        ))

    def stage_args(self, **extra):
        return argparse.Namespace(protocol=self.protocol, arm="G1", benchmark="RefCOCO", out_dir=self.out,
                                  alpha=0.05, delta=0.05, verifier_predictions=self.predictions, **extra)

    def test_full_order(self) -> None:
        self.commit()
        self.join("train")
        with self.assertRaises(SystemExit):
            self.freeze()  # no family yet
        jl.cmd_families(argparse.Namespace(protocol=self.protocol, arm="G1", generation=[self.gen]))
        self.commit()
        with self.assertRaises(SystemExit):
            self.join("select")  # V1 not frozen
        self.freeze()
        self.commit()
        protocol = jl.read_protocol(self.protocol)
        self.assertEqual(protocol["gated_families"]["G1-gated"]["RefCOCO"]["taus"], protocol["threshold_families"]["G1"]["RefCOCO"])
        other = self.root / "v1_other.pkl"
        other.write_bytes(b"another verifier")
        with self.assertRaises(SystemExit):
            self.freeze(other)  # V1 cannot be replaced
        self.join("select")
        with self.assertRaises(SystemExit):
            self.freeze()  # selection labels were read
        jl.cmd_select(self.stage_args())
        protocol = jl.read_protocol(self.protocol)
        self.assertIn("nu", protocol["frozen_rules"]["G1-gated"]["RefCOCO"])
        del protocol["frozen_rules"]["G1-gated"]
        jl.write_protocol(self.protocol, protocol)
        self.commit()
        with self.assertRaises(SystemExit):
            self.join("certify")  # gated rule not frozen
        jl.cmd_select(self.stage_args())
        self.commit()
        self.join("certify")
        self.join("confirm")
        jl.cmd_assemble(self.stage_args())
        gated = pd.read_csv(self.out / "G1-gated_RefCOCO_manifest.csv")
        self.assertIn("verifier_harm", gated.columns)
        self.assertTrue((self.out / "G1-listwise_RefCOCO_manifest.csv").exists())

    def test_predictions_must_match_the_frozen_hash(self) -> None:
        self.commit()
        self.join("train")
        jl.cmd_families(argparse.Namespace(protocol=self.protocol, arm="G1", generation=[self.gen]))
        self.freeze()
        self.commit()
        self.join("select")
        self.predictions.write_text("record_key,verifier_harm\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            jl.cmd_select(self.stage_args())


class TransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@example.com"], ["git", "config", "user.name", "t"]):
            subprocess.run(cmd, cwd=root, check=True)
        self.root = root
        self.protocol = root / "protocol.json"
        jl.write_protocol(self.protocol, {
            "protocol_version": "test",
            "frozen_rules": {
                "G1": {"RefCOCO": {"policy": "score_ge_0.5", "threshold": 0.5}},
                "G1-listwise": {"RefCOCO": {"policy": "score_ge_0.3", "threshold": 0.3}},
            },
            "transfer": {"arm": "G3b", "source_arm": "G1", "source_benchmark": "RefCOCO",
                         "target_arm": "G3", "target_benchmark": "RefCLEF", "stage": "confirm"},
        })
        rows = []
        for k in range(12):
            split = ["train", "select", "certify", "confirm"][k % 4]
            row = with_listwise(record(k, split, [0.0, 0.8, 0.4][k % 3], [0, 1, 1][k % 3]), [0.0, 0.2, 0.6][k % 3], [0, 1, 1][k % 3])
            rows.append({**row, "record_key": f"RefCLEF:{k}", "benchmark": "RefCLEF"})
        self.records = pd.DataFrame(rows)
        self.gen = root / "gen.jsonl"
        self.gen.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        self.gt = root / "gt.csv"
        pd.DataFrame([{"benchmark": "RefCLEF", "expression_id": str(k), "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5} for k in range(12)]).to_csv(self.gt, index=False)
        self.out = root / "joined"
        self.log = root / "label_join_log.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def commit_all(self) -> None:
        subprocess.run(["git", "add", "-A"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "c"], cwd=self.root, check=True)

    def join(self, stage: str) -> None:
        jl.cmd_join(argparse.Namespace(protocol=self.protocol, arm="G3", generation=[self.gen], benchmark="RefCLEF",
                                       stage=stage, gt_csv=self.gt, out_dir=self.out, log=self.log))

    def evaluate(self) -> None:
        jl.cmd_transfer_evaluate(argparse.Namespace(protocol=self.protocol, out_dir=self.out, log=self.log, alpha=0.05, delta=0.05))

    def test_transfer_comes_first_and_is_scored_once_labels_arrive(self) -> None:
        self.commit_all()
        with self.assertRaises(SystemExit):
            self.join("train")  # transfer not applied yet
        jl.cmd_transfer_apply(argparse.Namespace(protocol=self.protocol, generation=[self.gen], out_dir=self.out, log=self.log))
        routes = pd.read_csv(self.out / "G3b_RefCLEF_routes.csv")
        self.assertEqual(len(routes), 3)  # confirmation records only
        self.assertEqual(list(routes["routed_pairwise"]), [False, False, True])  # keys 11, 3, 7 as strings; only 7 scores 0.8
        with self.assertRaises(SystemExit):
            jl.cmd_transfer_apply(argparse.Namespace(protocol=self.protocol, generation=[self.gen], out_dir=self.out, log=self.log))
        self.commit_all()
        self.join("train")  # allowed after the transfer
        # Write the confirmation join the G3 arm would produce, then score the transfer.
        confirm = jl.joined_rows(self.records[self.records["split"] == "confirm"], jl.load_truth(self.gt))
        confirm[jl.MANIFEST_COLUMNS + jl.LISTWISE_COLUMNS].to_csv(self.out / "G3_RefCLEF_confirm.csv", index=False)
        self.evaluate()
        summary = json.loads((self.out / "G3b_RefCLEF_transfer_summary.json").read_text())
        self.assertEqual(summary["pairwise"]["route_count"], 1)
        self.assertIn("listwise", summary)

    def test_tampered_routes_are_rejected(self) -> None:
        self.commit_all()
        jl.cmd_transfer_apply(argparse.Namespace(protocol=self.protocol, generation=[self.gen], out_dir=self.out, log=self.log))
        routes = self.out / "G3b_RefCLEF_routes.csv"
        routes.write_text(routes.read_text().replace("False", "True"))
        self.commit_all()
        confirm = jl.joined_rows(self.records[self.records["split"] == "confirm"], jl.load_truth(self.gt))
        confirm[jl.MANIFEST_COLUMNS + jl.LISTWISE_COLUMNS].to_csv(self.out / "G3_RefCLEF_confirm.csv", index=False)
        with self.assertRaises(SystemExit):
            self.evaluate()


class ListwiseOrderEnforcementTests(OrderEnforcementTests):
    listwise = True

    def test_certification_needs_both_frozen_rules(self) -> None:
        self.commit()
        self.join("train")
        jl.cmd_families(argparse.Namespace(protocol=self.protocol, arm="G1", generation=[self.gen]))
        protocol = jl.read_protocol(self.protocol)
        self.assertIn("G1-listwise", protocol["threshold_families"])
        listwise_family = protocol["threshold_families"].pop("G1-listwise")
        jl.write_protocol(self.protocol, protocol)
        self.commit()
        with self.assertRaises(SystemExit):
            self.join("select")  # listwise family missing
        protocol["threshold_families"]["G1-listwise"] = listwise_family
        jl.write_protocol(self.protocol, protocol)
        self.commit()
        self.join("select")
        jl.cmd_select(argparse.Namespace(protocol=self.protocol, arm="G1", benchmark="RefCOCO", out_dir=self.out, alpha=0.05, delta=0.05))
        protocol = jl.read_protocol(self.protocol)
        self.assertIn("RefCOCO", protocol["frozen_rules"]["G1-listwise"])
        del protocol["frozen_rules"]["G1-listwise"]
        jl.write_protocol(self.protocol, protocol)
        self.commit()
        with self.assertRaises(SystemExit):
            self.join("certify")  # listwise rule not frozen


if __name__ == "__main__":
    unittest.main()
