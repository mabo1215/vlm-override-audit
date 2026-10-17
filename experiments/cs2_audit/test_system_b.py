from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.cs2_audit import system_b as sb


class ParseTests(unittest.TestCase):
    def test_parses_model_frame(self) -> None:
        box = sb.parse_model_box("<ref>the dog</ref><box>[[100, 200, 300, 400]]</box>")
        self.assertEqual(box, (0.1, 0.2, 0.3, 0.4))

    def test_clips_and_rejects_degenerate(self) -> None:
        self.assertEqual(sb.parse_model_box("[[900, 0, 1100, 50]]"), (0.9, 0.0, 1.0, 0.05))
        self.assertIsNone(sb.parse_model_box("[[300, 300, 300, 400]]"))
        self.assertIsNone(sb.parse_model_box("no box here"))

    def test_round_trip_formatting(self) -> None:
        self.assertEqual(sb.to_model_coordinates((0.1, 0.2, 0.30049, 1.2)), "100, 200, 300, 1000")


class MergeTests(unittest.TestCase):
    def test_order_dedup_and_cap(self) -> None:
        default = (0.1, 0.1, 0.5, 0.5)
        samples = [(0.1, 0.1, 0.5, 0.5), (0.6, 0.6, 0.9, 0.9), None]
        detections = [([0.0, 0.0, 0.2, 0.2], 0.3), ([0.6, 0.6, 0.9, 0.9], 0.9), ([0.3, 0.3, 0.7, 0.7], 0.5)]
        kept = sb.merge_candidates(default, samples, detections)
        self.assertEqual([c.source for c in kept], ["default", "sample", "detector", "detector"])
        self.assertEqual(kept[0].box, default)
        capped = sb.merge_candidates(default, samples, detections, max_candidates=2)
        self.assertEqual(len(capped), 2)

    def test_no_default(self) -> None:
        kept = sb.merge_candidates(None, [], [([0.1, 0.1, 0.4, 0.4], 0.5)])
        self.assertEqual([c.source for c in kept], ["detector"])


class ScoreTests(unittest.TestCase):
    def test_aggregation_recovers_consistent_differences(self) -> None:
        theta = [0.0, 1.0, -0.5]
        comparisons = {(i, j): theta[i] - theta[j] for i in range(3) for j in range(i + 1, 3)}
        scores = sb.aggregate_pairwise(comparisons, 3)
        self.assertAlmostEqual(sum(scores), 0.0)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(scores[i] - scores[j], theta[i] - theta[j])

    def test_switch_rule_and_ties(self) -> None:
        self.assertEqual(sb.switch_decision([0.0, 0.3, 0.1], 0), (1, 0.3))
        self.assertEqual(sb.switch_decision([0.2, 0.2 + 5e-7], 0), (0, 0.0))
        self.assertEqual(sb.switch_decision([0.4, 0.1], 0), (0, 0.0))
        self.assertEqual(sb.switch_decision([0.4, 0.1], None), (None, 0.0))

    def test_seed_is_deterministic(self) -> None:
        self.assertEqual(sb.example_seed("123"), sb.example_seed("123"))
        self.assertNotEqual(sb.example_seed("123"), sb.example_seed("124"))

    def test_protocol_hashes_match_prompts(self) -> None:
        settings = sb.protocol_settings()
        self.assertEqual(settings["pairwise_prompt_sha256"], sb.sha256_text(sb.PAIRWISE_PROMPT))
        self.assertEqual(settings["score"]["tie_tolerance"], sb.TIE_TOLERANCE)


class ListwiseTests(unittest.TestCase):
    def test_orders_and_prompt(self) -> None:
        self.assertEqual(sb.listwise_orders(1), [])
        self.assertEqual(sb.listwise_orders(3), [[0, 1, 2], [2, 1, 0]])
        boxes = [(0.1, 0.2, 0.3, 0.4), (0.5, 0.5, 0.9, 0.9)]
        prompt = sb.listwise_prompt(boxes, [1, 0], "the dog")
        self.assertTrue(prompt.startswith("Region A: <box>[[500, 500, 900, 900]]</box>\nRegion B: <box>[[100, 200, 300, 400]]</box>"))
        self.assertIn("'the dog'", prompt)

    def test_aggregation_renormalizes_and_cancels_position_bias(self) -> None:
        import math

        # True preference: candidate 2 > 0 > 1; the model adds +1.0 to whatever is shown first.
        true = [0.5, -1.0, 1.5]
        orders = sb.listwise_orders(3)
        raw = [[true[c] + (1.0 if pos == 0 else 0.0) - 7.0 for pos, c in enumerate(order)] for order in orders]
        phi = sb.aggregate_listwise(orders, raw)
        self.assertAlmostEqual(sum(phi), 0.0)
        self.assertEqual(max(range(3), key=lambda i: phi[i]), 2)
        self.assertGreater(phi[0], phi[1])
        # A constant offset in the raw log probabilities changes nothing.
        shifted = sb.aggregate_listwise(orders, [[v + 3.0 for v in r] for r in raw])
        for a, b in zip(phi, shifted):
            self.assertAlmostEqual(a, b)
        self.assertEqual(sb.switch_decision(phi, 0)[0], 2)
        self.assertEqual(sb.aggregate_listwise([], []), [])
        self.assertTrue(all(math.isfinite(v) for v in phi))

    def test_protocol_records_listwise_prompt(self) -> None:
        settings = sb.protocol_settings()
        self.assertEqual(settings["listwise_prompt_sha256"], sb.sha256_text(sb.LISTWISE_PROMPT))
        self.assertIn("secondary", settings["score_listwise"]["status"])
        self.assertGreaterEqual(len(sb.LISTWISE_LABELS), sb.MAX_CANDIDATES)


class ResumeTests(unittest.TestCase):
    def test_completed_keys_skip_errors_and_torn_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                sb.append_record(handle, {"record_key": "RefCOCO:1", "switch_score": 0.0})
                sb.append_record(handle, {"record_key": "RefCOCO:2", "error": "boom"})
                handle.write('{"record_key": "RefCOCO:3", "sw')
            self.assertEqual(sb.completed_keys(path), {"RefCOCO:1"})


class PrefixCacheTests(unittest.TestCase):
    def test_cached_log_odds_match_full_forward(self) -> None:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        )
        model = Qwen2ForCausalLM(config).eval()
        prefix = torch.randint(0, 64, (1, 20))
        suffixes = [torch.randint(0, 64, (n,)) for n in (5, 9, 3)]
        labels = [[7, 8], [9]]
        cached = sb.pairwise_log_odds_with_prefix_cache(
            model, {"input_ids": prefix, "attention_mask": torch.ones_like(prefix)}, suffixes, labels
        )
        for value, suffix in zip(cached, suffixes):
            full = torch.cat([prefix[0], suffix]).view(1, -1)
            with torch.no_grad():
                log_probs = torch.log_softmax(model(input_ids=full).logits[0, -1].float(), dim=-1)
            expected = torch.logsumexp(log_probs[labels[0]], 0) - torch.logsumexp(log_probs[labels[1]], 0)
            self.assertAlmostEqual(value, float(expected), places=4)


    def test_cached_label_log_probs_match_full_forward(self) -> None:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(1)
        config = Qwen2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        )
        model = Qwen2ForCausalLM(config).eval()
        prefix = torch.randint(0, 64, (1, 12))
        suffixes = [torch.randint(0, 64, (n,)) for n in (4, 7)]
        label_sets = [[[3], [4, 5]], [[10], [11], [12], [13]]]
        cached = sb.label_log_probs_with_prefix_cache(
            model, {"input_ids": prefix, "attention_mask": torch.ones_like(prefix)}, suffixes, label_sets
        )
        for values, suffix, labels in zip(cached, suffixes, label_sets):
            full = torch.cat([prefix[0], suffix]).view(1, -1)
            with torch.no_grad():
                log_probs = torch.log_softmax(model(input_ids=full).logits[0, -1].float(), dim=-1)
            self.assertEqual(len(values), len(labels))
            for value, ids in zip(values, labels):
                self.assertAlmostEqual(value, float(torch.logsumexp(log_probs[ids], 0)), places=4)


class CliGuardTests(unittest.TestCase):
    def test_print_protocol_is_json(self) -> None:
        text = json.dumps(sb.protocol_settings())
        self.assertIn("pairwise_prompt_sha256", json.loads(text))


if __name__ == "__main__":
    unittest.main()
