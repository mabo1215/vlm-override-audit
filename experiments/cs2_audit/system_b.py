"""System B: a second, fully specified grounding system (Design.md section 15.3).

System B has the same structure as the audited System A -- a VLM's own direct
answer is the default box, a re-ranker may replace it with another candidate
-- but every element is fixed here and written to the protocol file:

* default box: greedy answer of InternVL3-8B to its referring-grounding prompt;
* candidates: the default, k sampled answers to the same prompt, and the top
  GroundingDINO proposals for the expression, merged by a fixed rule;
* switch score: pairwise A/B preferences of the same VLM, asked in both orders,
  antisymmetrized and averaged per candidate;
* secondary score (author decision 2026-10-16, registered before any System B
  label join): a listwise choice among all candidates, asked in the merge
  order and in reverse order, averaged per candidate.  It is selected, frozen,
  certified and confirmed alongside the pairwise rule and reported separately.

Generation never reads ground truth.  IoU is added later by
``join_system_b_labels.py``.  Output is append-only JSONL, flushed and fsynced
per record; completed records are skipped on restart.

The pure functions at the top (parsing, merging, aggregation, switch rule,
seeds) carry the protocol and are unit-tested on CPU.  ``SystemB`` wraps the
models and needs the GPU host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

Box = tuple[float, float, float, float]

PROTOCOL_VERSION = "system_b_v1"
GROUNDING_PROMPT = (
    "Please provide the bounding box coordinate of the region this sentence describes: <ref>{text}</ref>"
)
PAIRWISE_PROMPT = (
    "Region A: <box>[[{a}]]</box>\n"
    "Region B: <box>[[{b}]]</box>\n"
    "(coordinates are normalized to 0-1000)\n"
    "Which region better shows '{text}'? Answer with only 'A' or 'B'."
)
LISTWISE_PROMPT = (
    "{regions}\n"
    "(coordinates are normalized to 0-1000)\n"
    "Which region best shows '{text}'? Answer with only the region letter."
)
LISTWISE_LABELS = "ABCDEF"
COORDINATE_SCALE = 1000.0
SAMPLE_COUNT = 8
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9
MAX_NEW_TOKENS = 64
DETECTOR_BOX_THRESHOLD = 0.25
DETECTOR_TEXT_THRESHOLD = 0.25
DETECTOR_TOP_N = 5
MERGE_IOU = 0.9
MAX_CANDIDATES = 6
TIE_TOLERANCE = 1e-6
SEED_BASE = 20271016

_BOX_RE = re.compile(
    r"\[\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\]"
)
_FOUR_NUMBERS_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
)


# ---------------------------------------------------------------------------
# Protocol functions (CPU, unit-tested)
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol_settings() -> dict[str, object]:
    """Every generation setting that the protocol file must freeze."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "grounding_prompt": GROUNDING_PROMPT,
        "grounding_prompt_sha256": sha256_text(GROUNDING_PROMPT),
        "pairwise_prompt": PAIRWISE_PROMPT,
        "pairwise_prompt_sha256": sha256_text(PAIRWISE_PROMPT),
        "listwise_prompt": LISTWISE_PROMPT,
        "listwise_prompt_sha256": sha256_text(LISTWISE_PROMPT),
        "listwise_region_line": "Region {letter}: <box>[[{x1}, {y1}, {x2}, {y2}]]</box>, letters A-F in listing order",
        "coordinate_frame": "model output in [0, 1000] on both axes, divided by 1000 and clipped to [0, 1]",
        "default_decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "sample_decoding": {
            "do_sample": True,
            "num_return_sequences": SAMPLE_COUNT,
            "temperature": SAMPLE_TEMPERATURE,
            "top_p": SAMPLE_TOP_P,
            "top_k": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": f"{SEED_BASE} + int(sha256(expression_id)[:8], 16)",
        },
        "detector": {
            "query": "expression stripped, trailing period removed, lower-cased, then '.' appended",
            "box_threshold": DETECTOR_BOX_THRESHOLD,
            "text_threshold": DETECTOR_TEXT_THRESHOLD,
            "top_n": DETECTOR_TOP_N,
        },
        "merge": {
            "order": "default, samples in sample order, detector boxes by descending score",
            "drop_if_iou_at_least": MERGE_IOU,
            "max_candidates_including_default": MAX_CANDIDATES,
        },
        "score": {
            "pairwise": "log p('A') - log p('B') at the next token, label variants pooled by logsumexp",
            "orders": "both; value = 0.5 * (forward - reverse)",
            "aggregation": "row mean over all K entries including the zero diagonal, then centred",
            "switch_score": "theta[best] - theta[default]; best = argmax theta, first index on ties",
            "tie_tolerance": TIE_TOLERANCE,
            "no_switch_score": 0.0,
            "default_parse_failure": "empty default (IoU 0) and switch score 0",
        },
        "score_listwise": {
            "status": "secondary preregistered score (author decision 2026-10-16); the primary claim uses the pairwise score",
            "orders": "two listings: merge order and reverse merge order; none when K = 1",
            "per_order": "log p(letter) pooled over label variants, renormalized over the K shown letters",
            "aggregation": "mean over the two listings of each candidate's renormalized log probability, then centred",
            "switch_score": "phi[best] - phi[default]; best = argmax phi, first index on ties; same tie tolerance",
            "audit": "own label-free threshold family; selected, frozen, certified and confirmed in the same joins as the pairwise rule; never substituted for it",
        },
    }


def example_seed(expression_id: str, base: int = SEED_BASE) -> int:
    return base + int(sha256_text(str(expression_id))[:8], 16)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def clip_box(box: Sequence[float]) -> Box | None:
    """Clip to [0, 1]; return None for a degenerate box."""

    x1, y1, x2, y2 = (min(1.0, max(0.0, float(v))) for v in box)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def parse_model_box(text: str) -> Box | None:
    """Parse the first ``[[x1, y1, x2, y2]]`` box in the model's 0-1000 frame."""

    match = _BOX_RE.search(text) or _FOUR_NUMBERS_RE.search(text)
    if match is None:
        return None
    values = [float(v) / COORDINATE_SCALE for v in match.groups()]
    return clip_box(values)


def to_model_coordinates(box: Sequence[float]) -> str:
    """Format a normalized box in the model's integer 0-1000 frame."""

    return ", ".join(str(int(round(min(1.0, max(0.0, v)) * COORDINATE_SCALE))) for v in box)


@dataclass(frozen=True)
class Candidate:
    box: Box
    source: str
    detector_score: float | None = None


def merge_candidates(
    default: Box | None,
    samples: Iterable[Box | None],
    detections: Iterable[tuple[Sequence[float], float]],
    *,
    drop_iou: float = MERGE_IOU,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """Merge boxes in the fixed order; the default, if any, is always first."""

    ordered: list[Candidate] = []
    if default is not None:
        ordered.append(Candidate(default, "default"))
    for box in samples:
        if box is not None:
            ordered.append(Candidate(box, "sample"))
    for raw, score in sorted(detections, key=lambda item: -float(item[1])):
        box = clip_box(raw)
        if box is not None:
            ordered.append(Candidate(box, "detector", float(score)))
    kept: list[Candidate] = []
    for candidate in ordered:
        if len(kept) >= max_candidates:
            break
        if candidate.source != "default" and any(box_iou(candidate.box, k.box) >= drop_iou for k in kept):
            continue
        kept.append(candidate)
    return kept


def aggregate_pairwise(comparisons: dict[tuple[int, int], float], size: int) -> list[float]:
    """Row mean of the antisymmetric preference matrix (diagonal included), centred."""

    if size <= 1:
        return [0.0] * size
    totals = [0.0] * size
    for (i, j), value in comparisons.items():
        totals[i] += value
        totals[j] -= value
    scores = [total / size for total in totals]
    mean = sum(scores) / size
    return [score - mean for score in scores]


def listwise_orders(size: int) -> list[list[int]]:
    """The two listings shown to the model: merge order and reverse merge order."""

    if size <= 1:
        return []
    forward = list(range(size))
    return [forward, forward[::-1]]


def listwise_prompt(boxes: Sequence[Box], order: Sequence[int], text: str) -> str:
    regions = "\n".join(
        f"Region {LISTWISE_LABELS[position]}: <box>[[{to_model_coordinates(boxes[index])}]]</box>"
        for position, index in enumerate(order)
    )
    return LISTWISE_PROMPT.format(regions=regions, text=text)


def aggregate_listwise(orders: Sequence[Sequence[int]], label_log_probs: Sequence[Sequence[float]]) -> list[float]:
    """Mean renormalized log probability per candidate over the listings, centred.

    ``label_log_probs[k][p]`` is the log probability of the letter at position
    ``p`` of listing ``k``; listing ``k`` shows candidate ``orders[k][p]`` there.
    """

    import math

    if not orders:
        return []
    size = len(orders[0])
    totals = [0.0] * size
    for order, raw in zip(orders, label_log_probs):
        values = [float(v) for v in raw[:size]]
        peak = max(values)
        norm = peak + math.log(sum(math.exp(v - peak) for v in values))
        for position, candidate in enumerate(order):
            totals[candidate] += values[position] - norm
    phi = [total / len(orders) for total in totals]
    mean = sum(phi) / size
    return [value - mean for value in phi]


def switch_decision(theta: Sequence[float], default_index: int | None, tie: float = TIE_TOLERANCE) -> tuple[int | None, float]:
    """Return (chosen index, switch score).  Ties within ``tie`` keep the default."""

    if default_index is None or len(theta) <= 1:
        return default_index, 0.0
    best = max(range(len(theta)), key=lambda i: (theta[i], -i))
    margin = float(theta[best] - theta[default_index])
    if best == default_index or margin < tie:
        return default_index, 0.0
    return best, margin


def detector_query(text: str) -> str:
    return text.strip().rstrip(".").lower() + "."


# ---------------------------------------------------------------------------
# Resumable output
# ---------------------------------------------------------------------------


def completed_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from an interrupted write is regenerated
            if "error" not in record:
                keys.add(str(record["record_key"]))
    return keys


def append_record(handle, record: dict[str, object]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def record_key(benchmark: str, expression_id: str) -> str:
    return f"{benchmark}:{expression_id}"


# ---------------------------------------------------------------------------
# Prefix-cached pairwise scoring (model-agnostic; unit-tested with a tiny LM)
# ---------------------------------------------------------------------------


def label_log_probs_with_prefix_cache(
    model, prefix_inputs: dict, suffixes: Sequence, label_sets: Sequence[Sequence[Sequence[int]]]
) -> list[list[float]]:
    """Log probability of each label after each suffix, encoding the prefix once.

    ``prefix_inputs`` are the model inputs up to and including the image;
    each suffix is a 1-D tensor of the remaining token ids and
    ``label_sets[k]`` lists the labels read after suffix ``k``, each label
    given by its token-id variants (pooled by logsumexp).  The key/value cache
    of the prefix is reused and cropped back after every suffix.
    """

    import torch

    with torch.no_grad():
        prefix_out = model(**prefix_inputs, use_cache=True)
        cache = prefix_out.past_key_values
        prefix_len = int(prefix_inputs["input_ids"].shape[1])
        values: list[list[float]] = []
        for suffix, labels in zip(suffixes, label_sets):
            out = model(input_ids=suffix.view(1, -1).to(prefix_inputs["input_ids"].device), past_key_values=cache, use_cache=True)
            log_probs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            values.append([float(torch.logsumexp(log_probs[list(ids)], dim=0)) for ids in labels])
            cache.crop(prefix_len)
    return values


def pairwise_log_odds_with_prefix_cache(model, prefix_inputs: dict, suffixes: Sequence, label_ids: Sequence[Sequence[int]]) -> list[float]:
    """Return log p(A) - log p(B) after each suffix, encoding the prefix once."""

    values = label_log_probs_with_prefix_cache(model, prefix_inputs, suffixes, [label_ids] * len(suffixes))
    return [a - b for a, b in values]


# ---------------------------------------------------------------------------
# GPU wrapper
# ---------------------------------------------------------------------------


@dataclass
class SystemB:
    model_path: str
    detector_path: str
    device: str = "cuda"
    max_patches: int = 6
    processor_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.det_processor = AutoProcessor.from_pretrained(self.detector_path)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(self.detector_path).to(self.device).eval()
        self.processor_kwargs = {"crop_to_patches": True, "max_patches": self.max_patches}
        tokenizer = self.processor.tokenizer
        self.label_ids = [self._label_variants(tokenizer, "A"), self._label_variants(tokenizer, "B")]
        self.listwise_label_ids = [self._label_variants(tokenizer, letter) for letter in LISTWISE_LABELS]

    @staticmethod
    def _label_variants(tokenizer, label: str) -> list[int]:
        ids = set()
        for variant in (label, " " + label):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        if not ids:
            raise ValueError(f"label {label!r} is not a single token")
        return sorted(ids)

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _inputs(self, image, text: str) -> dict:
        inputs = self.processor(images=[image], text=[text], return_tensors="pt", **self.processor_kwargs)
        return {k: (v.to(self.device, dtype=self.torch.bfloat16) if k == "pixel_values" else v.to(self.device)) for k, v in inputs.items()}

    def detect(self, image, text: str) -> list[tuple[list[float], float]]:
        torch = self.torch
        width, height = image.size
        inputs = self.det_processor(images=image, text=detector_query(text), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector(**inputs)
        try:
            result = self.det_processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=DETECTOR_BOX_THRESHOLD,
                text_threshold=DETECTOR_TEXT_THRESHOLD, target_sizes=[(height, width)],
            )[0]
        except TypeError:
            result = self.det_processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_threshold=DETECTOR_BOX_THRESHOLD,
                text_threshold=DETECTOR_TEXT_THRESHOLD, target_sizes=[(height, width)],
            )[0]
        boxes = result["boxes"].float().cpu().tolist()
        scores = result["scores"].float().cpu().tolist()
        normalized = [([b[0] / width, b[1] / height, b[2] / width, b[3] / height], s) for b, s in zip(boxes, scores)]
        normalized.sort(key=lambda item: -item[1])
        return normalized[:DETECTOR_TOP_N]

    def answers(self, image, text: str, seed: int) -> tuple[str, list[str]]:
        torch = self.torch
        inputs = self._inputs(image, self._chat_text(GROUNDING_PROMPT.format(text=text)))
        start = inputs["input_ids"].shape[1]
        decode = self.processor.tokenizer.batch_decode
        with torch.no_grad():
            greedy = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sampled = self.model.generate(
                **inputs, do_sample=True, temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=0,
                num_return_sequences=SAMPLE_COUNT, max_new_tokens=MAX_NEW_TOKENS,
            )
        default_text = decode(greedy[:, start:], skip_special_tokens=False)[0]
        sample_texts = decode(sampled[:, start:], skip_special_tokens=False)
        return default_text, sample_texts

    def score(self, image, text: str, boxes: Sequence[Box]) -> dict[str, object]:
        """Pairwise and listwise scores of the candidates, sharing one image prefix."""

        torch = self.torch
        size = len(boxes)
        if size <= 1:
            return {"theta": [0.0] * size, "pairwise": {}, "phi": [0.0] * size, "listwise": {}, "passes": 0, "cached": True}
        prompts, kinds, label_sets = [], [], []
        for i in range(size):
            for j in range(i + 1, size):
                for first, second in ((i, j), (j, i)):
                    prompts.append(PAIRWISE_PROMPT.format(a=to_model_coordinates(boxes[first]), b=to_model_coordinates(boxes[second]), text=text))
                    kinds.append(("pair", (first, second)))
                    label_sets.append(self.label_ids)
        for order in listwise_orders(size):
            prompts.append(listwise_prompt(boxes, order, text))
            kinds.append(("list", tuple(order)))
            label_sets.append(self.listwise_label_ids[:size])
        full_texts = [self._chat_text(p) for p in prompts]
        cut = full_texts[0].index(prompts[0])
        prefix_text = full_texts[0][:cut]
        prefix_inputs = self._inputs(image, prefix_text)
        tokenizer = self.processor.tokenizer
        prefix_ids = prefix_inputs["input_ids"][0].tolist()
        suffixes = []
        cached = True
        for full in full_texts:
            if full[:cut] != prefix_text:
                cached = False
                break
            suffixes.append(torch.tensor(tokenizer.encode(full[cut:], add_special_tokens=False), dtype=torch.long))
        if cached:
            # The prefix tokenization must be a prefix of the full tokenization.
            check = self._inputs(image, full_texts[0])["input_ids"][0].tolist()
            cached = check[: len(prefix_ids)] == prefix_ids and check[len(prefix_ids):] == suffixes[0].tolist()
        if cached:
            raw = label_log_probs_with_prefix_cache(self.model, prefix_inputs, suffixes, label_sets)
        else:
            raw = []
            for full, labels in zip(full_texts, label_sets):
                inputs = self._inputs(image, full)
                with torch.no_grad():
                    logits = self.model(**inputs).logits[0, -1].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                raw.append([float(torch.logsumexp(log_probs[ids], 0)) for ids in labels])
        directional = {key: values[0] - values[1] for (kind, key), values in zip(kinds, raw) if kind == "pair"}
        comparisons = {}
        recorded = {}
        for i in range(size):
            for j in range(i + 1, size):
                value = 0.5 * (directional[(i, j)] - directional[(j, i)])
                comparisons[(i, j)] = value
                recorded[f"{i}-{j}"] = value
        orders = [list(key) for kind, key in kinds if kind == "list"]
        listed = [values for (kind, _), values in zip(kinds, raw) if kind == "list"]
        return {
            "theta": aggregate_pairwise(comparisons, size),
            "pairwise": recorded,
            "phi": aggregate_listwise(orders, listed),
            "listwise": {"orders": orders, "letter_log_probs": listed},
            "passes": len(prompts) + 1 if cached else len(prompts),
            "cached": cached,
        }

    def run_record(self, image, row: dict[str, object]) -> dict[str, object]:
        torch = self.torch
        t0 = time.time()
        text = str(row["text"])
        seed = example_seed(str(row["expression_id"]))
        default_text, sample_texts = self.answers(image, text, seed)
        default_box = parse_model_box(default_text)
        sample_boxes = [parse_model_box(t) for t in sample_texts]
        detections = self.detect(image, text)
        candidates = merge_candidates(default_box, sample_boxes, detections)
        scores = self.score(image, text, [c.box for c in candidates])
        default_index = 0 if default_box is not None else None
        best, score = switch_decision(scores["theta"], default_index)
        listwise_best, listwise_score = switch_decision(scores["phi"], default_index)
        return {
            **{k: row[k] for k in ("benchmark", "split", "expression_id", "image_id", "file_name", "text", "image_path")},
            "record_key": record_key(str(row["benchmark"]), str(row["expression_id"])),
            "protocol_version": PROTOCOL_VERSION,
            "image_size": list(image.size),
            "seed": seed,
            "default_text": default_text,
            "default_box": list(default_box) if default_box else None,
            "sample_texts": sample_texts,
            "sample_boxes": [list(b) if b else None for b in sample_boxes],
            "detections": [{"box": list(b), "score": s} for b, s in detections],
            "candidates": [{"box": list(c.box), "source": c.source, "detector_score": c.detector_score} for c in candidates],
            "pairwise": scores["pairwise"],
            "theta": scores["theta"],
            "default_index": default_index,
            "chosen_index": best,
            "switch_score": score,
            "listwise": scores["listwise"],
            "phi": scores["phi"],
            "listwise_chosen_index": listwise_best,
            "listwise_switch_score": listwise_score,
            "passes": scores["passes"],
            "prefix_cached": scores["cached"],
            "seconds": round(time.time() - t0, 3),
            "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_ZIP_INDEX: dict[Path, tuple[object, dict[str, str]]] = {}


def load_image(row: dict[str, object], image_root: Path | None):
    """Open an image by absolute path, or relative to a directory or a zip archive."""

    from PIL import Image

    path = Path(str(row["image_path"]))
    if path.is_absolute() or image_root is None:
        return Image.open(path).convert("RGB")
    if image_root.suffix == ".zip":
        import io
        import zipfile

        if image_root not in _ZIP_INDEX:
            archive = zipfile.ZipFile(image_root)
            _ZIP_INDEX[image_root] = (archive, {Path(n).name: n for n in archive.namelist() if not n.endswith("/")})
        archive, index = _ZIP_INDEX[image_root]
        with archive.open(index[path.name]) as handle:
            return Image.open(io.BytesIO(handle.read())).convert("RGB")
    return Image.open(image_root / path).convert("RGB")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, help="CSV from prepare_system_b_inputs.py (no labels)")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--detector-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-patches", type=int, default=6)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--splits", default="", help="comma-separated split filter, e.g. train for pilots")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--print-protocol", action="store_true")
    args = parser.parse_args()

    if args.print_protocol:
        print(json.dumps(protocol_settings(), indent=2, sort_keys=True))
        return
    frame = pd.read_csv(args.records, dtype={"expression_id": str})
    forbidden = [c for c in frame.columns if re.search(r"iou|bbox|gt|help|harm|correct", c, re.IGNORECASE)]
    if forbidden:
        raise SystemExit(f"generation input must not carry label columns: {forbidden}")
    if args.splits:
        frame = frame[frame["split"].isin(args.splits.split(","))]
    frame = frame.sort_values(["benchmark", "expression_id"]).reset_index(drop=True)
    frame = frame.iloc[args.shard_index :: args.num_shards]
    if args.max_records is not None:
        frame = frame.head(args.max_records)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.output_jsonl)
    todo = [row for row in frame.to_dict("records") if record_key(row["benchmark"], row["expression_id"]) not in done]
    print(json.dumps({"records": len(frame), "already_done": len(frame) - len(todo), "todo": len(todo)}), flush=True)
    if not todo:
        return
    system = SystemB(args.model_path, args.detector_path, device=args.device, max_patches=args.max_patches)
    with args.output_jsonl.open("a", encoding="utf-8") as handle:
        for n, row in enumerate(todo, 1):
            key = record_key(row["benchmark"], row["expression_id"])
            try:
                record = system.run_record(load_image(row, args.image_root), row)
            except Exception as exc:  # recorded and retried on the next restart
                record = {"record_key": key, "error": repr(exc)}
            append_record(handle, record)
            if n % 25 == 0 or n == len(todo):
                print(json.dumps({"done": n, "of": len(todo), "last_seconds": record.get("seconds")}), flush=True)


if __name__ == "__main__":
    main()
