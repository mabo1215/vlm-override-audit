"""Generate a second, real candidate set with OWLv2.

This command consumes only image pixels and referring-expression metadata.
Ground-truth boxes are deliberately not loaded here.  The output is a JSONL
candidate artifact that can be joined to an audit manifest afterwards.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import pandas as pd


def _sentence_text(value: object, index: int) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        item = value[min(index, len(value) - 1)]
        if isinstance(item, dict):
            return str(item.get("sent") or item.get("raw") or "")
        return str(item)
    return "" if value is None else str(value)


def load_metadata(parquet_glob: str, sentence_index: int) -> dict[str, dict[str, object]]:
    paths = sorted(glob.glob(parquet_glob, recursive=True)) if any(
        char in parquet_glob for char in "*?["
    ) else [parquet_glob]
    metadata: dict[str, dict[str, object]] = {}
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["ref_id", "image_id", "file_name", "sentences", "image_path"],
        )
        for _, row in frame.iterrows():
            key = str(row["ref_id"])
            metadata.setdefault(
                key,
                {
                    "image_id": int(row["image_id"]),
                    "file_name": str(row["file_name"]),
                    "image_path": str(row["image_path"] or ""),
                    "sentence": _sentence_text(row["sentences"], sentence_index),
                },
            )
    if not metadata:
        raise FileNotFoundError(f"no metadata rows resolved from {parquet_glob}")
    return metadata


def _resolve_images(image_root: Path) -> dict[str, Path]:
    if not image_root.exists():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    return {
        path.name: path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def _normalized_image_name(file_name: str) -> str:
    """Remove the referring-expression suffix used by some COCO exports."""
    # Keep the canonical 12-digit COCO image id; expression suffixes in the
    # exported parquet are short numeric fields such as ``_0`` or ``_8``.
    return re.sub(r"_[0-9]{1,3}(?=\.[^.]+$)", "", file_name)


def _text_max_length(model: object) -> int | None:
    """Return the model's actual text context limit when it is declared."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "max_position_embeddings", None)
    if isinstance(value, int) and value > 0:
        return value
    return None


def _normal_box(box: object, width: int, height: int) -> list[float]:
    values = [float(value) for value in box]
    return [
        max(0.0, min(1.0, values[0] / width)),
        max(0.0, min(1.0, values[1] / height)),
        max(0.0, min(1.0, values[2] / width)),
        max(0.0, min(1.0, values[3] / height)),
    ]


def generate(args: argparse.Namespace) -> dict[str, object]:
    # Heavy imports stay inside the execution path so the metadata and CLI
    # remain usable in the CPU-only audit environment.
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    torch.set_grad_enabled(False)
    processor = Owlv2Processor.from_pretrained(args.model_path)
    model = Owlv2ForObjectDetection.from_pretrained(args.model_path).eval().to(device)
    metadata = load_metadata(args.parquet_glob, args.sentence_index)
    image_map = _resolve_images(Path(args.image_root))
    frame = pd.read_csv(args.manifest_csv)
    required = {"split", "image_id", "expression_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")
    frame = frame.sort_values(["split", "image_id", "expression_id"], kind="mergesort")
    if args.num_shards > 1:
        frame = frame.iloc[args.shard_index :: args.num_shards]
    if args.max_rows is not None:
        frame = frame.head(args.max_rows)
    rows = frame.to_dict("records")
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_count = 0
    processed = 0
    started = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            images = []
            valid = []
            for row in batch:
                key = str(row["expression_id"])
                meta = metadata.get(key)
                if meta is None:
                    raise KeyError(f"expression_id {key!r} is absent from the parquet metadata")
                image_path = Path(str(meta["image_path"]))
                if not image_path.is_file():
                    file_name = str(meta["file_name"])
                    normalized_name = _normalized_image_name(file_name)
                    image_path = image_map.get(file_name) or image_map.get(normalized_name) or Path()
                if not image_path.is_file():
                    message = f"image not found for {key}: {meta['file_name']}"
                    if not args.allow_missing:
                        raise FileNotFoundError(message)
                    error_count += 1
                    handle.write(json.dumps({"expression_id": key, "status": "missing_image", "error": message}) + "\n")
                    continue
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                valid.append((row, meta, image_path, image))
            if not valid:
                continue
            texts = [str(row.get("expression_text") or meta["sentence"]) for row, meta, _, _ in valid]
            text_limit = _text_max_length(model)
            tokenizer_lengths: list[int | None] = [None] * len(texts)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                tokenized = tokenizer(
                    texts,
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                )
                tokenizer_lengths = [len(ids) for ids in tokenized["input_ids"]]
            process_kwargs = {
                "text": texts,
                "images": images,
                "padding": True,
                "return_tensors": "pt",
            }
            if text_limit is not None:
                # OWLv2's text tower has a 16-token context window.  Passing
                # the limit explicitly prevents Transformers from constructing
                # a 17-position embedding table for a long referring expression.
                process_kwargs.update(truncation=True, max_length=text_limit)
            inputs = processor(**process_kwargs)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
            sizes = torch.tensor([[image.height, image.width] for image in images], device=device)
            if hasattr(processor, "post_process_object_detection"):
                results = processor.post_process_object_detection(
                    outputs, threshold=args.threshold, target_sizes=sizes
                )
            else:
                # Recent Transformers releases expose the OWLv2 helper under
                # the grounded-detection name and require the text labels.
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=args.threshold,
                    target_sizes=sizes,
                    text_labels=[[text] for text in texts],
                )
            for valid_index, ((row, meta, image_path, image), result) in enumerate(zip(valid, results)):
                scores = result["scores"].detach().float().cpu().tolist()
                boxes = result["boxes"].detach().float().cpu().tolist()
                if scores:
                    best = max(range(len(scores)), key=lambda index: scores[index])
                    box = _normal_box(boxes[best], image.width, image.height)
                    confidence = float(scores[best])
                else:
                    box = [0.0, 0.0, 0.0, 0.0]
                    confidence = 0.0
                x1, y1, x2, y2 = box
                tokenizer_count = tokenizer_lengths[valid_index]
                record = {
                    "status": "ok",
                    "split": str(row["split"]),
                    "image_id": int(row["image_id"]),
                    "expression_id": str(row["expression_id"]),
                    "expression_text": str(row.get("expression_text") or meta["sentence"]),
                    "image_path": str(image_path),
                    "model": "google/owlv2-base-patch16-ensemble",
                    "model_path": str(args.model_path),
                    "candidate_box": box,
                    "candidate_confidence": confidence,
                    "candidate_count": int(len(scores)),
                    "candidate_box_area": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
                    "candidate_aspect_ratio": float((x2 - x1) / max(y2 - y1, 1e-8)),
                    "query_token_count": int(len(str(row.get("expression_text") or meta["sentence"]).split())),
                    "query_tokenizer_count": tokenizer_count,
                    "query_truncated": bool(
                        text_limit is not None
                        and tokenizer_count is not None
                        and tokenizer_count > text_limit
                    ),
                    "tokenizer_max_length": text_limit,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                processed += 1
            for image in images:
                image.close()
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "rows_requested": int(len(rows)),
        "rows_processed": processed,
        "errors": error_count,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "model": "google/owlv2-base-patch16-ensemble",
        "output_jsonl": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--sentence-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("batch-size must be positive; shard-index must be in [0, num-shards)")
    return args


def main() -> None:
    args = parse_args()
    summary = generate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
