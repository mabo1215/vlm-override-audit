"""Summarize reproducibility and inference-cost JSONL events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(event: Mapping[str, object], key: str) -> float:
    value = event.get(key, 0)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"event field {key!r} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"event field {key!r} must be finite and non-negative")
    return result


def _duration_ms(event: Mapping[str, object]) -> float:
    if "duration_ms" in event:
        return _number(event, "duration_ms")
    if "started_at" not in event or "ended_at" not in event:
        return 0.0
    try:
        start = datetime.fromisoformat(str(event["started_at"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(event["ended_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("started_at/ended_at must be ISO-8601 timestamps") from exc
    duration = (end - start).total_seconds() * 1000.0
    if duration < 0:
        raise ValueError("ended_at must not precede started_at")
    return duration


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_events(
    events: Iterable[Mapping[str, object]],
    *,
    input_sha256: str | None = None,
    output_sha256: str | None = None,
) -> dict[str, object]:
    records = list(events)
    durations = [_duration_ms(event) for event in records]
    successful = sum(str(event.get("status", "ok")).lower() in {"ok", "success", "complete"} for event in records)
    wall_time_s = sum(durations) / 1000.0
    total_candidates = sum(_number(event, "candidates") for event in records)
    summary: dict[str, object] = {
        "event_count": len(records),
        "successful_events": int(successful),
        "failed_events": len(records) - int(successful),
        "wall_time_s": wall_time_s,
        "latency_ms": {
            "mean": mean(durations) if durations else None,
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": max(durations) if durations else None,
        },
        "vlm_calls": int(sum(_number(event, "vlm_calls") for event in records)),
        "candidates": int(total_candidates),
        "input_tokens": int(sum(_number(event, "input_tokens") for event in records)),
        "output_tokens": int(sum(_number(event, "output_tokens") for event in records)),
        "retries": int(sum(_number(event, "retries") for event in records)),
        "parse_failures": int(sum(_number(event, "parse_failures") for event in records)),
        "batch_sizes": sorted({int(_number(event, "batch_size")) for event in records if "batch_size" in event}),
        "peak_memory_mb": max((_number(event, "peak_memory_mb") for event in records), default=None),
        "gpu_names": sorted({str(event["gpu_name"]) for event in records if event.get("gpu_name")}),
        "throughput_candidates_per_s": total_candidates / wall_time_s if wall_time_s else None,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }
    return summary


def read_jsonl(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        events.append(event)
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--output-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_events(
        read_jsonl(args.input_jsonl),
        input_sha256=args.input_sha256,
        output_sha256=args.output_sha256,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
