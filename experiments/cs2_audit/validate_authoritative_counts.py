"""Validate that audit artifacts and manuscript counts share one ledger."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = Path(__file__).with_name("configs") / "authoritative_counts.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _relative_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_summary(path: Path, expected: dict[str, int]) -> dict[str, Any]:
    summary = _read_json(path)
    observed = summary.get("rows")
    if not isinstance(observed, dict):
        raise ValueError(f"{path}: missing object field 'rows'")
    normalized = {str(key): int(value) for key, value in observed.items()}
    expected_normalized = {str(key): int(value) for key, value in expected.items()}
    if normalized != expected_normalized:
        raise ValueError(
            f"{path}: rows {normalized} do not match authoritative "
            f"counts {expected_normalized}"
        )
    return {"path": str(path), "rows": normalized}


def validate_paper_text(text: str, benchmarks: dict[str, Any], forbidden: list[str]) -> dict[str, Any]:
    missing_sentences: list[str] = []
    for spec in benchmarks.values():
        legacy_sentence = str(spec["paper_sentence"])
        if legacy_sentence in text:
            continue
        counts = spec.get("counts")
        if isinstance(counts, dict):
            reader_sentence = (
                f"{int(counts['train']):,} training, "
                f"{int(counts['select']):,} selection, "
                f"{int(counts['certify']):,} certification, and "
                f"{int(counts['confirm']):,} confirmation images"
            )
            normalized_text = re.sub(r"\s+", " ", text)
            if reader_sentence in normalized_text:
                continue
        missing_sentences.append(legacy_sentence)
    stale = [value for value in forbidden if value in text]
    if missing_sentences or stale:
        details: list[str] = []
        if missing_sentences:
            details.append(f"missing benchmark sentences: {missing_sentences}")
        if stale:
            details.append(f"forbidden stale count strings: {stale}")
        raise ValueError("paper count validation failed: " + "; ".join(details))
    return {"checked_benchmarks": sorted(benchmarks), "stale_strings_found": []}


def validate_ledger(
    ledger_path: Path,
    *,
    repo_root: Path,
    paper_path: Path | None = None,
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    benchmarks = ledger.get("benchmarks")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError(f"{ledger_path}: 'benchmarks' must be a non-empty object")
    artifact_reports = []
    for name, spec in benchmarks.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{ledger_path}: benchmark {name!r} is not an object")
        artifact_value = spec.get("audit_summary")
        counts = spec.get("counts")
        if not isinstance(artifact_value, str) or not isinstance(counts, dict):
            raise ValueError(f"{ledger_path}: benchmark {name!r} lacks path/counts")
        artifact_reports.append(
            {"benchmark": name, **validate_summary(_relative_path(repo_root, artifact_value), counts)}
        )

    paper_report = None
    if paper_path is not None:
        paper_report = validate_paper_text(
            paper_path.read_text(encoding="utf-8"),
            benchmarks,
            [str(value) for value in ledger.get("forbidden_stale_count_strings", [])],
        )
    return {
        "ledger": str(ledger_path),
        "sampling_unit": ledger.get("sampling_unit"),
        "artifacts": artifact_reports,
        "paper": paper_report,
        "status": "pass",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--paper-tex", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_ledger(
        args.ledger,
        repo_root=args.repo_root,
        paper_path=args.paper_tex,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
