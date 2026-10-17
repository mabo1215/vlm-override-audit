"""Registry of audited record files (``configs/audit_records.json``).

Every analysis reads its input records through this module, so a corrected
record set or a new system is added in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().parent / "configs" / "audit_records.json"


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_manifests(system: str = "System A", path: Path = REGISTRY) -> dict[str, tuple[Path, float]]:
    """Benchmark -> (primary record file, frozen threshold) for one system."""

    entries = load_registry(path)["systems"][system]
    return {name: (ROOT / entry["primary"], float(entry["threshold"])) for name, entry in entries.items()}


def check_thresholds(system: str = "System A", path: Path = REGISTRY) -> None:
    """Fail if a registered threshold differs from the rule the audit froze."""

    for name, entry in load_registry(path)["systems"][system].items():
        summary = ROOT / entry.get("audit_summary", "")
        if entry.get("audit_summary") and summary.exists():
            policy = json.loads(summary.read_text(encoding="utf-8"))["certify"]["policy"]
            frozen = float(policy.rsplit("_", 1)[-1])
            # policy names carry the threshold to six significant digits
            if abs(frozen - float(f"{float(entry['threshold']):.6g}")) > 1e-12:
                raise ValueError(f"{system} {name}: registry threshold {entry['threshold']} != frozen {frozen}")
