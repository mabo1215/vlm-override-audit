"""Tables 3 and 4 of the paper: images per set, and images shared between benchmarks.

Both are read from the registered audit records (one expression per image);
nothing here uses scores or labels.

Example:
    python -m experiments.cs2_audit.data_tables
"""

from __future__ import annotations

import json
from itertools import combinations

import pandas as pd

from .records import load_registry, ROOT

SETS = ("train", "select", "certify", "confirm")


def set_sizes() -> dict[str, dict[str, int]]:
    sizes = {}
    for system in ("System A", "System B"):
        for name, entry in load_registry()["systems"][system].items():
            if name in sizes:
                continue
            frame = pd.read_csv(ROOT / entry["primary"])
            counts = frame["split"].value_counts()
            sizes[name] = {s: int(counts.get(s, 0)) for s in SETS} | {"total": int(len(frame))}
    return sizes


def overlaps() -> dict[str, dict[str, int]]:
    roles = {
        name: pd.read_csv(ROOT / entry["primary"]).set_index("image_id")["split"]
        for name, entry in load_registry()["systems"]["System A"].items()
    }
    out = {}
    for left, right in combinations(roles, 2):
        shared = roles[left].index.intersection(roles[right].index)
        same = int((roles[left].loc[shared] == roles[right].loc[shared]).sum())
        out[f"{left} / {right}"] = {"shared": int(len(shared)), "same_role": same, "different_role": int(len(shared) - same)}
    return out


def main() -> None:
    print(json.dumps({"images_per_set": set_sizes(), "shared_images": overlaps()}, indent=2))


if __name__ == "__main__":
    main()
