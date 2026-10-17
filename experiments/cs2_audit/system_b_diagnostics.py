"""The diagnostic suite of Section 3.5 for System B (review item J1).

For each System B benchmark in the registry (three COCO benchmarks and
RefCLEF), from the assembled pairwise manifest and the protocol file:

* risk--coverage on the selection set (every distinct positive score);
* the decomposition of harm among switches at every declared threshold of
  the benchmark's label-free family, on the selection set;
* the exact probability of certification at the selected rule's observed
  certification size and harm rate;
* the selection trace (all declared rules on the selection set).

The same computations as for System A (``diagnose_failure.py``); only the
declared family differs, because System B's family is taken from its own
training switches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .diagnose_failure import _jsonable, certification_power, decomposition, max_certifiable_harms, risk_coverage
from .records import ROOT, load_registry

ARMS = {"RefCOCO+": "G1", "RefCOCOg": "G1", "RefCOCO": "G1", "RefCLEF": "G3"}


def slug(name: str) -> str:
    return name.replace("+", "plus")


def benchmark_diagnostics(entry: dict, family: list[float], out_dir: Path, name: str) -> dict:
    frame = pd.read_csv(ROOT / entry["primary"])
    curve = risk_coverage(frame)
    curve.to_csv(out_dir / f"risk_coverage_{slug(name)}.csv", index=False)
    summary = json.loads((ROOT / entry["audit_summary"]).read_text(encoding="utf-8"))
    cert = summary["certify"]
    m, k = int(cert["route_count"]), int(cert["harm_count"])
    rate = k / m if m else None
    always = frame[(frame["split"] == "select") & (frame["score"] > 0)]
    return {
        "family": family,
        "selected_threshold": float(entry["threshold"]),
        "decomposition_select": decomposition(frame, "select", family).to_dict("records"),
        "always_switch_select": {"routed": int(len(always))},
        "power": {
            "certify_routed": m,
            "certify_harm_rate": rate,
            "max_certifiable_harms": max_certifiable_harms(m),
            "power_at_observed_rate_and_size": certification_power(m, rate) if m else None,
        },
        "selection_trace": [
            {key: row[key] for key in ("policy", "route_count", "help_count", "harm_count", "conditional_harm", "net_accuracy_gain", "selected")}
            for row in summary["selection_trace"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/cs2_audit/configs/system_b_protocol_v1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/cs2_audit/results/system_b_20261016/diagnostics")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, entry in load_registry()["systems"]["System B"].items():
        family = [float(t) for t in protocol["threshold_families"][ARMS[name]][name]]
        result[name] = benchmark_diagnostics(entry, family, args.out_dir, name)
    out = args.out_dir / "system_b_diagnostics.json"
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
