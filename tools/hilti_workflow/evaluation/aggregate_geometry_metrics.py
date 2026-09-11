#!/usr/bin/env python3
"""Aggregate per-run geometry reports into the evaluation comparison schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results: dict = {}
    failures = []
    for path in sorted(args.metrics_root.glob(f"floor_*/*/run_*/reconstruction/metrics_{args.variant}.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("variant") != args.variant:
            failures.append(str(path))
            continue
        floor, date, run = Path(report["relative_path"]).parts
        results.setdefault(floor, {}).setdefault(date, {})[run] = report["metrics"]
    payload = {
        "metadata": {"method_name": args.method_name, "variant": args.variant,
                     "per_run_reports": sum(len(runs) for dates in results.values() for runs in dates.values()),
                     "invalid_reports": failures},
        "results": {args.method_name: results},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
