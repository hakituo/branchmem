"""Command line.

    python -m branchbench validate [scenario_dir]
    python -m branchbench context-report [scenario_dir]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from branchbench.context_baselines import BASELINES, contexts
from branchbench.schema import load_dir
from branchbench.score import aggregate, score_texts
from branchbench.validate import validate_all

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "scenarios"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="branchbench")
    parser.add_argument("command", choices=("validate", "context-report"))
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args(argv)
    scenarios = load_dir(args.directory)
    errors = validate_all(scenarios)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    probes = sum(len(s.probes) for s in scenarios)
    if args.command == "validate":
        print(f"OK: {len(scenarios)} scenarios, {probes} probes, all valid")
        return 0
    report = {
        name: aggregate(score_texts(scenarios, contexts(scenarios, fn)))
        for name, fn in BASELINES.items()
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
