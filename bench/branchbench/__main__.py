"""Command line.

    python -m branchbench validate [scenario_dir] [--split all|dev|test]
    python -m branchbench context-report [scenario_dir] [--split all|dev|test]
    python -m branchbench freeze | check-frozen [scenario_dir]
    python -m branchbench estimate [--split test] [--runs 3] [--systems ...]
    python -m branchbench answer-run --split dev|test --label NAME [--runs 3] [--systems ...] [--workers 4]
    python -m branchbench answer-report results/NAME [--markdown FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from branchbench.schema import load_dir
from branchbench.validate import validate_all

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "scenarios"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """``KEY=value`` lines from the repo's .env; variables already set are left alone."""

    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="branchbench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "context-report", "freeze", "check-frozen"):
        p = sub.add_parser(name)
        p.add_argument("directory", nargs="?", type=Path, default=DEFAULT_DIR)
        p.add_argument("--split", choices=("all", "dev", "test"), default="all")
    from branchbench.answer_systems import VIEWS

    for name in ("estimate", "answer-run"):
        p = sub.add_parser(name)
        p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
        p.add_argument("--split", choices=("dev", "test"), default="test" if name == "estimate" else None,
                       required=name == "answer-run")
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--systems", nargs="+", choices=VIEWS, default=list(VIEWS))
        if name == "answer-run":
            p.add_argument("--label", required=True)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--limit", type=int, default=0, help="only the first N scenarios (smoke test)")
            p.add_argument("--out", type=Path, default=RESULTS_DIR)
            p.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --label")
    p = sub.add_parser("answer-report")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    if args.command in ("validate", "context-report", "freeze", "check-frozen"):
        scenarios = load_dir(args.directory, args.split)
        errors = validate_all(scenarios)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        probes = sum(len(s.probes) for s in scenarios)
        if args.command == "validate":
            print(f"OK: {len(scenarios)} scenarios, {probes} probes, all valid")
        elif args.command == "context-report":
            from branchbench.context_baselines import BASELINES, contexts
            from branchbench.score import aggregate, score_texts

            report = {name: aggregate(score_texts(scenarios, contexts(scenarios, fn))) for name, fn in BASELINES.items()}
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "freeze":
            from branchbench.freeze import freeze

            record = freeze(args.directory)
            print(f"frozen {len(record['files'])} files, set sha256 {record['set_sha256']}")
        else:
            from branchbench.freeze import check

            problems = check(args.directory, args.split)
            print("\n".join(problems) or "unchanged since freeze")
            return 1 if problems else 0
        return 0

    from branchbench import answer_run

    if args.command == "estimate":
        print(json.dumps(answer_run.estimate(args.dir, args.split, args.runs, args.systems), ensure_ascii=False, indent=2))
        return 0
    if args.command == "answer-run":
        from branchbench.llm import ChatClient

        if args.limit:
            original = answer_run.load_dir
            answer_run.load_dir = lambda d, s: original(d, s)[: args.limit]
        out = answer_run.answer_run(scenario_dir=args.dir, split=args.split, views=args.systems, runs=args.runs,
                                    out_root=args.out, label=args.label, workers=args.workers,
                                    llm_factory=ChatClient.from_env, resume=args.resume)
        print(answer_run.report_markdown(out))
        return 0
    text = answer_run.report_markdown(args.run_dir)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
