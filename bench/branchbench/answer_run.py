"""Answer-level runs, reports and cost estimates.

    python -m branchbench answer-run --split dev --runs 1 --label dev-smoke
    python -m branchbench answer-run --split test --runs 3 --label test-v1
    python -m branchbench answer-report results/test-v1 [--markdown docs/results_test.md]
    python -m branchbench estimate --split test --runs 3

Layout of a run directory (``results/`` is not committed):

    results/<label>/manifest.json         what ran: git commit, scenario hashes, model, prompts, Mem0 version and config
    results/<label>/run<k>/answers.jsonl  one row per (system, probe): context, answer, leaked, hit, correct, error
    results/<label>/run<k>/summary.json   metrics of that run
    results/<label>/summary.json          mean / min / max across runs
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from branchbench import answer_systems as AS
from branchbench.freeze import check as check_frozen, file_hashes, set_digest
from branchbench.schema import Scenario, load_dir
from branchbench.score import judge, normalize

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- running

def run_scenario(scenario: Scenario, views: List[str], llm: Callable, workdir: Path, run_tag: str,
                 mem0_factory: Optional[Callable] = None, prompts: Optional[List[str]] = None) -> List[dict]:
    """Feed the scenario in arrival order; answer each probe once its anchor has arrived,
    once per answer prompt (the memory is built once and shared by all prompts)."""

    prompts = list(prompts or AS.ANSWER_PROMPTS)
    rows: List[dict] = []

    def failed(probe, view, context, error):
        return [_row(scenario, probe, view, context, None, error, prompt) for prompt in prompts]

    probes = sorted(scenario.probes, key=lambda p: (scenario.by_id[p.anchor].t, p.id))
    errors: Dict[str, str] = {}
    try:
        systems = AS.build_systems(scenario, views, llm, workdir, run_tag, mem0_factory)
    except Exception as exc:  # noqa: BLE001 - recorded, never silently skipped
        return [row for p in probes for view in views for row in failed(p, view, None, f"setup: {type(exc).__name__}: {exc}")]
    events = list(scenario.events)
    cursor = 0
    try:
        for probe in probes:
            cutoff = scenario.by_id[probe.anchor].t
            while cursor < len(events) and events[cursor].t <= cutoff:
                for system in systems:
                    if any(v in errors for v in system.views):
                        continue
                    try:
                        system.ingest(events[cursor])
                    except Exception as exc:  # noqa: BLE001
                        for v in system.views:
                            errors[v] = f"ingest {events[cursor].id}: {type(exc).__name__}: {exc}"
                cursor += 1
            for system in systems:
                wanted = [v for v in system.views if v in views]
                if all(v in errors for v in wanted):
                    rows += [row for v in wanted for row in failed(probe, v, None, errors[v])]
                    continue
                try:
                    contexts = system.contexts(probe)
                except Exception as exc:  # noqa: BLE001
                    rows += [row for v in wanted for row in failed(probe, v, None, f"context: {type(exc).__name__}: {exc}")]
                    continue
                for view in wanted:
                    if view in errors:
                        rows += failed(probe, view, None, errors[view])
                        continue
                    for prompt in prompts:
                        try:
                            answer = llm(AS.answer_messages(contexts[view], probe.question, prompt), purpose="answer")
                            rows.append(_row(scenario, probe, view, contexts[view], answer, None, prompt))
                        except Exception as exc:  # noqa: BLE001
                            rows.append(_row(scenario, probe, view, contexts[view], None,
                                             f"answer: {type(exc).__name__}: {exc}", prompt))
    finally:
        for system in systems:
            system.close()
    for system in systems:
        if isinstance(system, AS.BranchmemSystem) and system.failures():
            for row in rows:
                if row["system"] == "branchmem":
                    row["extraction_failed_segments"] = system.failures()
    return rows


def _row(scenario: Scenario, probe, view: str, context: Optional[str], answer: Optional[str], error: Optional[str],
         prompt: str = "neutral") -> dict:
    leaked = hit = None
    if answer is not None:
        leaked, hit = judge(answer, probe)
    return {"scenario_id": scenario.id, "probe_id": probe.id, "kind": probe.kind, "branch_op": scenario.branch_op,
            "fact_type": scenario.fact_type, "system": view, "prompt": prompt, "question": probe.question, "context": context,
            "answer": answer, "leaked": leaked, "hit": hit,
            "correct": None if answer is None else bool(hit and not leaked), "error": error}


def run_once(scenarios: List[Scenario], views: List[str], llm: Callable, run_dir: Path, run_tag: str,
             workers: int, mem0_factory: Optional[Callable] = None) -> List[dict]:
    """Scenarios that finished without errors are saved under ``done/`` and reused when the
    run is resumed; any other scenario is rerun from an empty work directory (its Mem0 and
    note stores may hold a partial state)."""

    import shutil

    run_dir.mkdir(parents=True, exist_ok=True)
    work = run_dir / "work"
    done = run_dir / "done"
    done.mkdir(exist_ok=True)

    def task(scenario: Scenario) -> List[dict]:
        saved = done / f"{scenario.id}.jsonl"
        if saved.exists():
            return [json.loads(line) for line in saved.read_text(encoding="utf-8").splitlines() if line]
        shutil.rmtree(work / scenario.id, ignore_errors=True)
        rows = run_scenario(scenario, views, llm, work / scenario.id, run_tag, mem0_factory)
        if not any(row["error"] for row in rows):
            tmp = saved.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            tmp.replace(saved)
        return rows

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = [row for chunk in pool.map(task, scenarios) for row in chunk]
    with (run_dir / "answers.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(metrics(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- metrics

def _rate(values: List[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def _block(rows: List[dict]) -> dict:
    ok = [r for r in rows if r["answer"] is not None]
    leak = [r for r in ok if r["kind"] == "leak"]
    retain = [r for r in ok if r["kind"] == "retain"]
    return {"probes": len(rows), "errors": len(rows) - len(ok), "leak_probes": len(leak), "retain_probes": len(retain),
            "leak_rate": _rate([r["leaked"] for r in leak]),
            "correct_rate": _rate([r["correct"] for r in ok]),
            "retain_hit_rate": _rate([r["hit"] for r in retain]),
            "leak_correct_rate": _rate([r["correct"] for r in leak]),
            # the two kinds of leak: the sibling fact is mentioned next to the right answer,
            # or it replaces the right answer
            "leak_mention_rate": _rate([bool(r["leaked"] and r["hit"]) for r in leak]),
            "leak_wrong_rate": _rate([bool(r["leaked"] and not r["hit"]) for r in leak])}


def result_key(row: dict) -> str:
    """``system`` for the neutral answer prompt, ``system@prompt`` for the others."""
    prompt = row.get("prompt") or "neutral"
    return row["system"] if prompt == "neutral" else f"{row['system']}@{prompt}"


def _key_order(key: str):
    system, _, prompt = key.partition("@")
    prompts = list(AS.ANSWER_PROMPTS)
    return (prompts.index(prompt or "neutral") if (prompt or "neutral") in prompts else 99,
            AS.VIEWS.index(system) if system in AS.VIEWS else 99)


def metrics(rows: List[dict]) -> dict:
    out: Dict[str, Any] = {}
    for system in sorted({result_key(r) for r in rows}, key=_key_order):
        mine = [r for r in rows if result_key(r) == system]
        ops = sorted({r["branch_op"] for r in mine})
        out[system] = {"overall": _block(mine), "by_branch_op": {op: _block([r for r in mine if r["branch_op"] == op]) for op in ops}}
    return out


def across_runs(per_run: List[dict]) -> dict:
    """mean / min / max of each metric over runs (``None`` values are left out)."""

    def combine(blocks: List[dict]) -> dict:
        out = {}
        for key in blocks[0]:
            values = [b[key] for b in blocks if b.get(key) is not None]
            if key in ("probes", "leak_probes", "retain_probes", "errors"):
                out[key] = values
            else:
                out[key] = ({"mean": round(sum(values) / len(values), 4), "min": min(values), "max": max(values), "n": len(values)}
                            if values else None)
        return out

    systems = per_run[0].keys()
    result = {}
    for system in systems:
        result[system] = {"overall": combine([r[system]["overall"] for r in per_run]),
                          "by_branch_op": {op: combine([r[system]["by_branch_op"][op] for r in per_run])
                                           for op in per_run[0][system]["by_branch_op"]}}
    return result


# ---------------------------------------------------------------- manifest

def _git() -> dict:
    def sh(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"commit": sh("rev-parse", "HEAD"), "dirty": bool(sh("status", "--porcelain"))}


def _mem0_info(views: List[str], workdir_example: Path) -> Optional[dict]:
    if not {"mem0", "mem0_raw_filtered"} & set(views):
        return None
    info: Dict[str, Any] = {"integration": "mem0.Memory.from_config; add([message], user_id) once per message in arrival "
                                           "order (tool messages relabelled as assistant with prefix "
                                           f"{AS.MEM0_TOOL_PREFIX!r}); search(question, top_k={AS.MEM0_SEARCH_LIMIT}, filters={{user_id}}), "
                                           "default threshold; infer=True (mem0ai 2.x additive pipeline: one LLM call per add, "
                                           "context = existing memories + the session's last 10 messages in arrival order)",
                            "search_limit": AS.MEM0_SEARCH_LIMIT}
    try:
        import importlib.metadata as md

        info["version"] = md.version("mem0ai")
    except Exception as exc:  # noqa: BLE001
        info["version_error"] = f"{type(exc).__name__}: {exc}"
    import importlib.util

    # optional extras change Mem0's behaviour (spaCy lemmatisation, fastembed BM25 hybrid search)
    info["optional_extras_installed"] = {name: importlib.util.find_spec(name) is not None for name in ("spacy", "fastembed")}
    try:
        info["config"] = AS.redact(AS.mem0_config(workdir_example))
    except KeyError as exc:
        info["config_error"] = f"missing environment variable {exc}"
    return info


def manifest(*, label: str, split: str, views: List[str], runs: int, llm: Any, scenario_dir: Path,
             scenarios: List[Scenario], out_dir: Path) -> dict:
    hashes = file_hashes(scenario_dir)
    return {
        "label": label, "split": split, "runs": runs, "systems": views,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git(), "python": sys.version.split()[0], "platform": platform.platform(),
        "scenarios": {"set_sha256": set_digest(hashes), "frozen_check": check_frozen(scenario_dir, "test") or "test split unchanged since freeze",
                      "ids": [s.id for s in scenarios], "probes": sum(len(s.probes) for s in scenarios),
                      "files": {k: v for k, v in hashes.items() if k.startswith(split + "/") or split == "all"}},
        "answer_llm": llm.describe(),
        "prompts": AS.prompt_fingerprint(),
        "answer_prompts": list(AS.ANSWER_PROMPTS),
        "params": {"summary_every": AS.SUMMARY_EVERY, "summary_recent": AS.SUMMARY_RECENT,
                   "notes_search_limit": AS.NOTES_SEARCH_LIMIT, "mem0_search_limit": AS.MEM0_SEARCH_LIMIT,
                   "branchmem_segment_max_turns": AS.BranchmemSystem.SEGMENT_MAX_TURNS},
        "mem0": _mem0_info(views, out_dir / "run1" / "work" / "<scenario>" / "mem0"),
    }


def answer_run(*, scenario_dir: Path, split: str, views: List[str], runs: int, out_root: Path, label: str,
               workers: int, llm_factory: Callable, allow_unfrozen: bool = False, resume: bool = False) -> Path:
    scenarios = load_dir(scenario_dir, split)
    problems = check_frozen(scenario_dir, "test")
    if split == "test" and problems and not allow_unfrozen:
        raise SystemExit("test split is not frozen or has changed:\n" + "\n".join(problems))
    out_dir = out_root / label
    llm = llm_factory()
    info = manifest(label=label, split=split, views=views, runs=runs, llm=llm, scenario_dir=scenario_dir,
                    scenarios=scenarios, out_dir=out_dir)
    if out_dir.exists():
        if not resume:
            raise SystemExit(f"{out_dir} already exists; pick another --label or pass --resume")
        old = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        same = ("split", "runs", "systems")
        mismatched = [k for k in same if old.get(k) != info.get(k)]
        if old["scenarios"]["set_sha256"] != info["scenarios"]["set_sha256"]:
            mismatched.append("scenario set")
        if old["answer_llm"] != info["answer_llm"] or old["prompts"] != info["prompts"]:
            mismatched.append("model or prompts")
        if mismatched:
            raise SystemExit(f"cannot resume {out_dir}: {', '.join(mismatched)} differ from the original run")
        sessions = old.get("sessions") or [{k: old.get(k) for k in ("started_at", "finished_at", "git", "answer_llm_usage", "aborted")}]
        info = old
        info["sessions"] = sessions
        info["started_at_resume"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        info.pop("aborted", None)
    else:
        out_dir.mkdir(parents=True)
        info["sessions"] = []
    (out_dir / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    per_run = []
    try:
        for k in range(1, runs + 1):
            rows = run_once(scenarios, views, llm, out_dir / f"run{k}", f"{label}-r{k}", workers)
            per_run.append(metrics(rows))
            print(f"run {k}/{runs} done: {sum(1 for r in rows if r['error'])} errors", file=sys.stderr)
    except Exception:
        info["aborted"] = traceback.format_exc()
        raise
    finally:
        info["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        info["answer_llm_usage"] = llm.usage.to_dict()  # this session only; earlier ones are in "sessions"
        info["sessions"] = info.get("sessions", []) + [{"started_at": info.get("started_at_resume", info["started_at"]),
                                                       "finished_at": info["finished_at"], "git": _git(),
                                                       "answer_llm_usage": info["answer_llm_usage"],
                                                       "aborted": info.get("aborted")}]
        (out_dir / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    if per_run:
        (out_dir / "summary.json").write_text(json.dumps(across_runs(per_run), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


# ---------------------------------------------------------------- report

LABELS = {"full_history": "全量历史", "rolling_summary": "整段摘要式记忆", "mem0": "Mem0 开源版",
          "mem0_raw_filtered": "Mem0 + 原文按分支过滤", "branchmem": "按来源判断可见性（branchmem）",
          "path_only": "只给当前分支原文（上界）"}
OP_ORDER = ("edit_user", "regenerate", "fork_checkpoint", "update_in_branch", "switch_back")


def _fmt(cell: Optional[dict]) -> str:
    if not cell:
        return "–"
    mean, lo, hi = (round(100 * cell[k], 1) for k in ("mean", "min", "max"))
    return f"{mean:g}%" if lo == hi else f"{mean:g}% ({lo:g}–{hi:g})"


def rescore(out_dir: Path, scenario_dir: Path = REPO / "bench" / "scenarios") -> dict:
    """Re-judge every saved answer with the current scorer and rewrite the summaries.

    The report is always built from ``answers.jsonl`` this way, so every number traces back
    to raw answers, and a scorer fix never needs new model calls.
    """

    info = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    probes = {(s.id, p.id): p for s in load_dir(scenario_dir, info["split"]) for p in s.probes}
    per_run = []
    for run_dir in sorted(out_dir.glob("run*"), key=lambda d: int(d.name[3:])):
        path = run_dir / "answers.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            if row["answer"] is not None:
                row["leaked"], row["hit"] = judge(row["answer"], probes[(row["scenario_id"], row["probe_id"])])
                row["correct"] = bool(row["hit"] and not row["leaked"])
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        per_run.append(metrics(rows))
        (run_dir / "summary.json").write_text(json.dumps(per_run[-1], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = across_runs(per_run)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def question_echo_probes(scenarios) -> List[tuple]:
    """Leak probes whose forbidden word already appears in the question ("我需要上夜班吗？").

    A correct denial ("不需要上夜班") repeats the word and is scored as a leak, so these probes
    flag almost every answer. Found after the test run; the frozen set is left unchanged and the
    report shows a sensitivity analysis without them next to the main result.
    """

    return [(s.id, p.id) for s in scenarios for p in s.probes
            if p.kind == "leak" and any(normalize(w) in normalize(p.question) for w in p.must_not_include)]


def _summary_without(out_dir: Path, exclude: set) -> dict:
    per_run = []
    for run_dir in sorted(out_dir.glob("run*"), key=lambda d: int(d.name[3:])):
        path = run_dir / "answers.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            per_run.append(metrics([r for r in rows if (r["scenario_id"], r["probe_id"]) not in exclude]))
    return across_runs(per_run)


def _overall_table(groups: Dict[str, dict]) -> List[str]:
    prompts = list(groups)
    lines = ["| 系统 | " + " | ".join(f"泄漏率 ({p})" for p in prompts) + " | " + " | ".join(f"正确率 ({p})" for p in prompts) + " |",
             "|---|" + "---|" * (2 * len(prompts))]
    for system in groups[prompts[0]]:
        cells = [_fmt(groups[p].get(system, {}).get("overall", {}).get(k)) for k in ("leak_rate", "correct_rate") for p in prompts]
        lines.append(f"| {LABELS.get(system, system)} | " + " | ".join(cells) + " |")
    return lines


PROMPT_TITLES = {"neutral": "主结果：中性回答提示词", "conflict": "稳健性检查：冲突处理回答提示词（事先写定，未调参）"}


def _split_by_prompt(summary: dict) -> Dict[str, dict]:
    groups: Dict[str, dict] = {}
    for key, data in summary.items():
        system, _, prompt = key.partition("@")
        groups.setdefault(prompt or "neutral", {})[system] = data
    return groups


def _tables(summary: dict) -> List[str]:
    lines = ["| 系统 | 泄漏率 ↓ (leak 题) | 其中答成兄弟分支 | 其中答对但提及 | 正确率 ↑ (全部题) | 保留题命中率 ↑ | 错误数 |",
             "|---|---|---|---|---|---|---|"]
    for system, data in summary.items():
        o = data["overall"]
        lines.append(f"| {LABELS.get(system, system)} | {_fmt(o['leak_rate'])} | {_fmt(o.get('leak_wrong_rate'))} | "
                     f"{_fmt(o.get('leak_mention_rate'))} | {_fmt(o['correct_rate'])} | "
                     f"{_fmt(o['retain_hit_rate'])} | {sum(o['errors'])} |")
    lines.append("")
    for base, filtered, what in (("mem0", "mem0_raw_filtered", "Mem0 只过滤原文"),
                                 ("full_history", "path_only", "原文全量 → 只给当前分支")):
        if base in summary and filtered in summary:
            a = (summary[base]["overall"]["leak_rate"] or {}).get("mean")
            b = (summary[filtered]["overall"]["leak_rate"] or {}).get("mean")
            if a:
                lines.append(f"- {what}：泄漏率 {100 * a:.1f}% → {100 * b:.1f}%，挡住 {100 * (a - b) / a:.0f}% 的泄漏。")
    ops = [op for op in OP_ORDER if any(op in d["by_branch_op"] for d in summary.values())]
    lines += ["", "按分叉方式的泄漏率（`switch_back` 单独放在最后一列）：", "",
              "| 系统 | " + " | ".join(f"{op} (n={_n(summary, op)})" for op in ops) + " |",
              "|---|" + "---|" * len(ops)]
    for system, data in summary.items():
        lines.append(f"| {LABELS.get(system, system)} | " + " | ".join(_fmt(data["by_branch_op"].get(op, {}).get("leak_rate")) for op in ops) + " |")
    lines += ["", "按分叉方式的正确率：", "",
              "| 系统 | " + " | ".join(ops) + " |", "|---|" + "---|" * len(ops)]
    for system, data in summary.items():
        lines.append(f"| {LABELS.get(system, system)} | " + " | ".join(_fmt(data["by_branch_op"].get(op, {}).get("correct_rate")) for op in ops) + " |")
    return lines


def report_markdown(out_dir: Path, scenario_dir: Path = REPO / "bench" / "scenarios") -> str:
    info = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    groups = _split_by_prompt(rescore(out_dir, scenario_dir))
    lines = [f"<!-- generated by `python -m branchbench answer-report {out_dir.as_posix()}`; do not edit by hand -->", "",
             f"Run `{info['label']}`: split `{info['split']}`, {len(info['scenarios']['ids'])} scenarios, "
             f"{info['scenarios']['probes']} probes, {info['runs']} runs; answer model `{info['answer_llm'].get('model')}` "
             f"(temperature 0); scenario set `{info['scenarios']['set_sha256'][:12]}`; commit `{info['git']['commit'][:10]}`"
             + (" (dirty)" if info["git"]["dirty"] else "") + ".", "",
             "单元格：多次运行的均值，各次不同时括号内为（最小–最大）。泄漏率 = 回答含任一禁止词的 leak 题比例，"
             "拆成两类：没有答案词（答成兄弟分支），以及有答案词但同时提到兄弟分支的事实。", ""]
    for prompt in [p for p in AS.ANSWER_PROMPTS if p in groups] + [p for p in groups if p not in AS.ANSWER_PROMPTS]:
        lines += [f"## {PROMPT_TITLES.get(prompt, prompt)}", ""] + _tables(groups[prompt]) + [""]
    if len(groups) > 1 and "neutral" in groups:
        others = [p for p in groups if p != "neutral"]
        lines += ["## 两种回答提示词的泄漏率对比", "",
                  "| 系统 | neutral | " + " | ".join(others) + " |", "|---|---|" + "---|" * len(others)]
        for system, data in groups["neutral"].items():
            cells = [_fmt(data["overall"]["leak_rate"])] + [
                _fmt(groups[p].get(system, {}).get("overall", {}).get("leak_rate")) for p in others]
            lines.append(f"| {LABELS.get(system, system)} | " + " | ".join(cells) + " |")
        lines.append("")
    echo = question_echo_probes(load_dir(scenario_dir, info["split"]))
    if echo:
        reduced = _summary_without(out_dir, set(echo))
        (out_dir / "summary_sensitivity.json").write_text(
            json.dumps({"excluded": echo, "summary": reduced}, ensure_ascii=False, indent=2), encoding="utf-8")
        sub = _split_by_prompt(reduced)
        n_leak = next(iter(reduced.values()))["overall"]["leak_probes"][0]
        lines += ["## 敏感性分析：去掉禁止词出现在问题里的 leak 题（事后发现）", "",
                  f"去掉 {len(echo)} 道题：{', '.join(f'{a} {b}' for a, b in echo)}。问题本身含禁止词"
                  "（如「我需要上夜班吗？」禁止词「夜班」），正确的否定回答也会重复这个词而被判为泄漏，"
                  f"几乎所有系统在这几道题上都被判泄漏。测试集已冻结，未改动；主结果保留这些题，下表只是对照。"
                  f"剩 {n_leak} 道 leak 题。数据：`summary_sensitivity.json`。", ""] + _overall_table(sub) + [""]
    if info.get("mem0"):
        m = info["mem0"]
        lines += [f"Mem0: `mem0ai` {m.get('version', m.get('version_error'))}; {m['integration']}."]
    lines += ["", "Answer prompts:", ""] + [f"- `{name}`: {text}" for name, text in AS.ANSWER_PROMPTS.items()]
    return "\n".join(lines) + "\n"


def _n(summary: dict, op: str) -> int:
    first = next(iter(summary.values()))
    counts = first["by_branch_op"].get(op, {}).get("leak_probes") or [0]
    return counts[0]


# ---------------------------------------------------------------- estimate

#: Mem0 cost model. Mem0's own prompts are not counted by our client, so they are modelled
#: on mem0ai 2.2.1: one extraction call per add with a ~33k-character system prompt.
MEM0_ASSUMPTIONS = {"llm_calls_per_add": 1, "prompt_tokens_per_call": 11000, "completion_tokens_per_call": 200,
                    "embedding_calls_per_add": 2, "embedding_calls_per_search": 1}
TOKENS_PER_CHAR = 0.7  # mixed Chinese/English prompts; conservative


class _CountingMem0(AS.System):
    views = ("mem0", "mem0_raw_filtered")
    adds = 0
    searches = 0

    def ingest(self, event) -> None:
        super().ingest(event)
        type(self).adds += 1

    def contexts(self, probe) -> Dict[str, str]:
        type(self).searches += 1
        return {"mem0": "[记忆]\n" + "- 记忆条目\n" * 5,
                "mem0_raw_filtered": AS.render(self.scenario.path(probe.anchor)) + "\n\n[记忆]\n" + "- 记忆条目\n" * 5}


def estimate(scenario_dir: Path, split: str, runs: int, views: List[str]) -> dict:
    from branchbench.llm import CountingFake

    scenarios = load_dir(scenario_dir, split)
    fake = CountingFake()
    _CountingMem0.adds = _CountingMem0.searches = 0
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rows = run_once(scenarios, views, fake, Path(tmp) / "run", "estimate", 1,
                        mem0_factory=lambda s, w, t: _CountingMem0(s))
    errors = [r for r in rows if r["error"]]
    u = fake.usage
    ours_prompt = round(u.prompt_chars * TOKENS_PER_CHAR)
    ours_completion_real = round(u.by_purpose.get("answer", 0) * 60 + u.by_purpose.get("summary", 0) * 300
                                 + u.by_purpose.get("extract", 0) * 250)
    m = MEM0_ASSUMPTIONS
    mem0_llm = _CountingMem0.adds * m["llm_calls_per_add"]
    per_run = {
        "our_client_calls": dict(u.by_purpose), "our_client_prompt_tokens": ours_prompt,
        "our_client_completion_tokens": ours_completion_real,
        "mem0_adds": _CountingMem0.adds, "mem0_searches": _CountingMem0.searches,
        "mem0_llm_calls": mem0_llm, "mem0_prompt_tokens": mem0_llm * m["prompt_tokens_per_call"],
        "mem0_completion_tokens": mem0_llm * m["completion_tokens_per_call"],
        "embedding_calls": _CountingMem0.adds * m["embedding_calls_per_add"] + _CountingMem0.searches * m["embedding_calls_per_search"],
    }
    total_llm_calls = u.calls + mem0_llm
    return {"split": split, "scenarios": len(scenarios), "probes": sum(len(s.probes) for s in scenarios), "systems": views,
            "per_run": per_run, "runs": runs,
            "total": {"llm_calls": total_llm_calls * runs,
                      "prompt_tokens": (ours_prompt + per_run["mem0_prompt_tokens"]) * runs,
                      "completion_tokens": (ours_completion_real + per_run["mem0_completion_tokens"]) * runs,
                      "embedding_calls": per_run["embedding_calls"] * runs},
            "assumptions": {"tokens_per_char": TOKENS_PER_CHAR, "mem0": m,
                            "completion_tokens": "answer 60, summary 300, extraction 250 per call"},
            "dry_run_errors": [r["error"] for r in errors][:5]}
