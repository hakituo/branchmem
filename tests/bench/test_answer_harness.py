"""Answer-level harness without a model: freezing, the dry run, the branchmem adapter, metrics."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from branchbench import answer_run
from branchbench import answer_systems as AS
from branchbench.freeze import check, freeze
from branchbench.schema import load_dir
from branchbench.score import judge

DATA = Path(__file__).resolve().parents[2] / "bench" / "scenarios"


def _echo_llm(messages, purpose="other"):
    """Extraction copies every message of an exchange into a note citing that exchange."""
    if purpose != "extract":
        return "好的"
    notes, current = [], None
    for line in messages[-1]["content"].splitlines():
        if line.startswith("["):
            current = int(line.strip("[]"))
        elif current is not None and line.strip():
            notes.append({"kind": "fact", "text": line.strip(), "exchanges": [current]})
    return json.dumps({"notes": notes})


def test_split_sizes():
    assert len(load_dir(DATA, "dev")) == 10
    assert len(load_dir(DATA, "test")) == 20
    assert len(load_dir(DATA)) == 30


def test_freeze_detects_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "s"
        shutil.copytree(DATA, root, ignore=shutil.ignore_patterns("FROZEN.json"))
        assert check(root)  # not frozen yet
        freeze(root)
        assert check(root) == []
        target = next((root / "test").glob("*.json"))
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
        assert check(root) == [f"changed: test/{target.name}"]


def test_branchmem_adapter_stores_sibling_notes_but_never_shows_them():
    for scenario in load_dir(DATA):
        with tempfile.TemporaryDirectory() as tmp:
            system = AS.BranchmemSystem(scenario, _echo_llm, Path(tmp))
            for event in scenario.events:
                system.ingest(event)
            assert system.failures() == 0
            for probe in scenario.probes:
                leaked, hit = judge(system.contexts(probe)["branchmem"], probe)
                assert not leaked and hit, (scenario.id, probe.id)


def test_dry_run_covers_every_system_without_errors():
    report = answer_run.estimate(DATA, "dev", 1, list(AS.VIEWS))
    assert report["dry_run_errors"] == []
    assert report["per_run"]["our_client_calls"]["answer"] == report["probes"] * 6


def test_metrics_and_across_runs():
    rows = [
        {"system": "path_only", "branch_op": "edit_user", "kind": "leak", "answer": "a", "leaked": False, "hit": True, "correct": True},
        {"system": "path_only", "branch_op": "edit_user", "kind": "retain", "answer": "b", "leaked": False, "hit": False, "correct": False},
        {"system": "path_only", "branch_op": "switch_back", "kind": "leak", "answer": None, "leaked": None, "hit": None, "correct": None},
    ]
    m = answer_run.metrics(rows)["path_only"]
    assert m["overall"]["errors"] == 1 and m["overall"]["leak_rate"] == 0.0
    assert m["overall"]["retain_hit_rate"] == 0.0 and m["overall"]["correct_rate"] == 0.5
    combined = answer_run.across_runs([answer_run.metrics(rows)] * 3)
    assert combined["path_only"]["overall"]["correct_rate"] == {"mean": 0.5, "min": 0.5, "max": 0.5, "n": 3}


def test_resume_reuses_finished_scenarios_and_reruns_the_rest():
    from branchbench.llm import CountingFake

    views = ["full_history", "rolling_summary", "branchmem", "path_only"]
    fakes = []

    def factory():
        fakes.append(CountingFake())
        return fakes[-1]

    with tempfile.TemporaryDirectory() as tmp:
        kwargs = dict(scenario_dir=DATA, split="dev", views=views, runs=2, out_root=Path(tmp), label="r",
                      workers=2, llm_factory=factory)
        out = answer_run.answer_run(**kwargs)
        first = (out / "run2" / "answers.jsonl").read_text(encoding="utf-8").count("\n")
        (out / "run2" / "done" / "s003.jsonl").unlink()  # pretend s003 failed in run 2
        try:
            answer_run.answer_run(**kwargs)
        except SystemExit as exc:
            assert "--resume" in str(exc)
        else:
            raise AssertionError("an existing label must not be overwritten without --resume")
        answer_run.answer_run(**kwargs, resume=True)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["sessions"]) == 2
        assert (out / "run2" / "answers.jsonl").read_text(encoding="utf-8").count("\n") == first
        s003 = next(s for s in load_dir(DATA, "dev") if s.id == "s003")
        assert fakes[-1].usage.by_purpose["answer"] == len(s003.probes) * len(views)  # only s003 was redone
