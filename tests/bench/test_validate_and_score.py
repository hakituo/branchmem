"""Validator, scorer and context baselines (no model needed)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from branchbench.context_baselines import BASELINES, contexts
from branchbench.schema import load_dir, parse_scenario
from branchbench.score import aggregate, score_texts
from branchbench.validate import validate, validate_all

DATA = Path(__file__).resolve().parents[2] / "bench" / "scenarios"

BASE = {
    "id": "t1",
    "title": "测试",
    "branch_op": "edit_user",
    "fact_type": "plan",
    "events": [
        {"id": "a", "parent": None, "role": "user", "text": "我养了一只狗叫豆豆。", "t": 0},
        {"id": "b", "parent": "a", "role": "user", "text": "我要去北京。", "t": 10},
        {"id": "c", "parent": "a", "role": "user", "text": "我要去上海。", "t": 20},
    ],
    "probes": [
        {"id": "q1", "anchor": "c", "kind": "leak", "question": "去哪？",
         "must_include_any": ["上海"], "must_not_include": ["北京"]},
        {"id": "q2", "anchor": "c", "kind": "retain", "question": "狗叫什么？",
         "must_include_any": ["豆豆"], "must_not_include": []},
    ],
}


def _variant(**probe_changes):
    raw = copy.deepcopy(BASE)
    raw["probes"][0].update(probe_changes)
    return parse_scenario(raw)


def test_shipped_scenarios_are_valid():
    scenarios = load_dir(DATA)
    assert len(scenarios) >= 5
    assert validate_all(scenarios) == []


def test_base_scenario_valid_and_path():
    scenario = parse_scenario(BASE)
    assert validate(scenario) == []
    assert [e.id for e in scenario.path("c")] == ["a", "c"]
    assert [e.id for e in scenario.arrived_before("b")] == ["a", "b"]


def test_forbidden_word_on_current_path_is_rejected():
    errors = validate(_variant(must_not_include=["豆豆"]))
    assert any("is on the current branch path" in e for e in errors)


def test_forbidden_word_must_exist_in_arrived_sibling():
    errors = validate(_variant(must_not_include=["广州"]))
    assert any("not in a sibling that arrived before the question" in e for e in errors)


def test_sibling_arriving_after_question_is_not_a_leak_source():
    # asked at b, c has not arrived yet: it cannot be a leak source
    errors = validate(_variant(anchor="b", must_include_any=["北京"], must_not_include=["上海"]))
    assert any("not in a sibling that arrived before the question" in e for e in errors)


def test_answer_word_must_be_on_path():
    errors = validate(_variant(must_include_any=["北京"]))
    assert any("no answer word is on the current branch path" in e for e in errors)


def test_structural_errors():
    raw = copy.deepcopy(BASE)
    raw["events"][1]["parent"] = "c"  # the parent arrives later
    assert any("has not arrived first" in e for e in validate(parse_scenario(raw)))
    raw = copy.deepcopy(BASE)
    raw["events"][2]["parent"] = "b"  # a straight line, no fork
    raw["probes"] = []
    assert any("no fork at all" in e for e in validate(parse_scenario(raw)))


def test_score_and_aggregate():
    scenario = parse_scenario(BASE)
    results = score_texts([scenario], {("t1", "q1"): "你要去上海，不是北京", ("t1", "q2"): "豆豆"})
    by_id = {r.probe_id: r for r in results}
    assert by_id["q1"].leaked and by_id["q1"].hit and not by_id["q1"].correct
    assert by_id["q2"].correct
    summary = aggregate(results)["overall"]
    assert summary["leak_rate"] == 1.0 and summary["correct_rate"] == 0.5


@pytest.mark.parametrize("name,expect_leak", [
    ("all_history", True),
    ("raw_filtered_notes_unfiltered", True),
    ("source_gated", False),
    ("path_only", False),
    ("branchmem", False),
])
def test_context_baselines_separate_leaky_from_gated(name, expect_leak):
    scenarios = load_dir(DATA)
    overall = aggregate(score_texts(scenarios, contexts(scenarios, BASELINES[name])))["overall"]
    assert (overall["leak_rate"] > 0) is expect_leak
    assert overall["hit_rate"] == 1.0  # gating must not hide what the current branch needs
