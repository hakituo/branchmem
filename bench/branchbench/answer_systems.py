"""Answer-level systems. Each one receives every message of a scenario in arrival order
(abandoned branches included) and, at a probe's anchor, hands a context to the answer model.

One object per (scenario, run). ``ingest(event)`` is called in arrival order;
``contexts(probe)`` is called once all events up to the anchor have arrived and returns
``{view name: context text}`` (one system can serve several views from one memory store,
e.g. Mem0 with and without raw-message filtering, so both read the very same memories).

Views:

- ``full_history``: every arrived message, in arrival order.
- ``rolling_summary``: an LLM summary updated every ``SUMMARY_EVERY`` arrived messages,
  plus the last ``SUMMARY_RECENT`` arrived messages (both branch-blind).
- ``mem0``: Mem0 OSS ``add`` per message in arrival order; context = ``search`` results only.
- ``mem0_raw_filtered``: the current branch's raw messages + the same Mem0 search results.
- ``branchmem``: the reference library. Extraction runs on every arrival along the branch
  the arriving message is on (notes cite their source events), retrieval goes through the
  derived gate with the probe anchor in strict mode; context = raw current branch + notes.
- ``path_only``: the current branch's raw messages (upper bound).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from branchbench.schema import Event, Probe, Scenario

ANSWER_SYSTEM = (
    "你是用户的个人助手。下面给出你目前掌握的关于用户和这段对话的信息（对话记录和/或记忆条目）。"
    "请只根据这些信息，用一两句话直接回答用户的问题。"
)

SUMMARY_SYSTEM = (
    "你负责维护一段对话的滚动摘要。给你旧摘要和新到达的若干条消息，输出更新后的完整摘要。"
    "保留人名、地点、数字、决定、偏好和已执行的操作等具体信息，不超过 300 字。只输出摘要本身。"
)

SUMMARY_EVERY = 4
SUMMARY_RECENT = 4
MEM0_SEARCH_LIMIT = 10
NOTES_SEARCH_LIMIT = 10

VIEWS = ("full_history", "rolling_summary", "mem0", "mem0_raw_filtered", "branchmem", "path_only")


def prompt_fingerprint() -> dict:
    from branchmem.extraction.llm_extractor import SYSTEM_PROMPT as EXTRACT_SYSTEM

    texts = {"answer_system": ANSWER_SYSTEM, "summary_system": SUMMARY_SYSTEM, "branchmem_extract_system": EXTRACT_SYSTEM}
    return {name: {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text": text} for name, text in texts.items()}


def render(events: List[Event]) -> str:
    return "\n".join(f"[{event.role}] {event.text}" for event in events)


def answer_messages(context: str, question: str) -> list:
    return [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"{context}\n\n用户的问题：{question}"}]


class System:
    views: tuple = ()

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.arrived: List[Event] = []

    def ingest(self, event: Event) -> None:
        self.arrived.append(event)

    def contexts(self, probe: Probe) -> Dict[str, str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RawSystem(System):
    views = ("full_history", "path_only")

    def contexts(self, probe: Probe) -> Dict[str, str]:
        return {"full_history": render(self.arrived), "path_only": render(self.scenario.path(probe.anchor))}


class RollingSummary(System):
    views = ("rolling_summary",)

    def __init__(self, scenario: Scenario, llm: Callable) -> None:
        super().__init__(scenario)
        self.llm = llm
        self.summary = ""
        self.pending: List[Event] = []

    def ingest(self, event: Event) -> None:
        super().ingest(event)
        self.pending.append(event)
        if len(self.pending) >= SUMMARY_EVERY:
            content = f"旧摘要：\n{self.summary or '（无）'}\n\n新消息：\n{render(self.pending)}"
            self.summary = self.llm([{"role": "system", "content": SUMMARY_SYSTEM},
                                     {"role": "user", "content": content}], purpose="summary").strip()
            self.pending = []

    def contexts(self, probe: Probe) -> Dict[str, str]:
        recent = render(self.arrived[-SUMMARY_RECENT:])
        return {"rolling_summary": f"[对话摘要]\n{self.summary or '（无）'}\n\n[最近消息]\n{recent}"}


def _payload(scenario: Scenario, event: Event) -> Dict[str, Any]:
    return {"event_id": event.id, "conversation_id": scenario.id, "role": event.role, "content": event.text,
            "timestamp": float(event.t), "client_node_id": event.id, "parent_node_id": event.parent or ""}


class BranchmemSystem(System):
    """The reference library through a thin adapter (see the module docstring)."""

    views = ("branchmem",)
    #: two turns per segment; the open tail is treated as closed on every arrival so that
    #: extraction happens as soon as a message arrives (like Mem0's add)
    SEGMENT_MAX_TURNS = 2

    def __init__(self, scenario: Scenario, llm: Callable, workdir: Path) -> None:
        super().__init__(scenario)
        from branchmem.extraction import SegmentParams
        from branchmem.extraction.llm_extractor import LLMExtractor
        from branchmem.notes import NoteStore

        self.payloads: List[Dict[str, Any]] = []
        self.store = NoteStore(Path(workdir) / "notes")
        self.extractor = LLMExtractor(lambda messages: llm(messages, purpose="extract"), version="bench-llm-1")
        self.params = SegmentParams(max_turns=self.SEGMENT_MAX_TURNS)
        self.reports: List[dict] = []

    def _reader(self, _cid: str) -> List[Dict[str, Any]]:
        return self.payloads

    def ingest(self, event: Event) -> None:
        from branchmem.extraction import build_conversation_inputs, run_extraction

        super().ingest(event)
        self.payloads.append(_payload(self.scenario, event))
        items = build_conversation_inputs(
            self.payloads, conversation_id=self.scenario.id, params=self.params,
            now=float(event.t) + self.params.idle_seconds + 1, tree_reader=self._reader,
            cursors=lambda _cid, leaf=event.id: [leaf],
        )
        report = asyncio.run(run_extraction(items, extractor=self.extractor, store=self.store,
                                            now=float(event.t) + self.params.idle_seconds + 1))
        self.reports.append({k: report[k] for k in ("done", "skipped", "failed", "notes_written", "notes_rejected")})

    def contexts(self, probe: Probe) -> Dict[str, str]:
        from branchmem.notes import NoteRetriever, reset_note_retrieval_cache
        from branchmem.visibility import MODE_STRICT, filter_raw_events

        gate = {"conversation_id": self.scenario.id, "active_path_ids": [probe.anchor]}
        raw = filter_raw_events(self.payloads, tree_reader=self._reader, mode=MODE_STRICT, **gate).events
        reset_note_retrieval_cache()
        retriever = NoteRetriever(self.store, tree_reader=self._reader, mode=MODE_STRICT)
        hits = retriever.search(probe.question, limit=NOTES_SEARCH_LIMIT, **gate)
        lines = [f"[{p['role']}] {p['content']}" for p in sorted(raw, key=lambda p: p["timestamp"])]
        # a segment that grows is re-extracted, so identical note texts can repeat; show each once
        memory = "\n".join(f"- {text}" for text in dict.fromkeys(hit["text"] for hit in hits))
        return {"branchmem": "\n".join(lines) + "\n\n[记忆]\n" + (memory or "（无）")}

    def failures(self) -> int:
        return sum(r["failed"] for r in self.reports)


# ---------------------------------------------------------------- Mem0

MEM0_TOOL_PREFIX = "[工具结果] "


def mem0_config(workdir: Path) -> dict:
    """Mem0 OSS config from environment variables. Contains the API keys: never write it out
    as is; ``redact`` it first."""

    llm_cfg = {"model": os.environ["BB_LLM_MODEL"], "temperature": 0, "api_key": os.environ["BB_LLM_API_KEY"],
               "openai_base_url": os.environ["BB_LLM_BASE_URL"]}
    emb_cfg = {"model": os.environ["BB_EMBED_MODEL"], "api_key": os.environ.get("BB_EMBED_API_KEY") or os.environ["BB_LLM_API_KEY"],
               "openai_base_url": os.environ["BB_EMBED_BASE_URL"]}
    # embedding_dims is left out of the embedder on purpose: Mem0 then does not send
    # ``dimensions`` to the API (non-OpenAI providers reject it); the vector store still needs it
    return {
        "llm": {"provider": "openai", "config": llm_cfg},
        "embedder": {"provider": "openai", "config": emb_cfg},
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "branchbench", "path": str(Path(workdir) / "qdrant"),
            "embedding_model_dims": int(os.environ["BB_EMBED_DIMS"]), "on_disk": True}},
        "history_db_path": str(Path(workdir) / "history.db"),
    }


def redact(config: Any) -> Any:
    if isinstance(config, dict):
        return {k: ("<from env>" if "key" in k.lower() else redact(v)) for k, v in config.items()}
    return config


class Mem0System(System):
    views = ("mem0", "mem0_raw_filtered")

    def __init__(self, scenario: Scenario, workdir: Path, run_tag: str) -> None:
        super().__init__(scenario)
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        from mem0 import Memory

        self.memory = Memory.from_config(mem0_config(workdir))
        self.user_id = f"{scenario.id}-{run_tag}"
        self.add_results: List[Any] = []

    def ingest(self, event: Event) -> None:
        super().ingest(event)
        if event.role == "tool":  # Mem0 accepts user/assistant roles; tool output is relabelled
            message = {"role": "assistant", "content": MEM0_TOOL_PREFIX + event.text}
        else:
            message = {"role": event.role, "content": event.text}
        self.add_results.append(self.memory.add([message], user_id=self.user_id))

    def search(self, question: str) -> List[str]:
        # mem0ai 2.x: scoping goes through ``filters``; top-level user_id is rejected
        found = self.memory.search(question, top_k=MEM0_SEARCH_LIMIT, filters={"user_id": self.user_id})
        items = found.get("results", []) if isinstance(found, dict) else list(found or [])
        return [str(item.get("memory", "")) for item in items if isinstance(item, dict)]

    def contexts(self, probe: Probe) -> Dict[str, str]:
        memories = "\n".join(f"- {m}" for m in self.search(probe.question)) or "（无）"
        return {"mem0": f"[记忆]\n{memories}",
                "mem0_raw_filtered": render(self.scenario.path(probe.anchor)) + f"\n\n[记忆]\n{memories}"}

    def close(self) -> None:
        client = getattr(getattr(self.memory, "vector_store", None), "client", None)
        if client is not None and hasattr(client, "close"):
            client.close()


def build_systems(scenario: Scenario, views: List[str], llm: Callable, workdir: Path, run_tag: str,
                  mem0_factory: Optional[Callable] = None) -> List[System]:
    systems: List[System] = []
    if {"full_history", "path_only"} & set(views):
        systems.append(RawSystem(scenario))
    if "rolling_summary" in views:
        systems.append(RollingSummary(scenario, llm))
    if "branchmem" in views:
        systems.append(BranchmemSystem(scenario, llm, Path(workdir) / "branchmem"))
    if {"mem0", "mem0_raw_filtered"} & set(views):
        factory = mem0_factory or (lambda s, w, t: Mem0System(s, w, t))
        systems.append(factory(scenario, Path(workdir) / "mem0", run_tag))
    return systems


def tempdir() -> str:
    return tempfile.mkdtemp(prefix="branchbench-")
