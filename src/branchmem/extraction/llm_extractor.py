"""An example LLM extractor: one call per segment, notes cite numbered exchanges.

The model sees the segment as numbered exchanges and answers with JSON; every note
lists the exchange numbers it is based on, which are mapped back to event IDs. That
mapping is what makes the notes gateable: provenance comes from the extractor's own
citations, checked against the segment, never from a label.

Bring your own model: ``complete(messages) -> str`` may be sync or async. The prompt
is deliberately neutral and short; it is an example, not a tuned production prompt.
Changing any part of the prompt should bump ``version``.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Sequence, Union

from .contracts import SKIP_NO_REUSABLE_INFO, ExtractionResult, NoteDraft, SegmentInput

Messages = List[Dict[str, str]]
Complete = Callable[[Messages], Union[str, Awaitable[str]]]

SYSTEM_PROMPT = """You turn one conversation segment into memory notes.
Rules:
1. Write short, self-contained notes (one topic each) about what was said or decided.
2. Every note lists the exchange numbers it is based on in "exchanges".
3. Only use what is in the segment. Do not guess. Keep names, numbers and dates verbatim.
4. kind is one of: episode, fact, preference, plan.
5. If nothing is worth remembering, return {"notes": [], "skip_reason": "no_reusable_info"}.
Answer with JSON only: {"notes": [{"kind": "...", "text": "...", "exchanges": [0], "importance": 0.5}]}"""

CONTEXT_HEADER = "Earlier context (for understanding only, do not summarize):"


def group_exchanges(events: Sequence[Mapping[str, Any]]) -> List[List[Mapping[str, Any]]]:
    """Consecutive user messages plus the replies that follow form one exchange."""
    exchanges: List[List[Mapping[str, Any]]] = []
    for event in events:
        role = str(event.get("role") or "").lower()
        if not exchanges or (role == "user" and str(exchanges[-1][-1].get("role")).lower() != "user"):
            exchanges.append([event])
        else:
            exchanges[-1].append(event)
    return exchanges


def build_messages(item: SegmentInput) -> Messages:
    lines: List[str] = []
    if item.context_events:
        lines.append(CONTEXT_HEADER)
        lines += [f"  {e.get('role')}: {e.get('content')}" for e in item.context_events]
        lines.append("")
    for number, exchange in enumerate(group_exchanges(item.events)):
        lines.append(f"[{number}]")
        lines += [f"  {e.get('role')}: {e.get('content')}" for e in exchange]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(lines)}]


def parse_response(text: str, exchanges: Sequence[Sequence[Mapping[str, Any]]]) -> ExtractionResult:
    match = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not match:
        raise ValueError("no_json_object")
    data = json.loads(match.group(0))
    drafts = []
    for raw in data.get("notes") or []:
        numbers = [int(n) for n in raw.get("exchanges") or [] if str(n).lstrip("-").isdigit()]
        evidence = tuple(
            str(event.get("event_id"))
            for n in sorted(set(numbers))
            if 0 <= n < len(exchanges)
            for event in exchanges[n]
        )
        try:
            importance = float(raw.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        drafts.append(NoteDraft(kind=str(raw.get("kind") or "episode"), text=str(raw.get("text") or ""),
                                source_event_ids=evidence, importance=importance))
    skip = "" if drafts else str(data.get("skip_reason") or SKIP_NO_REUSABLE_INFO)
    return ExtractionResult(notes=tuple(drafts), skip_reason=skip)


class LLMExtractor:
    def __init__(self, complete: Complete, *, version: str = "llm-example-1") -> None:
        self._complete = complete
        self.version = version

    async def extract(self, item: SegmentInput) -> ExtractionResult:
        answer = self._complete(build_messages(item))
        if inspect.isawaitable(answer):
            answer = await answer
        result = parse_response(str(answer), group_exchanges(item.events))
        return ExtractionResult(notes=result.notes, skip_reason=result.skip_reason, extractor_version=self.version)
