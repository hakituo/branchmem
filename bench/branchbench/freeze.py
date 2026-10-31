"""Freeze the scenario set: record a SHA-256 per file so later runs can prove they used it.

    python -m branchbench freeze          # write bench/scenarios/FROZEN.json
    python -m branchbench check-frozen    # exit 1 if any file changed, appeared or vanished

The manifest of every answer-level run embeds these hashes, and a run on the test split
refuses to start unless the files match ``FROZEN.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FROZEN_NAME = "FROZEN.json"


def file_hashes(directory: Path) -> dict[str, str]:
    directory = Path(directory)
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*.json"))
        if path.name != FROZEN_NAME
    }


def set_digest(hashes: dict[str, str]) -> str:
    basis = "\n".join(f"{name} {digest}" for name, digest in sorted(hashes.items()))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def freeze(directory: Path) -> dict:
    hashes = file_hashes(directory)
    record = {"set_sha256": set_digest(hashes), "files": hashes}
    (Path(directory) / FROZEN_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def check(directory: Path, split: str = "all") -> list[str]:
    """Differences between the files on disk and FROZEN.json (empty list = frozen and unchanged).

    ``split="test"`` checks only ``test/``: the dev split may still be edited after freezing.
    """

    path = Path(directory) / FROZEN_NAME
    if not path.exists():
        return [f"{path} does not exist; the scenario set has not been frozen"]
    frozen = json.loads(path.read_text(encoding="utf-8"))["files"]
    current = file_hashes(directory)
    if split != "all":
        frozen = {k: v for k, v in frozen.items() if k.startswith(split + "/")}
        current = {k: v for k, v in current.items() if k.startswith(split + "/")}
    problems = []
    for name in sorted(set(frozen) | set(current)):
        if name not in current:
            problems.append(f"missing: {name}")
        elif name not in frozen:
            problems.append(f"not in FROZEN.json: {name}")
        elif frozen[name] != current[name]:
            problems.append(f"changed: {name}")
    return problems
