"""Content-addressed cache for pipeline intermediates.

A cached object is identified by a key built from everything that could change
its value: the hash of the raw input files, the fingerprint of the relevant
config section, the version of the code that produced it, and a stage name.
If any of those change, the key changes and the stage reruns; if none do, the
stored result is returned and the stage is skipped. The cache records every
hit and miss so a run can report which stages actually executed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def code_hash(*objects: Any) -> str:
    """Hash of the source code of the given functions/classes/modules.

    This is deliberately the *source text*, not the git SHA: an uncommitted edit
    to a stage must invalidate that stage's cache entries too."""
    h = hashlib.sha256()
    for obj in objects:
        try:
            h.update(inspect.getsource(obj).encode())
        except (OSError, TypeError):  # e.g. defined interactively: fall back to its name
            h.update(
                f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', obj)}".encode()
            )
    return h.hexdigest()


def config_hash(section: Any) -> str:
    return hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode()).hexdigest()


def make_key(stage: str, **parts: str) -> str:
    payload = {"stage": stage, **parts}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class CacheEvent:
    stage: str
    key: str
    hit: bool


@dataclass
class Cache:
    root: Path
    events: list[CacheEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.pkl"

    def get_or_compute(
        self, stage: str, key: str, compute: Callable[[], Any], meta: dict | None = None
    ) -> Any:
        path = self._path(key)
        if path.exists():
            self.events.append(CacheEvent(stage, key, True))
            with path.open("rb") as fh:
                return pickle.load(fh)  # noqa: S301 - our own files
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        if meta:
            path.with_suffix(".json").write_text(
                json.dumps({"stage": stage, **meta}, indent=2, default=str)
            )
        self.events.append(CacheEvent(stage, key, False))
        return value

    def stage_summary(self) -> dict[str, dict[str, int]]:
        """``{stage: {"hits": n, "misses": m}}`` for this session."""
        out: dict[str, dict[str, int]] = {}
        for ev in self.events:
            d = out.setdefault(ev.stage, {"hits": 0, "misses": 0})
            d["hits" if ev.hit else "misses"] += 1
        return out

    def reset_events(self) -> None:
        self.events.clear()
