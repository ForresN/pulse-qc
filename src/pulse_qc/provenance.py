"""Provenance for a run: code, environment, config and data identity."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pulse_qc.pipeline import RunResult

PACKAGES = ("pulse-qc", "wfdb", "neurokit2", "numpy", "scipy", "pandas", "pyyaml", "matplotlib")


def git_info() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def provenance(result: RunResult) -> dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "config_name": result.config["name"],
        "config_hash": result.config_hash,
        "config": {k: v for k, v in result.config.items() if k not in {"output_dir", "cache_dir"}},
        "records": result.records,
        "cache_stages": result.cache_summary,
    }
