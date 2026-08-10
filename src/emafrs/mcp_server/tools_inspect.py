"""Project-inspection dev tools: cache stats, doc listing, git status, test runner.

Local development aid only -- not part of any research claim and not exposed in
production. Subprocess calls use argument lists (never shell=True / string interpolation)
so nothing here is vulnerable to shell injection from tool arguments.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_ALLOWED_MARKERS = {None, "integration", "not integration", "unit"}


def list_configured_sources() -> dict:
    """Which DAL sources have API keys configured right now."""
    from emafrs.data.config import CONFIG

    return {"ok": True, "sources": CONFIG.configured_sources()}


def cache_stats() -> dict:
    """File count and total size of the on-disk API response cache, per provider."""
    cache_dir = _PROJECT_ROOT / "data" / "cache"
    if not cache_dir.exists():
        return {"ok": True, "providers": {}, "note": "cache directory does not exist yet"}
    stats: dict[str, dict[str, Any]] = {}
    for provider_dir in cache_dir.iterdir():
        if not provider_dir.is_dir():
            continue
        files = list(provider_dir.glob("*.json"))
        stats[provider_dir.name] = {
            "n_files": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
        }
    return {"ok": True, "providers": stats}


def list_docs() -> dict:
    """List every docs/*.md file with its first heading, so an agent can see what's
    already documented before writing something that duplicates it."""
    docs_dir = _PROJECT_ROOT / "docs"
    if not docs_dir.exists():
        return {"ok": True, "docs": []}
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        first_heading = ""
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("#"):
                        first_heading = line.strip().lstrip("#").strip()
                        break
        except OSError:
            pass
        docs.append({"file": path.name, "heading": first_heading, "size_bytes": path.stat().st_size})
    return {"ok": True, "docs": docs}


def repo_status() -> dict:
    """git status/log summary (read-only)."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        log = subprocess.run(
            ["git", "log", "--oneline", "-10"], cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return {
            "ok": True,
            "branch": branch,
            "changed_files": status.splitlines() if status else [],
            "recent_commits": log.splitlines() if log else [],
        }
    except (subprocess.SubprocessError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def run_tests(marker: str | None = None) -> dict:
    """Run the project's pytest suite. `marker` must be one of: None, 'unit',
    'integration', 'not integration' (rejects anything else to avoid arbitrary pytest -k
    expression injection from tool input)."""
    if marker not in _ALLOWED_MARKERS:
        return {"ok": False, "error": f"marker must be one of {sorted(m for m in _ALLOWED_MARKERS if m)}"}
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if marker:
        cmd += ["-m", marker]
    try:
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=600)
        tail = "\n".join(proc.stdout.strip().splitlines()[-40:])
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "summary_tail": tail}
    except (subprocess.SubprocessError, OSError) as exc:
        return {"ok": False, "error": str(exc)}
