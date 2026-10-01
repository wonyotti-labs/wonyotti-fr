from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def new_run(root: Path, label: str, settings: dict) -> Path:
    now = datetime.now(UTC)
    destination = root / f"{now:%Y%m%dT%H%M%S}-{label}-{uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source = Path(__file__).parent
    code_hashes = {p.name: sha256(p) for p in sorted(source.glob("*.py"))}
    versions = {}
    for name in ["pandas", "numpy", "pyarrow", "httpx", "scikit-learn", "matplotlib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    lock = Path("uv.lock")
    save_json(destination / "manifest.json", {
        "created_utc": now, "settings": settings, "git_commit": commit,
        "source_sha256": code_hashes, "python": platform.python_version(),
        "dependencies": versions, "uv_lock_sha256": sha256(lock) if lock.exists() else None,
        "execution_mode": "offline_research_no_order_api",
    })
    return destination


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))
