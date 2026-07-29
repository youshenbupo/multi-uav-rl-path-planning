"""Atomic persistence for live training diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from multiuav.safety import SafetyFilterTelemetry

_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_SECONDS = 0.05


def write_live_training_telemetry(
    path: Path,
    *,
    total_transitions: int,
    cbf: SafetyFilterTelemetry,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically retain CBF failures after every update so interrupted jobs stay diagnosable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    if extra is not None:
        existing.update(extra)
    existing.update({"total_transitions": total_transitions, "cbf": cbf.as_dict()})
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(existing, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_SECONDS)
