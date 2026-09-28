"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        # key: request_id (or user_id) → start time float
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp."""
        import time
        key = request_id or user_id
        self._open[key] = time.time()
        # Giữ partial log entry để record_output bổ sung
        self.logs.append({
            "request_id": key,
            "user_id": user_id,
            "input": text,
            "timestamp": utc_now_iso(),
            "output": None,
            "blocked": None,
            "layer": None,
            "latency_ms": None,
        })

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; update the latest open entry."""
        import time
        key = request_id or user_id
        start = self._open.pop(key, None)
        latency_ms = round((time.time() - start) * 1000, 1) if start else None

        # Tìm entry chưa có output để cập nhật
        for entry in reversed(self.logs):
            if entry.get("request_id") == key and entry.get("output") is None:
                entry["output"] = text
                entry["blocked"] = blocked
                entry["layer"] = layer
                entry["latency_ms"] = latency_ms
                return

        # Fallback: thêm mới nếu không tìm thấy
        self.logs.append({
            "request_id": key,
            "user_id": user_id,
            "input": None,
            "timestamp": utc_now_iso(),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root outputs/ by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
