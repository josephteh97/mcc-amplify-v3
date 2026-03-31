"""
base_agent.py — Abstract BaseAgent for the MCC-Amplify-v2 Decentralized Pipeline
==================================================================================
Every agent (Validation, BIM-Translator, …) inherits from BaseAgent.

Contract enforced on every subclass:
  • It must live inside a directory that contains a local tools.py
    and a local memory.sqlite (auto-created on first run).
  • Before any geometry payload is finalised, check_memory_before_finalizing()
    MUST be called — the decorator @memory_first enforces this automatically
    when wrapping the agent's core processing method.
  • run() is the single public entry point; subclasses implement _process().

Architecture note:
  BaseAgent carries NO tool references.  Each subclass imports its own local
  tools module so tool sets stay completely isolated between agents.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# ── Decorator: forces a memory check before the decorated method executes ─────

def memory_first(method: Callable) -> Callable:
    """
    Decorator that guarantees check_memory_before_finalizing() is called
    before the wrapped method runs.

    Usage inside a BaseAgent subclass:

        @memory_first
        def _process(self, payload: dict) -> dict:
            ...
    """
    def wrapper(self: "BaseAgent", payload: dict, *args, **kwargs) -> Any:
        # Pull feature_signature from the incoming payload (Detection Agent output)
        feature_sig = (
            payload.get("feature_signature")
            or payload.get("geometry", {}).get("feature_signature")
            or ""
        )
        memory_hint = self.check_memory_before_finalizing(feature_sig, payload)
        if memory_hint:
            self._log(f"[memory_first] Applying memory hint: {memory_hint.get('summary', '')[:120]}")
            # Inject into payload so _process() can act on it
            payload = {**payload, "_memory_hint": memory_hint}
        return method(self, payload, *args, **kwargs)
    wrapper.__name__ = method.__name__
    return wrapper


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all agents in the v2 decentralized pipeline.

    Subclasses MUST:
      • Set self.agent_dir to the directory of the concrete agent (use Path(__file__).parent).
      • Implement _process(payload: dict) -> dict.
      • Call super().__init__() before anything else.

    The run() method wraps _process() with:
      1. Pre-run memory lookup (via @memory_first on _process, or explicit call).
      2. SQLite run-logging.
      3. Error capture and structured return.
    """

    def __init__(self, agent_dir: Path):
        self.agent_dir  = Path(agent_dir)
        self.agent_name = self.__class__.__name__
        self.run_id     = str(uuid.uuid4())
        self._db_path   = self.agent_dir / "memory.sqlite"
        self._mjson_path = self.agent_dir / "memory.json"

        self._ensure_base_schema()
        self._log(f"Initialised. run_id={self.run_id}")

    # ── Tool loader (shared utility, no tool code shared) ─────────────────────

    @staticmethod
    def _load_agent_tools(label: str, path: Path) -> Any:
        """
        Load an agent's private tools.py under a unique sys.modules key so
        two agents imported in the same process don't collide on the name 'tools'.
        """
        spec = importlib.util.spec_from_file_location(label, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, payload: dict) -> dict:
        """
        Execute the agent.

        Args:
            payload: Input dict produced by the upstream agent or controller.

        Returns:
            Result dict with at minimum:
              {"ok": bool, "run_id": str, "agent": str, ...agent-specific keys...}
        """
        self._log("=== run() started ===")
        self._save_run_start(payload)

        try:
            result = self._process(payload)
            result.setdefault("ok",     True)
            result.setdefault("run_id", self.run_id)
            result.setdefault("agent",  self.agent_name)
            self._save_run_end(success=True)
            self._log("=== run() completed successfully ===")
            return result

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self._log(f"[ERROR] {error_msg}")
            self._save_run_end(success=False, error=error_msg)
            return {
                "ok":     False,
                "run_id": self.run_id,
                "agent":  self.agent_name,
                "error":  error_msg,
            }

    # ── Abstract: subclasses implement this ───────────────────────────────────

    @abstractmethod
    def _process(self, payload: dict) -> dict:
        """Core agent logic. Must return a result dict."""
        ...

    # ── Memory: check before finalising geometry ──────────────────────────────

    def check_memory_before_finalizing(
        self,
        feature_signature: str,
        payload: dict,
    ) -> dict | None:
        """
        Query local memory.sqlite for prior corrections relevant to this payload.

        Returns the highest-scoring correction dict, or None if nothing relevant.

        Agents call this BEFORE locking in any geometry decision.  The
        @memory_first decorator on _process() calls this automatically.
        """
        try:
            con  = sqlite3.connect(self._db_path)
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT * FROM agent_corrections "
                "WHERE feature_signature=? OR feature_signature='*' "
                "ORDER BY success_count DESC, timestamp DESC LIMIT 5",
                (feature_signature,),
            ).fetchall()
            con.close()
        except sqlite3.OperationalError:
            # Schema not yet initialised or table missing — safe to ignore.
            return None

        if not rows:
            return None

        best = dict(rows[0])
        try:
            best["correction_data"] = json.loads(best.get("correction_data") or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        best["summary"] = (
            f"[{best.get('error_code','')}] {best.get('correction_desc','')[:100]}"
        )
        return best

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] [{self.agent_name}] {msg}", flush=True)

    # ── Internal SQLite helpers ───────────────────────────────────────────────

    def _ensure_base_schema(self) -> None:
        """Create the universal base tables every agent shares."""
        con = sqlite3.connect(self._db_path)
        con.executescript("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id       TEXT PRIMARY KEY,
            agent_name   TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            status       TEXT,          -- 'started'|'success'|'failure'
            input_sig    TEXT,
            error_msg    TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_corrections (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            feature_signature TEXT NOT NULL,
            error_code        TEXT,
            error_pattern     TEXT NOT NULL,
            correction_desc   TEXT NOT NULL,
            correction_data   TEXT,
            success_count     INTEGER DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_ar_ts   ON agent_runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_ac_feat ON agent_corrections(feature_signature);
        """)
        con.commit()
        con.close()

    def _save_run_start(self, payload: dict) -> None:
        sig = payload.get("feature_signature", "")
        con = sqlite3.connect(self._db_path)
        with con:
            con.execute(
                "INSERT OR REPLACE INTO agent_runs "
                "(run_id, agent_name, timestamp, status, input_sig) VALUES (?,?,?,?,?)",
                (self.run_id, self.agent_name,
                 datetime.now().isoformat(), "started", sig),
            )
        con.close()

    def _save_run_end(self, success: bool, error: str = "") -> None:
        status = "success" if success else "failure"
        con    = sqlite3.connect(self._db_path)
        with con:
            con.execute(
                "UPDATE agent_runs SET status=?, error_msg=? WHERE run_id=?",
                (status, error, self.run_id),
            )
        con.close()

    # ── Correction memory helpers (called by subclasses) ──────────────────────

    def _save_correction(
        self,
        feature_signature: str,
        error_code: str,
        error_pattern: str,
        correction_desc: str,
        correction_data: dict | None = None,
    ) -> None:
        """
        Persist a successful correction to local memory so future runs can
        apply it proactively (retrieved via check_memory_before_finalizing).
        """
        ts        = datetime.now().isoformat()
        data_json = json.dumps(correction_data) if correction_data else None
        con       = sqlite3.connect(self._db_path)
        with con:
            existing = con.execute(
                "SELECT id FROM agent_corrections "
                "WHERE feature_signature=? AND error_code=? AND error_pattern=? LIMIT 1",
                (feature_signature, error_code, error_pattern),
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE agent_corrections SET success_count=success_count+1, timestamp=? WHERE id=?",
                    (ts, existing[0]),
                )
            else:
                con.execute(
                    "INSERT INTO agent_corrections "
                    "(timestamp, feature_signature, error_code, error_pattern, "
                    "correction_desc, correction_data) VALUES (?,?,?,?,?,?)",
                    (ts, feature_signature, error_code, error_pattern,
                     correction_desc, data_json),
                )
        con.close()

    def _load_lessons_learned(self) -> list[dict]:
        """
        Read the human-readable memory.json "Lessons Learned" log.
        Returns a list of lesson dicts (empty list if file absent or malformed).
        """
        if not self._mjson_path.exists():
            return []
        try:
            data = json.loads(self._mjson_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("lessons", [])
        except (json.JSONDecodeError, OSError):
            return []

    def _append_lesson(self, lesson: dict) -> None:
        """Append a lesson to memory.json (human-readable audit trail)."""
        lessons = self._load_lessons_learned()
        lessons.append({**lesson, "timestamp": datetime.now().isoformat()})
        self._mjson_path.write_text(
            json.dumps(lessons, indent=2), encoding="utf-8"
        )
