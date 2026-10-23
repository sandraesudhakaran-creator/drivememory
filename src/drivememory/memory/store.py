"""SQLite persistence for episodes, preferences and an audit log.

SQLite keeps the project zero-setup. The ``MemoryStore`` interface is small,
so it can be swapped for Postgres + pgvector, DynamoDB or similar in production.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

import numpy as np

from .embeddings import cosine, embed
from .models import Episode, Preference

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    driver_id TEXT NOT NULL,
    trip_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    summary TEXT NOT NULL,
    observations TEXT NOT NULL,
    manual_adjustments INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,
    PRIMARY KEY (driver_id, trip_id)
);
CREATE INDEX IF NOT EXISTS idx_episodes_driver_ts ON episodes(driver_id, ts);
CREATE TABLE IF NOT EXISTS preferences (
    driver_id TEXT NOT NULL,
    key TEXT NOT NULL,
    context TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (driver_id, key, context)
);
CREATE TABLE IF NOT EXISTS audit_log (
    driver_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class MemoryStore:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    # ----------------------------------------------------------- episodes
    def add_episode(self, ep: Episode) -> None:
        vec = embed(ep.summary).astype(np.float32).tobytes()
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?)",
                (ep.driver_id, ep.trip_id, ep.timestamp.isoformat(), ep.summary,
                 json.dumps(ep.observations, default=str), ep.manual_adjustments, vec),
            )
            self.conn.commit()

    def _row_to_episode(self, r) -> Episode:
        return Episode(r[0], r[1], datetime.fromisoformat(r[2]), r[3], json.loads(r[4]), r[5])

    def episodes(self, driver_id: str, limit: int = 20) -> list[Episode]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT driver_id, trip_id, ts, summary, observations, manual_adjustments FROM episodes "
                "WHERE driver_id=? ORDER BY ts DESC LIMIT ?", (driver_id, limit)).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def search_episodes(self, driver_id: str, query: str, k: int = 5) -> list[tuple[float, Episode]]:
        q = embed(query)
        with self._lock:
            rows = self.conn.execute(
                "SELECT driver_id, trip_id, ts, summary, observations, manual_adjustments, embedding "
                "FROM episodes WHERE driver_id=?", (driver_id,)).fetchall()
        scored = [(cosine(q, np.frombuffer(r[6], dtype=np.float32)), self._row_to_episode(r)) for r in rows]
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return scored[:k]

    def count_episodes(self, driver_id: str) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM episodes WHERE driver_id=?", (driver_id,)).fetchone()[0]

    # -------------------------------------------------------- preferences
    def get_preference(self, driver_id: str, key: str, context: str) -> Preference | None:
        with self._lock:
            r = self.conn.execute("SELECT data FROM preferences WHERE driver_id=? AND key=? AND context=?",
                                  (driver_id, key, context)).fetchone()
        return Preference.from_json(r[0]) if r else None

    def put_preference(self, pref: Preference) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO preferences VALUES (?,?,?,?)",
                              (pref.driver_id, pref.key, pref.context, pref.to_json()))
            self.conn.commit()

    def preferences(self, driver_id: str, key_prefix: str = "") -> list[Preference]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT data FROM preferences WHERE driver_id=? AND key LIKE ? ORDER BY key, context",
                (driver_id, key_prefix + "%")).fetchall()
        return [Preference.from_json(r[0]) for r in rows]

    def delete_preferences(self, driver_id: str, key_prefix: str = "", context: str | None = None) -> int:
        sql = "DELETE FROM preferences WHERE driver_id=? AND key LIKE ?"
        args: list = [driver_id, key_prefix + "%"]
        if context is not None:
            sql += " AND context=?"
            args.append(context)
        with self._lock:
            n = self.conn.execute(sql, args).rowcount
            self.conn.commit()
        return n

    def delete_episodes(self, driver_id: str) -> int:
        with self._lock:
            n = self.conn.execute("DELETE FROM episodes WHERE driver_id=?", (driver_id,)).rowcount
            self.conn.commit()
        return n

    # ----------------------------------------------------------- audit log
    def audit(self, driver_id: str, action: str, detail: dict) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO audit_log VALUES (?,?,?,?)",
                              (driver_id, datetime.now().isoformat(), action, json.dumps(detail, default=str)))
            self.conn.commit()

    def audit_log(self, driver_id: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT ts, action, detail FROM audit_log WHERE driver_id=? ORDER BY rowid DESC LIMIT ?",
                (driver_id, limit)).fetchall()
        return [{"ts": r[0], "action": r[1], "detail": json.loads(r[2])} for r in rows]

    def drivers(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT driver_id FROM preferences UNION SELECT driver_id FROM episodes").fetchall()
        return sorted(r[0] for r in rows)
