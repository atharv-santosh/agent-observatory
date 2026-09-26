"""SQLite store for an analysed corpus.

The web app needs random access to any one run's events to draw its span tree, and re-scanning a
directory of shards per request is not random access. So the analysis writes everything it derived
into one file, with the raw events kept alongside in compressed form so a trace can be rebuilt on
demand rather than pre-rendered for two thousand runs nobody will open.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .events import Event, normalise

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    source       TEXT,
    cohort       TEXT,
    status       TEXT,
    n_events     INTEGER NOT NULL,
    duration_s   REAL NOT NULL,
    n_tool_calls INTEGER NOT NULL,
    n_llm_calls  INTEGER NOT NULL,
    n_errors     INTEGER NOT NULL,
    tokens_in    INTEGER NOT NULL,
    tokens_out   INTEGER NOT NULL,
    cost_usd     REAL NOT NULL,
    wall_ms      REAL NOT NULL,
    cluster_cid  INTEGER,
    cluster_slug TEXT,
    cluster_name TEXT,
    pathologies  TEXT NOT NULL DEFAULT '[]',
    findings     TEXT NOT NULL DEFAULT '[]',
    hotspots     TEXT NOT NULL DEFAULT '[]',
    events_gz    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_cluster ON runs(cluster_slug);
CREATE INDEX IF NOT EXISTS runs_cohort  ON runs(cohort);
CREATE INDEX IF NOT EXISTS runs_status  ON runs(status);
CREATE TABLE IF NOT EXISTS clusters (
    cid             INTEGER PRIMARY KEY,
    slug            TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    size            INTEGER NOT NULL,
    cohesion        REAL NOT NULL,
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    top_features    TEXT NOT NULL,
    representatives TEXT NOT NULL,
    quality         TEXT
);
CREATE TABLE IF NOT EXISTS aggregates (
    scope TEXT NOT NULL,
    key   TEXT NOT NULL,
    body  TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);
"""


class Store:
    """Thin wrapper over one SQLite file. Connections are per-operation, not held open."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialise(self, *, reset: bool = False) -> None:
        if reset and self.path.exists():
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(self.path) + suffix)
                if p.exists():
                    p.unlink()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # --- writes -----------------------------------------------------------------------------

    def set_meta(self, **values: Any) -> None:
        with self.connect() as conn:
            conn.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                             [(k, json.dumps(v)) for k, v in values.items()])

    def put_runs(self, rows: list[dict[str, Any]]) -> None:
        cols = ("run_id", "source", "cohort", "status", "n_events", "duration_s", "n_tool_calls",
                "n_llm_calls", "n_errors", "tokens_in", "tokens_out", "cost_usd", "wall_ms",
                "cluster_cid", "cluster_slug", "cluster_name", "pathologies", "findings",
                "hotspots", "events_gz")
        placeholders = ", ".join("?" * len(cols))
        with self.connect() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO runs({', '.join(cols)}) VALUES ({placeholders})",
                [tuple(row[c] for c in cols) for row in rows])

    def put_clusters(self, rows: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM clusters")
            conn.executemany(
                "INSERT INTO clusters(cid, slug, name, size, cohesion, x, y, top_features,"
                " representatives, quality) VALUES (:cid, :slug, :name, :size, :cohesion, :x, :y,"
                " :top_features, :representatives, :quality)", rows)

    def put_aggregate(self, scope: str, key: str, body: Any) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO aggregates(scope, key, body) VALUES (?, ?, ?)",
                         (scope, key, json.dumps(body)))

    # --- reads ------------------------------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        with self.connect() as conn:
            return {r["key"]: json.loads(r["value"])
                    for r in conn.execute("SELECT key, value FROM meta")}

    def count_runs(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT count(*) AS n FROM runs").fetchone()["n"])

    def list_runs(self, *, cluster: str | None = None, cohort: str | None = None,
                  status: str | None = None, pathology: str | None = None,
                  query: str | None = None, order: str = "cost", limit: int = 100,
                  offset: int = 0) -> list[dict[str, Any]]:
        where, params = [], []
        if cluster:
            where.append("cluster_slug = ?")
            params.append(cluster)
        if cohort:
            where.append("cohort = ?")
            params.append(cohort)
        if status:
            where.append("status = ?")
            params.append(status)
        if pathology:
            where.append("pathologies LIKE ?")
            params.append(f'%"{pathology}"%')
        if query:
            where.append("run_id LIKE ?")
            params.append(f"%{query}%")
        orders = {"cost": "cost_usd DESC", "duration": "wall_ms DESC", "events": "n_events DESC",
                  "run_id": "run_id ASC", "errors": "n_errors DESC, cost_usd DESC"}
        sql = ("SELECT run_id, source, cohort, status, n_events, duration_s, n_tool_calls,"
               " n_llm_calls, n_errors, tokens_in, tokens_out, cost_usd, wall_ms, cluster_slug,"
               " cluster_name, pathologies FROM runs")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {orders.get(order, orders['cost'])} LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self.connect() as conn:
            return [self._run_row(r) for r in conn.execute(sql, params)]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        out = self._run_row(row)
        out["findings"] = json.loads(row["findings"])
        out["hotspots"] = json.loads(row["hotspots"])
        return out

    def get_events(self, run_id: str) -> list[Event]:
        with self.connect() as conn:
            row = conn.execute("SELECT events_gz FROM runs WHERE run_id = ?",
                               (run_id,)).fetchone()
        if row is None:
            return []
        text = gzip.decompress(row["events_gz"]).decode()
        return [normalise(json.loads(line)) for line in text.splitlines() if line.strip()]

    def clusters(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM clusters ORDER BY size DESC").fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["top_features"] = json.loads(r["top_features"])
            item["representatives"] = json.loads(r["representatives"])
            item["quality"] = json.loads(r["quality"]) if r["quality"] else None
            out.append(item)
        return out

    def aggregate(self, scope: str, key: str) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT body FROM aggregates WHERE scope = ? AND key = ?",
                               (scope, key)).fetchone()
        return json.loads(row["body"]) if row else None

    def aggregates(self, scope: str) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, body FROM aggregates WHERE scope = ?",
                                (scope,)).fetchall()
        return {r["key"]: json.loads(r["body"]) for r in rows}

    def cohorts(self) -> list[str]:
        with self.connect() as conn:
            return [r["cohort"] for r in conn.execute(
                "SELECT DISTINCT cohort FROM runs WHERE cohort IS NOT NULL ORDER BY cohort")]

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        out = {k: row[k] for k in row.keys() if k != "events_gz"}
        if isinstance(out.get("pathologies"), str):
            out["pathologies"] = json.loads(out["pathologies"])
        return out


def compress_events(events: list[Event]) -> bytes:
    payload = "\n".join(json.dumps(e.to_dict(), separators=(",", ":")) for e in events)
    return gzip.compress(payload.encode(), compresslevel=6)
