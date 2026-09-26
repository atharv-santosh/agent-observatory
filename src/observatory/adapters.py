"""Adapters that turn each sibling project's on-disk format into canonical event streams.

Two of the three siblings already write the shared JSONL schema (with their own dialect of type
names, which events.normalise handles). agent-sandbox writes a structured scenario result instead,
so it gets a real adapter here: one scenario becomes two runs, the naive branch and the sandboxed
branch, which is what makes it interesting to compare.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .events import Event, parse_jsonl

Trajectory = list[Event]


def _ev(run_id: str, seq: int, ts: float, type_: str, payload: dict[str, Any]) -> Event:
    return Event(run_id=run_id, seq=seq, ts=ts, type=type_, payload=payload)


def from_jsonl_file(path: Path) -> list[Trajectory]:
    """Read a JSONL file that may hold one or many interleaved runs."""
    by_run: dict[str, Trajectory] = {}
    with path.open() as fh:
        for event in parse_jsonl(fh):
            by_run.setdefault(event.run_id, []).append(event)
    out = []
    for events in by_run.values():
        events.sort(key=lambda e: (e.seq, e.ts))
        out.append(events)
    return out


def from_sandbox_scenario(path: Path) -> list[Trajectory]:
    """Synthesise event streams from an agent-sandbox scenario result file."""
    doc = json.loads(path.read_text())
    scenario = str(doc.get("scenario_id") or path.stem)
    category = doc.get("category")
    model = doc.get("model")
    elapsed = float(doc.get("elapsed_seconds") or 0.0)
    out: list[Trajectory] = []

    for branch in ("naive", "sandboxed"):
        body = doc.get(branch)
        if not isinstance(body, dict):
            continue
        calls = [c for c in body.get("calls", []) if isinstance(c, dict)]
        run_id = f"sandbox:{scenario}:{branch}"
        events: Trajectory = []
        ts = 0.0
        # Budget the recorded wall clock across the calls so durations are not all zero.
        slot = (elapsed / max(len(calls) * 2 + 2, 1)) if elapsed else 0.05
        seq = 0

        events.append(_ev(run_id, seq, ts, "run_start", {
            "source": "agent-sandbox", "scenario": scenario, "category": category,
            "branch": branch, "model": model, "is_benign": bool(doc.get("is_benign")),
            "step": 0,
        }))
        seq += 1

        blocked_by: str | None = None
        results = body.get("results_by_config")
        if isinstance(results, dict):
            for cfg in results.values():
                if isinstance(cfg, dict) and cfg.get("blocked_by"):
                    blocked_by = str(cfg["blocked_by"])
                    break

        for step, call in enumerate(calls):
            ts += slot
            events.append(_ev(run_id, seq, ts, "llm_call", {
                "step": step, "model": call.get("model") or model, "kind": call.get("kind"),
                "input_chars": call.get("input_chars") or call.get("trusted_chars"),
            }))
            seq += 1
            ts += slot
            usage = call.get("usage") or {}
            events.append(_ev(run_id, seq, ts, "llm_result", {
                "step": step, "model": call.get("model") or model,
                "usage": usage, "text": call.get("response_text"),
                "function_call": call.get("function_call"),
                "parsed": call.get("parsed"),
            }))
            seq += 1
            fc = call.get("function_call")
            if isinstance(fc, dict) and fc.get("name"):
                ts += slot * 0.2
                events.append(_ev(run_id, seq, ts, "tool_call", {
                    "step": step, "tool": fc["name"], "args": fc.get("args") or {},
                    "arg_sources": call.get("arg_sources"),
                }))
                seq += 1
                ts += slot * 0.2
                if blocked_by and branch == "sandboxed":
                    events.append(_ev(run_id, seq, ts, "error", {
                        "step": step, "message": f"blocked by {blocked_by}",
                        "kind": "policy_block",
                    }))
                    seq += 1
                else:
                    events.append(_ev(run_id, seq, ts, "tool_result", {
                        "step": step, "tool": fc["name"], "result": {"status": "ok"},
                    }))
                    seq += 1

        attack_succeeded = bool(body.get("attack_succeeded"))
        if branch == "sandboxed" and isinstance(results, dict):
            attack_succeeded = any(
                isinstance(c, dict) and c.get("attack_succeeded") for c in results.values())
        ts += slot
        status = "compromised" if attack_succeeded else ("blocked" if blocked_by else "completed")
        events.append(_ev(run_id, seq, ts, "run_end", {
            "status": status, "attack_succeeded": attack_succeeded, "blocked_by": blocked_by,
        }))
        out.append(events)
    return out


def discover(root: Path) -> Iterator[Trajectory]:
    """Walk a directory and yield every trajectory found, whatever format it is stored in."""
    if root.is_file():
        yield from _load_one(root)
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".jsonl", ".json"):
            yield from _load_one(path)


def _load_one(path: Path) -> list[Trajectory]:
    if path.suffix == ".jsonl":
        return from_jsonl_file(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(doc, dict) and "scenario_id" in doc:
        return from_sandbox_scenario(path)
    if isinstance(doc, list):
        # A plain JSON array of events.
        from .events import normalise
        by_run: dict[str, Trajectory] = {}
        for rec in doc:
            if isinstance(rec, dict) and "run_id" in rec:
                ev = normalise(rec)
                by_run.setdefault(ev.run_id, []).append(ev)
        return [sorted(v, key=lambda e: e.seq) for v in by_run.values()]
    return []
