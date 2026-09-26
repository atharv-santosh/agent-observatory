"""The shared trajectory event schema, plus tolerance for the dialects siblings actually emit."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

# The nine canonical types. Everything in this repo reasons over these and only these.
CANONICAL_TYPES = (
    "run_start",
    "llm_call",
    "llm_result",
    "tool_call",
    "tool_result",
    "decision",
    "human_gate",
    "error",
    "run_end",
)

# Sibling projects grew richer internal vocabularies than the shared nine. durable-agent-runtime
# splits tool_call into intent/commit and human_gate into wait/approved/rejected; its compensation
# events have no canonical equivalent and map onto tool_call/tool_result pairs so that unwinding
# work still shows up in cost attribution. Nothing is dropped silently: anything unmapped becomes
# a `decision` carrying its original type, which is visible in the span tree.
TYPE_ALIASES = {
    "tool_intent": "tool_call",
    "tool_invoke": "tool_call",
    "tool_response": "tool_result",
    "tool_output": "tool_result",
    "human_gate_wait": "human_gate",
    "human_gate_approved": "human_gate",
    "human_gate_rejected": "human_gate",
    "human_gate_timeout": "human_gate",
    "model_call": "llm_call",
    "model_result": "llm_result",
    "compensation_step_done": "tool_result",
    "step_failed": "error",
    "exception": "error",
    "run_finish": "run_end",
    "run_complete": "run_end",
}

GATE_OUTCOMES = {
    "human_gate_wait": "wait",
    "human_gate_approved": "approved",
    "human_gate_rejected": "rejected",
    "human_gate_timeout": "timeout",
}


class SchemaError(ValueError):
    """A line that cannot be read as a trajectory event."""


@dataclass(slots=True)
class Event:
    """One line of a trajectory, normalised onto the canonical vocabulary."""

    run_id: str
    seq: int
    ts: float
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    raw_type: str = ""

    def __post_init__(self) -> None:
        if not self.raw_type:
            self.raw_type = self.type

    @property
    def step(self) -> int | None:
        v = self.payload.get("step")
        return v if isinstance(v, int) else None

    def get(self, *path: str, default: Any = None) -> Any:
        """Walk a nested payload path, returning default on any miss."""
        cur: Any = self.payload
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "seq": self.seq, "ts": self.ts, "type": self.type,
                "payload": self.payload}


def normalise(record: dict[str, Any]) -> Event:
    """Coerce one decoded JSON object into an Event, raising SchemaError if it cannot be."""
    for required in ("run_id", "seq", "ts", "type"):
        if required not in record:
            raise SchemaError(f"missing required field {required!r}")
    raw_type = str(record["type"])
    payload = record.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {"value": payload}
    else:
        payload = dict(payload)

    mapped = TYPE_ALIASES.get(raw_type, raw_type)
    if mapped not in CANONICAL_TYPES:
        # Unknown, but keep it in the stream rather than pretending the run is shorter than it was.
        payload.setdefault("_unmapped_type", raw_type)
        mapped = "decision"
    if raw_type in GATE_OUTCOMES:
        payload.setdefault("outcome", GATE_OUTCOMES[raw_type])
    if raw_type.startswith("compensation"):
        payload.setdefault("compensating", True)
        if "tool" not in payload and "action" in payload:
            payload["tool"] = payload["action"]

    try:
        seq = int(record["seq"])
        ts = float(record["ts"])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"seq/ts not numeric: {exc}") from exc

    return Event(run_id=str(record["run_id"]), seq=seq, ts=ts, type=mapped,
                 payload=payload, raw_type=raw_type)


def parse_jsonl(lines: Iterable[str], *, strict: bool = False) -> Iterator[Event]:
    """Yield Events from JSONL text, skipping blanks and (unless strict) unreadable lines."""
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield normalise(json.loads(line))
        except (json.JSONDecodeError, SchemaError) as exc:
            if strict:
                raise SchemaError(f"line {lineno}: {exc}") from exc


def usage_tokens(event: Event) -> tuple[int, int]:
    """Extract (input, output) token counts from an llm_result under any of the known shapes."""
    if event.type != "llm_result":
        return (0, 0)
    candidates = [
        ("prompt_tokens", "completion_tokens"),
        ("prompt_tokens", "candidates_tokens"),
        ("promptTokenCount", "candidatesTokenCount"),
        ("input_tokens", "output_tokens"),
        ("tokens_in", "tokens_out"),
    ]
    for holder in (event.payload, event.get("usage", default={}) or {},
                   event.get("response", "usage", default={}) or {}, event.get("response",
                   default={}) or {}):
        if not isinstance(holder, dict):
            continue
        for in_key, out_key in candidates:
            if in_key in holder or out_key in holder:
                return (int(holder.get(in_key) or 0), int(holder.get(out_key) or 0))
    return (0, 0)


def model_name(event: Event) -> str | None:
    """Best-effort model identifier for an llm_call or llm_result."""
    for holder in (event.payload, event.get("request", default={}) or {},
                   event.get("response", default={}) or {}):
        if isinstance(holder, dict) and isinstance(holder.get("model"), str):
            return holder["model"]
    return None


def tool_name(event: Event) -> str | None:
    """Tool identifier for a tool_call or tool_result."""
    for key in ("tool", "name", "tool_name", "action"):
        v = event.payload.get(key)
        if isinstance(v, str):
            return v
    fc = event.get("response", "function_call", default=None) or event.get(
        "function_call", default=None)
    if isinstance(fc, dict) and isinstance(fc.get("name"), str):
        return fc["name"]
    return None
