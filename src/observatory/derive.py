"""One pass over a trajectory that produces the facts every downstream stage needs.

Featurisation and the pathology detectors both want the same things: the tool call sequence with
outcomes, the token curve, the advertised tool schemas, what the run said when it stopped. Deriving
that twice would be two chances to disagree, so it happens once here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .events import Event, model_name, tool_name, usage_tokens

_NUM = re.compile(r"\d+")
_HEX = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_PATH = re.compile(r"(?:/[\w.\-]+){2,}")
_WS = re.compile(r"\s+")
_ORDINAL = re.compile(r"[_\-]?\d+$")

TERMINAL_OK = {"completed", "done", "ok", "success", "succeeded"}
TRUNCATION_FINISH = {"length", "max_tokens", "MAX_TOKENS", "LENGTH", "truncated", "token_limit"}
TRUNCATION_MARKERS = ("...[truncated", "[truncated", "<truncated", "output truncated",
                      "(truncated", "…")


def error_shape(message: str) -> str:
    """Collapse an error message to a signature that groups the same failure across runs."""
    text = message.strip().lower()
    text = _QUOTED.sub("<s>", text)
    text = _PATH.sub("<path>", text)
    text = _HEX.sub("<id>", text)
    text = _NUM.sub("#", text)
    text = _WS.sub(" ", text)
    return " ".join(text.split()[:8])[:90] or "empty"


@dataclass(slots=True)
class ToolCall:
    seq: int
    step: int | None
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    error: str | None = None
    compensating: bool = False
    latency_s: float | None = None

    @property
    def arg_signature(self) -> str:
        """Canonical rendering of args, so the same call twice is recognisably the same call."""
        try:
            return json.dumps(self.args, sort_keys=True, default=str)[:400]
        except (TypeError, ValueError):
            return str(sorted(self.args))[:400]


@dataclass(slots=True)
class LlmCall:
    seq: int
    step: int | None
    model: str | None
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str | None = None
    text_len: int = 0
    called_tool: str | None = None
    latency_s: float | None = None


@dataclass(slots=True)
class RunFacts:
    """Everything the rest of the pipeline reads about one run."""

    run_id: str
    n_events: int
    duration_s: float
    status: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    llm_calls: list[LlmCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    error_shapes: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    gates: list[str] = field(default_factory=list)
    steps_seen: int = 0
    plan_steps: int | None = None
    terminal_tool: str | None = None
    advertised_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    context_limit: int | None = None
    truncation_markers: int = 0
    labels: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def tool_sequence(self) -> list[str]:
        return [c.tool for c in self.tool_calls]

    @property
    def tokens_in(self) -> int:
        return sum(c.tokens_in for c in self.llm_calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.tokens_out for c in self.llm_calls)

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for c in self.tool_calls if c.ok is False)

    @property
    def terminated_ok(self) -> bool:
        return self.status in TERMINAL_OK


def derive(events: list[Event]) -> RunFacts:
    """Fold one run's events into RunFacts."""
    events = sorted(events, key=lambda e: (e.seq, e.ts))
    first, last = events[0], events[-1]
    facts = RunFacts(
        run_id=first.run_id,
        n_events=len(events),
        duration_s=max(last.ts - first.ts, 0.0),
        status="unterminated",
    )
    pending_tools: dict[str, list[ToolCall]] = {}
    pending_llm: list[tuple[LlmCall, float]] = []
    steps: set[int] = set()

    for event in events:
        if event.step is not None:
            steps.add(event.step)

        if event.type == "run_start":
            facts.source = event.payload.get("source")
            plan = event.payload.get("plan")
            if isinstance(plan, list):
                facts.plan_steps = len(plan)
                for step in plan:
                    if not isinstance(step, dict):
                        continue
                    if step.get("terminal_tool"):
                        facts.terminal_tool = str(step["terminal_tool"])
                    for spec in step.get("tools") or []:
                        if isinstance(spec, dict) and spec.get("name"):
                            facts.advertised_tools[str(spec["name"])] = spec
            for key in ("scenario", "category", "branch", "task_id", "is_benign", "cohort",
                        "ground_truth", "injected", "model", "seed"):
                if key in event.payload:
                    facts.labels[key] = event.payload[key]
            if isinstance(event.payload.get("context_limit"), int):
                facts.context_limit = event.payload["context_limit"]

        elif event.type == "llm_call":
            for spec in (event.get("request", "tools", default=None)
                         or event.payload.get("tools") or []):
                if isinstance(spec, dict) and spec.get("name"):
                    facts.advertised_tools.setdefault(str(spec["name"]), spec)
            call = LlmCall(seq=event.seq, step=event.step, model=model_name(event))
            facts.llm_calls.append(call)
            pending_llm.append((call, event.ts))

        elif event.type == "llm_result":
            if pending_llm:
                call, started = pending_llm.pop()
                call.latency_s = max(event.ts - started, 0.0)
            else:
                call = LlmCall(seq=event.seq, step=event.step, model=model_name(event))
                facts.llm_calls.append(call)
            call.tokens_in, call.tokens_out = usage_tokens(event)
            call.model = call.model or model_name(event)
            call.finish_reason = _finish(event)
            text = event.get("response", "text", default=None) or event.payload.get("text")
            if isinstance(text, str):
                call.text_len = len(text)
                if any(m in text for m in TRUNCATION_MARKERS):
                    facts.truncation_markers += 1
            fc = (event.get("response", "function_call", default=None)
                  or event.payload.get("function_call"))
            if isinstance(fc, dict) and fc.get("name"):
                call.called_tool = str(fc["name"])

        elif event.type == "tool_call":
            name = tool_name(event) or "unknown"
            args = event.payload.get("args")
            call = ToolCall(seq=event.seq, step=event.step, tool=name,
                            args=args if isinstance(args, dict) else {},
                            compensating=bool(event.payload.get("compensating")))
            facts.tool_calls.append(call)
            pending_tools.setdefault(name, []).append(call)

        elif event.type == "tool_result":
            name = tool_name(event) or "unknown"
            queue = pending_tools.get(name) or pending_tools.get("unknown")
            call = queue.pop(0) if queue else None
            if call is None:
                call = ToolCall(seq=event.seq, step=event.step, tool=name,
                                compensating=bool(event.payload.get("compensating")))
                facts.tool_calls.append(call)
            result = event.payload.get("result")
            status = result.get("status") if isinstance(result, dict) else None
            if isinstance(status, str) and status.lower() in ("error", "failed", "fail"):
                call.ok = False
                call.error = str((result or {}).get("message") or (result or {}).get("error")
                                 or status)
                facts.error_shapes.append(error_shape(call.error))
            else:
                call.ok = True
            blob = json.dumps(result, default=str) if result is not None else ""
            if any(m in blob for m in TRUNCATION_MARKERS):
                facts.truncation_markers += 1

        elif event.type == "decision":
            value = event.payload.get("value")
            label = event.payload.get("_unmapped_type") or event.payload.get("var") or "decision"
            # Strip the step ordinal off decision variable names. Without this, `route_3=auto` and
            # `route_7=auto` are different features, which turns a step index into high-cardinality
            # noise that fragments the healthy population all by itself.
            label = _ORDINAL.sub("", str(label))
            facts.decisions.append(f"{label}={value}" if value is not None else label)

        elif event.type == "human_gate":
            facts.gates.append(str(event.payload.get("outcome") or "wait"))

        elif event.type == "error":
            message = str(event.payload.get("message") or event.payload.get("error") or "error")
            facts.errors.append(message)
            facts.error_shapes.append(error_shape(message))

        elif event.type == "run_end":
            facts.status = str(event.payload.get("status") or "unknown")

    for queue in pending_tools.values():
        for call in queue:
            call.ok = None
    facts.steps_seen = len(steps)
    return facts


def _finish(event: Event) -> str | None:
    for holder in (event.payload, event.get("response", default={}) or {}):
        if not isinstance(holder, dict):
            continue
        for key in ("finish_reason", "finishReason", "stop_reason"):
            if isinstance(holder.get(key), str):
                return holder[key]
    return None
