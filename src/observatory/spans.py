"""Flat event streams into a span tree, with OpenTelemetry gen_ai attribute names.

Span and trace ids are derived deterministically from the run id and sequence number rather than
generated randomly, so the same trajectory always produces the same trace. That matters here: this
tool is re-run over the same recorded corpus constantly, and a diff between two runs of the tool
should be empty.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .events import Event, model_name, tool_name, usage_tokens


class SpanKind(str, Enum):
    INTERNAL = "SPAN_KIND_INTERNAL"
    CLIENT = "SPAN_KIND_CLIENT"
    SERVER = "SPAN_KIND_SERVER"


class Status(str, Enum):
    UNSET = "STATUS_CODE_UNSET"
    OK = "STATUS_CODE_OK"
    ERROR = "STATUS_CODE_ERROR"


@dataclass(slots=True)
class SpanEvent:
    name: str
    time_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start_ns: int
    end_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    status: Status = Status.UNSET
    status_message: str = ""
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


@dataclass(slots=True)
class Trace:
    """A run's span tree, plus the flat span list it was built from."""

    run_id: str
    trace_id: str
    spans: list[Span]

    @property
    def root(self) -> Span:
        return self.spans[0]

    def by_id(self) -> dict[str, Span]:
        return {s.span_id: s for s in self.spans}

    def children(self) -> dict[str | None, list[Span]]:
        out: dict[str | None, list[Span]] = {}
        for span in self.spans:
            out.setdefault(span.parent_id, []).append(span)
        for kids in out.values():
            kids.sort(key=lambda s: s.start_ns)
        return out

    def walk(self) -> list[tuple[int, Span]]:
        """Depth-first pre-order traversal as (depth, span) pairs."""
        kids = self.children()
        out: list[tuple[int, Span]] = []

        def rec(span: Span, depth: int) -> None:
            out.append((depth, span))
            for child in kids.get(span.span_id, []):
                rec(child, depth + 1)

        rec(self.root, 0)
        return out


def _hex(*parts: object, width: int = 16) -> str:
    digest = hashlib.blake2b("\x1f".join(str(p) for p in parts).encode(),
                             digest_size=width // 2).hexdigest()
    return digest


def _ns(ts: float) -> int:
    return int(round(ts * 1e9))


def build_trace(events: list[Event]) -> Trace:
    """Assemble a span tree from one run's events."""
    if not events:
        raise ValueError("cannot build a trace from zero events")
    events = sorted(events, key=lambda e: (e.seq, e.ts))
    run_id = events[0].run_id
    trace_id = _hex(run_id, width=32)

    start_ns, end_ns = _ns(events[0].ts), _ns(events[-1].ts)
    root = Span(
        span_id=_hex(run_id, "root"),
        trace_id=trace_id,
        parent_id=None,
        name="agent.run",
        kind=SpanKind.SERVER,
        start_ns=start_ns,
        end_ns=end_ns,
        attributes={"agent.run.id": run_id, "gen_ai.operation.name": "invoke_agent"},
    )
    spans = [root]

    step_spans: dict[int, Span] = {}
    open_llm: list[Span] = []
    open_tools: dict[str | None, list[Span]] = {}
    seen_models: set[str] = set()
    n_errors = 0

    def step_parent(event: Event) -> Span:
        """Return the step span for this event, creating it on first sight."""
        step = event.step
        if step is None:
            return root
        span = step_spans.get(step)
        if span is None:
            span = Span(
                span_id=_hex(run_id, "step", step),
                trace_id=trace_id,
                parent_id=root.span_id,
                name=f"agent.step[{step}]",
                kind=SpanKind.INTERNAL,
                start_ns=_ns(event.ts),
                end_ns=_ns(event.ts),
                attributes={"agent.step.index": step},
            )
            step_spans[step] = span
            spans.append(span)
        span.end_ns = max(span.end_ns, _ns(event.ts))
        return span

    for event in events:
        parent = step_parent(event)
        t = _ns(event.ts)

        if event.type == "run_start":
            for key in ("scenario", "category", "branch", "task_id", "model", "cohort"):
                if event.payload.get(key) is not None:
                    root.attributes[f"agent.run.{key}"] = event.payload[key]

        elif event.type == "llm_call":
            model = model_name(event) or root.attributes.get("agent.run.model") or "unknown"
            span = Span(
                span_id=_hex(run_id, "llm", event.seq),
                trace_id=trace_id,
                parent_id=parent.span_id,
                name=f"chat {model}",
                kind=SpanKind.CLIENT,
                start_ns=t,
                end_ns=t,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": model,
                    "gen_ai.system": _system_of(model),
                },
            )
            if event.payload.get("turn") is not None:
                span.attributes["agent.llm.turn"] = event.payload["turn"]
            spans.append(span)
            open_llm.append(span)
            seen_models.add(model)

        elif event.type == "llm_result":
            span = open_llm.pop() if open_llm else None
            if span is None:
                span = Span(_hex(run_id, "llm", event.seq), trace_id, parent.span_id,
                            "chat unknown", SpanKind.CLIENT, t, t,
                            {"gen_ai.operation.name": "chat"})
                spans.append(span)
            span.end_ns = t
            tok_in, tok_out = usage_tokens(event)
            if tok_in or tok_out:
                span.attributes["gen_ai.usage.input_tokens"] = tok_in
                span.attributes["gen_ai.usage.output_tokens"] = tok_out
            model = model_name(event)
            if model:
                span.attributes["gen_ai.response.model"] = model
            finish = _finish_reason(event)
            if finish:
                span.attributes["gen_ai.response.finish_reasons"] = [finish]
            if span.status is Status.UNSET:
                span.status = Status.OK

        elif event.type == "tool_call":
            name = tool_name(event) or "unknown"
            span = Span(
                span_id=_hex(run_id, "tool", event.seq),
                trace_id=trace_id,
                parent_id=parent.span_id,
                name=f"execute_tool {name}",
                kind=SpanKind.INTERNAL,
                start_ns=t,
                end_ns=t,
                attributes={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": name,
                    "gen_ai.tool.type": "function",
                },
            )
            if event.payload.get("idempotency_key"):
                span.attributes["gen_ai.tool.call.id"] = event.payload["idempotency_key"]
            if event.payload.get("compensating"):
                span.attributes["agent.tool.compensating"] = True
            if isinstance(event.payload.get("args"), dict):
                span.attributes["agent.tool.arg_keys"] = sorted(event.payload["args"])
            spans.append(span)
            open_tools.setdefault(name, []).append(span)

        elif event.type == "tool_result":
            name = tool_name(event) or "unknown"
            queue = open_tools.get(name) or open_tools.get("unknown") or []
            span = queue.pop() if queue else None
            if span is None:
                span = Span(_hex(run_id, "tool", event.seq), trace_id, parent.span_id,
                            f"execute_tool {name}", SpanKind.INTERNAL, t, t,
                            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name})
                spans.append(span)
            span.end_ns = t
            result = event.payload.get("result")
            status = result.get("status") if isinstance(result, dict) else None
            if isinstance(status, str) and status.lower() in ("error", "failed", "fail"):
                span.status = Status.ERROR
                span.status_message = str(
                    (result or {}).get("message") or (result or {}).get("error") or status)
            elif span.status is Status.UNSET:
                span.status = Status.OK

        elif event.type == "human_gate":
            outcome = event.payload.get("outcome")
            if outcome in (None, "wait"):
                span = Span(
                    span_id=_hex(run_id, "gate", event.seq),
                    trace_id=trace_id,
                    parent_id=parent.span_id,
                    name=f"human_gate {event.payload.get('gate_id', '')}".strip(),
                    kind=SpanKind.INTERNAL,
                    start_ns=t,
                    end_ns=t,
                    attributes={"agent.gate.id": event.payload.get("gate_id"),
                                "agent.gate.outcome": "pending"},
                )
                spans.append(span)
                open_tools.setdefault("__gate__", []).append(span)
            else:
                queue = open_tools.get("__gate__") or []
                span = queue.pop() if queue else None
                if span is not None:
                    span.end_ns = t
                    span.attributes["agent.gate.outcome"] = outcome
                    span.status = Status.OK if outcome == "approved" else Status.ERROR

        elif event.type == "decision":
            unmapped = event.payload.get("_unmapped_type")
            label = unmapped or event.payload.get("var") or "decision"
            span = Span(
                span_id=_hex(run_id, "decision", event.seq),
                trace_id=trace_id,
                parent_id=parent.span_id,
                name=f"decision {label}",
                kind=SpanKind.INTERNAL,
                start_ns=t,
                end_ns=t,
                attributes={"agent.decision.value": event.payload.get("value"),
                            "agent.decision.raw_type": event.raw_type},
                status=Status.OK,
            )
            spans.append(span)

        elif event.type == "error":
            n_errors += 1
            message = str(event.payload.get("message") or event.payload.get("error") or "error")
            target = parent if parent is not root else root
            target.status = Status.ERROR
            target.status_message = message
            target.events.append(SpanEvent("exception", t, {
                "exception.message": message,
                "exception.type": event.payload.get("kind") or "AgentError",
            }))
            root.events.append(SpanEvent("exception", t, {"exception.message": message}))

        elif event.type == "run_end":
            status = str(event.payload.get("status") or "unknown")
            root.attributes["agent.run.status"] = status
            root.status = Status.OK if status in ("completed", "done", "ok",
                                                  "success") else Status.ERROR
            if root.status is Status.ERROR:
                root.status_message = status

    for span in open_llm:
        span.status = Status.ERROR
        span.status_message = "no llm_result observed"
    for queue in open_tools.values():
        for span in queue:
            span.status = Status.ERROR
            span.status_message = "no result observed"

    root.attributes["agent.run.error_count"] = n_errors
    root.attributes["agent.run.span_count"] = len(spans)
    if seen_models:
        root.attributes["gen_ai.request.model"] = sorted(seen_models)[0]
    if "agent.run.status" not in root.attributes:
        root.attributes["agent.run.status"] = "truncated"
        root.status = Status.ERROR
        root.status_message = "no run_end observed"

    for span in spans:
        span.end_ns = max(span.end_ns, span.start_ns)
    return Trace(run_id=run_id, trace_id=trace_id, spans=spans)


def _system_of(model: str) -> str:
    low = model.lower()
    if "gemini" in low:
        return "gcp.gemini"
    if "claude" in low:
        return "anthropic"
    if "gpt" in low or low.startswith("o1") or low.startswith("o3"):
        return "openai"
    return "unknown"


def _finish_reason(event: Event) -> str | None:
    for holder in (event.payload, event.get("response", default={}) or {}):
        if not isinstance(holder, dict):
            continue
        for key in ("finish_reason", "finishReason", "stop_reason"):
            v = holder.get(key)
            if isinstance(v, str):
                return v
    return None
