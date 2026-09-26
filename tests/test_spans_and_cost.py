"""Span tree construction, cost attribution and OTLP export."""

from __future__ import annotations

import gzip
import json

import pytest

from observatory.adapters import from_jsonl_file
from observatory.attribution import DEFAULT_PRICES, PriceTable, aggregate, attribute
from observatory.events import normalise
from observatory.otlp import post_otlp, traces_to_otlp, write_otlp
from observatory.spans import SpanKind, Status, build_trace

from .conftest import FIXTURES


def _stream(rows):
    return [normalise(r) for r in rows]


SIMPLE = [
    {"run_id": "r", "seq": 0, "ts": 100.0, "type": "run_start",
     "payload": {"step": 0, "model": "gemini-2.5-flash"}},
    {"run_id": "r", "seq": 1, "ts": 100.0, "type": "llm_call",
     "payload": {"step": 0, "model": "gemini-2.5-flash"}},
    {"run_id": "r", "seq": 2, "ts": 101.0, "type": "llm_result",
     "payload": {"step": 0, "model": "gemini-2.5-flash",
                 "usage": {"prompt_tokens": 1000, "candidates_tokens": 200}}},
    {"run_id": "r", "seq": 3, "ts": 101.0, "type": "tool_call",
     "payload": {"step": 1, "tool": "charge", "args": {"amount": 5}}},
    {"run_id": "r", "seq": 4, "ts": 103.5, "type": "tool_result",
     "payload": {"step": 1, "tool": "charge", "result": {"status": "ok"}}},
    {"run_id": "r", "seq": 5, "ts": 103.5, "type": "run_end", "payload": {"status": "completed"}},
]


def test_span_tree_shape_and_parents():
    trace = build_trace(_stream(SIMPLE))
    walk = trace.walk()
    names = [(depth, span.name) for depth, span in walk]
    assert names == [
        (0, "agent.run"),
        (1, "agent.step[0]"),
        (2, "chat gemini-2.5-flash"),
        (1, "agent.step[1]"),
        (2, "execute_tool charge"),
    ]
    assert trace.root.parent_id is None
    assert all(s.parent_id in trace.by_id() for s in trace.spans[1:])


def test_ids_are_deterministic_so_reruns_diff_empty():
    a, b = build_trace(_stream(SIMPLE)), build_trace(_stream(SIMPLE))
    assert a.trace_id == b.trace_id
    assert [s.span_id for s in a.spans] == [s.span_id for s in b.spans]
    assert len(a.trace_id) == 32 and len(a.spans[0].span_id) == 16


def test_otel_gen_ai_attributes_are_present():
    trace = build_trace(_stream(SIMPLE))
    chat = next(s for s in trace.spans if s.name.startswith("chat"))
    assert chat.kind is SpanKind.CLIENT
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.request.model"] == "gemini-2.5-flash"
    assert chat.attributes["gen_ai.system"] == "gcp.gemini"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 1000
    assert chat.attributes["gen_ai.usage.output_tokens"] == 200
    tool = next(s for s in trace.spans if s.name.startswith("execute_tool"))
    assert tool.attributes["gen_ai.tool.name"] == "charge"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"


def test_durations_come_from_matched_call_and_result():
    trace = build_trace(_stream(SIMPLE))
    chat = next(s for s in trace.spans if s.name.startswith("chat"))
    tool = next(s for s in trace.spans if s.name.startswith("execute_tool"))
    assert chat.duration_ms == pytest.approx(1000.0)
    assert tool.duration_ms == pytest.approx(2500.0)
    assert trace.root.duration_ms == pytest.approx(3500.0)


def test_error_marks_the_enclosing_step_and_the_root():
    rows = SIMPLE[:5] + [
        {"run_id": "r", "seq": 5, "ts": 104.0, "type": "error",
         "payload": {"step": 1, "message": "processor outage"}},
        {"run_id": "r", "seq": 6, "ts": 104.0, "type": "run_end", "payload": {"status": "failed"}},
    ]
    trace = build_trace(_stream(rows))
    step1 = next(s for s in trace.spans if s.name == "agent.step[1]")
    assert step1.status is Status.ERROR
    assert step1.status_message == "processor outage"
    assert trace.root.status is Status.ERROR
    assert any(e.name == "exception" for e in trace.root.events)
    assert trace.root.attributes["agent.run.error_count"] == 1


def test_a_run_with_no_run_end_is_reported_as_truncated():
    trace = build_trace(_stream(SIMPLE[:4]))
    assert trace.root.attributes["agent.run.status"] == "truncated"
    assert trace.root.status is Status.ERROR
    unclosed = next(s for s in trace.spans if s.name.startswith("execute_tool"))
    assert unclosed.status is Status.ERROR


def test_failed_tool_result_sets_span_error():
    rows = SIMPLE[:4] + [
        {"run_id": "r", "seq": 4, "ts": 102.0, "type": "tool_result",
         "payload": {"step": 1, "tool": "charge",
                     "result": {"status": "error", "message": "declined"}}},
        {"run_id": "r", "seq": 5, "ts": 102.0, "type": "run_end", "payload": {"status": "failed"}},
    ]
    trace = build_trace(_stream(rows))
    tool = next(s for s in trace.spans if s.name.startswith("execute_tool"))
    assert tool.status is Status.ERROR and tool.status_message == "declined"


def test_empty_event_list_is_rejected():
    with pytest.raises(ValueError, match="zero events"):
        build_trace([])


# --- cost attribution -----------------------------------------------------------------------

def test_inclusive_cost_rolls_up_and_exclusive_does_not():
    trace = build_trace(_stream(SIMPLE))
    a = attribute(trace, PriceTable())
    root = a.spans[a.order[0]]
    step0 = next(s for s in a.spans.values() if s.name == "agent.step[0]")
    chat = next(s for s in a.spans.values() if s.name.startswith("chat"))
    # 1000 in at $0.30/M plus 200 out at $2.50/M.
    assert chat.cost_usd == pytest.approx(1000 * 0.30 / 1e6 + 200 * 2.50 / 1e6)
    assert step0.cost_usd == 0.0, "a step span spends no tokens of its own"
    assert step0.incl_cost_usd == pytest.approx(chat.cost_usd)
    assert root.incl_cost_usd == pytest.approx(chat.cost_usd)
    assert root.incl_tokens == 1200


def test_self_time_excludes_children():
    trace = build_trace(_stream(SIMPLE))
    a = attribute(trace, PriceTable())
    root = a.spans[a.order[0]]
    assert root.total_ms == pytest.approx(3500.0)
    # The two steps cover 1000 and 2500 ms, so the root has no time of its own.
    assert root.self_ms == pytest.approx(0.0, abs=1e-6)


def test_hottest_ranks_by_inclusive_cost_and_skips_the_root():
    trace = build_trace(_stream(SIMPLE))
    hottest = attribute(trace, PriceTable()).hottest(3)
    assert hottest[0].name in ("agent.step[0]", "chat gemini-2.5-flash")
    assert all(s.depth > 0 for s in hottest)


def test_price_table_longest_prefix_match_and_default():
    table = PriceTable({"gemini-2.5": (1.0, 2.0), "gemini-2.5-flash": (0.1, 0.2),
                        "_default": (9.0, 9.0)})
    assert table.rate("gemini-2.5-flash-002") == (0.1, 0.2)
    assert table.rate("gemini-2.5-pro") == (1.0, 2.0)
    assert table.rate("something-else") == (9.0, 9.0)
    assert table.rate(None) == (9.0, 9.0)


def test_price_table_from_file(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"m": [1.0, 2.0], "_default": [0.0, 0.0]}))
    assert PriceTable.from_file(path).cost("m", 1_000_000, 1_000_000) == pytest.approx(3.0)


def test_default_price_table_has_a_fallback():
    assert "_default" in DEFAULT_PRICES


def test_grouping_by_tool_model_and_step():
    trace = build_trace(_stream(SIMPLE))
    a = attribute(trace, PriceTable())
    assert set(a.by_tool) == {"charge"}
    assert set(a.by_model) == {"gemini-2.5-flash"}
    assert set(a.by_step) == {"0", "1"}
    assert a.by_step["0"]["cost_usd"] > 0 and a.by_step["1"]["cost_usd"] == 0
    assert a.slowest_tools()[0] == ("charge", pytest.approx(2500.0))


def test_aggregate_sums_calls_and_takes_max_of_maxima():
    trace = build_trace(_stream(SIMPLE))
    a = attribute(trace, PriceTable())
    total = aggregate([a, a])
    assert total["by_tool"]["charge"]["calls"] == 2
    assert total["by_tool"]["charge"]["max_ms"] == pytest.approx(2500.0)
    assert total["by_tool"]["charge"]["mean_ms"] == pytest.approx(2500.0)
    assert total["by_tool"]["charge"]["error_rate"] == 0.0


def test_real_fixture_attribution_is_nonzero():
    events = from_jsonl_file(FIXTURES / "durable-runtime" / "real-invoices-1.jsonl")[0]
    a = attribute(build_trace(events), PriceTable())
    assert a.total_tokens > 0
    assert a.total_cost_usd > 0
    assert a.wall_ms > 0


# --- OTLP -----------------------------------------------------------------------------------

def test_otlp_body_is_shaped_like_an_export_request():
    trace = build_trace(_stream(SIMPLE))
    body = traces_to_otlp([trace])
    scope = body["resourceSpans"][0]["scopeSpans"][0]
    assert len(scope["spans"]) == len(trace.spans)
    span = scope["spans"][0]
    assert span["traceId"] == trace.trace_id
    assert "parentSpanId" not in span, "the root span has no parent"
    assert span["startTimeUnixNano"] == str(trace.root.start_ns)
    assert isinstance(span["startTimeUnixNano"], str), "nanos are strings in OTLP/JSON"
    kinds = {s["kind"] for s in scope["spans"]}
    assert kinds <= {"SPAN_KIND_INTERNAL", "SPAN_KIND_CLIENT", "SPAN_KIND_SERVER"}
    chat = next(s for s in scope["spans"] if s["name"].startswith("chat"))
    attrs = {a["key"]: a["value"] for a in chat["attributes"]}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "1000"}
    assert attrs["gen_ai.request.model"] == {"stringValue": "gemini-2.5-flash"}


def test_otlp_encodes_arrays_and_booleans():
    rows = SIMPLE[:3] + [
        {"run_id": "r", "seq": 3, "ts": 102.0, "type": "tool_call",
         "payload": {"step": 1, "tool": "charge", "args": {"a": 1, "b": 2},
                     "compensating": True}},
        {"run_id": "r", "seq": 4, "ts": 102.5, "type": "tool_result",
         "payload": {"step": 1, "tool": "charge", "result": {"status": "ok"}}},
        {"run_id": "r", "seq": 5, "ts": 103.0, "type": "run_end", "payload": {"status": "ok"}},
    ]
    scope = traces_to_otlp([build_trace(_stream(rows))])["resourceSpans"][0]["scopeSpans"][0]
    tool = next(s for s in scope["spans"] if s["name"].startswith("execute_tool"))
    attrs = {a["key"]: a["value"] for a in tool["attributes"]}
    assert attrs["agent.tool.arg_keys"] == {
        "arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}}
    assert attrs["agent.tool.compensating"] == {"boolValue": True}


def test_write_otlp_round_trips_and_gzips_on_suffix(tmp_path):
    trace = build_trace(_stream(SIMPLE))
    plain = tmp_path / "t.json"
    write_otlp([trace], plain)
    assert json.loads(plain.read_text())["resourceSpans"]
    zipped = tmp_path / "t.json.gz"
    write_otlp([trace], zipped)
    assert json.loads(gzip.decompress(zipped.read_bytes()))["resourceSpans"]
    assert zipped.stat().st_size < plain.stat().st_size


def test_export_needs_no_collector_and_reports_a_dead_one_without_raising():
    """The whole point: a missing collector is a returned status, not an exception."""
    status, body = post_otlp([build_trace(_stream(SIMPLE))],
                             "http://127.0.0.1:9/v1/traces", timeout=0.4)
    assert status == 0
    assert body
