"""The ingest layer, including the sibling dialects it has to tolerate."""

from __future__ import annotations

import json

import pytest

from observatory.adapters import discover, from_jsonl_file, from_sandbox_scenario
from observatory.derive import derive, error_shape
from observatory.events import (
    CANONICAL_TYPES,
    Event,
    SchemaError,
    model_name,
    normalise,
    parse_jsonl,
    tool_name,
    usage_tokens,
)

from .conftest import FIXTURES


def test_canonical_types_are_the_agreed_nine():
    assert set(CANONICAL_TYPES) == {
        "run_start", "llm_call", "llm_result", "tool_call", "tool_result",
        "decision", "human_gate", "error", "run_end",
    }


@pytest.mark.parametrize(("raw", "expected"), [
    ("tool_intent", "tool_call"),
    ("human_gate_wait", "human_gate"),
    ("human_gate_approved", "human_gate"),
    ("compensation_step_done", "tool_result"),
    ("run_start", "run_start"),
])
def test_dialect_aliases_map_onto_the_canonical_vocabulary(raw, expected):
    event = normalise({"run_id": "r", "seq": 0, "ts": 1.0, "type": raw, "payload": {}})
    assert event.type == expected
    assert event.raw_type == raw


def test_unknown_type_is_kept_rather_than_dropped():
    event = normalise({"run_id": "r", "seq": 1, "ts": 1.0, "type": "plan_extended",
                       "payload": {"step": 3}})
    assert event.type == "decision"
    assert event.payload["_unmapped_type"] == "plan_extended"
    assert event.raw_type == "plan_extended"


def test_gate_outcome_is_recovered_from_the_dialect_name():
    approved = normalise({"run_id": "r", "seq": 2, "ts": 1.0, "type": "human_gate_approved",
                          "payload": {}})
    assert approved.payload["outcome"] == "approved"


def test_missing_required_field_is_a_schema_error():
    with pytest.raises(SchemaError):
        normalise({"seq": 0, "ts": 1.0, "type": "run_start"})


def test_non_numeric_seq_is_a_schema_error():
    with pytest.raises(SchemaError):
        normalise({"run_id": "r", "seq": "later", "ts": 1.0, "type": "run_start"})


def test_parse_jsonl_skips_junk_unless_strict():
    lines = ['{"run_id":"r","seq":0,"ts":1,"type":"run_start","payload":{}}', "not json", ""]
    assert len(list(parse_jsonl(lines))) == 1
    with pytest.raises(SchemaError):
        list(parse_jsonl(lines, strict=True))


def test_normalise_is_idempotent_so_the_store_round_trips():
    once = normalise({"run_id": "r", "seq": 0, "ts": 1.0, "type": "tool_intent",
                      "payload": {"tool": "charge"}})
    twice = normalise(once.to_dict())
    assert (twice.type, twice.payload) == (once.type, once.payload)


@pytest.mark.parametrize(("payload", "expected"), [
    ({"usage": {"prompt_tokens": 10, "candidates_tokens": 3}}, (10, 3)),
    ({"usage": {"prompt_tokens": 10, "completion_tokens": 4}}, (10, 4)),
    ({"response": {"usage": {"promptTokenCount": 7, "candidatesTokenCount": 2}}}, (7, 2)),
    ({"response": {"tokens_in": 5, "tokens_out": 1}}, (5, 1)),
    ({}, (0, 0)),
])
def test_usage_tokens_reads_every_shape_the_siblings_emit(payload, expected):
    """Regression: matching on either key alone reported every Gemini completion as 0 output."""
    event = normalise({"run_id": "r", "seq": 1, "ts": 1.0, "type": "llm_result",
                       "payload": payload})
    assert usage_tokens(event) == expected


def test_tool_and_model_names_are_found_under_alternate_keys():
    call = normalise({"run_id": "r", "seq": 0, "ts": 0.0, "type": "llm_result",
                      "payload": {"response": {"function_call": {"name": "submit_score"},
                                               "model": "gemini-2.5-flash"}}})
    assert tool_name(call) == "submit_score"
    assert model_name(call) == "gemini-2.5-flash"


def test_error_shape_collapses_ids_numbers_and_paths():
    a = error_shape("Failed to charge account 'acct-1234' at /var/lib/ledger/db after 3 tries")
    b = error_shape("Failed to charge account 'acct-9999' at /var/lib/other/db after 71 tries")
    assert a == b
    assert error_shape("") == "empty"


# --- real sibling fixtures ------------------------------------------------------------------

def test_durable_runtime_fixtures_load_and_derive():
    path = FIXTURES / "durable-runtime" / "real-invoices-1.jsonl"
    trajectories = from_jsonl_file(path)
    assert len(trajectories) == 1
    facts = derive(trajectories[0])
    assert facts.run_id == "real-invoices-1"
    assert facts.status == "compensated"
    assert facts.tool_calls and facts.llm_calls
    assert facts.errors
    # The compensation events map onto tool results, so the unwinding is visible as work done.
    assert any(call.compensating for call in facts.tool_calls)
    assert facts.tokens_in > 0 and facts.tokens_out > 0


def test_eval_cassettes_load_and_carry_token_usage():
    paths = sorted((FIXTURES / "eval-cassettes").glob("*.jsonl"))
    assert len(paths) >= 20
    total_in = 0
    for path in paths:
        for events in from_jsonl_file(path):
            facts = derive(events)
            assert facts.n_events > 0
            total_in += facts.tokens_in
    assert total_in > 0


def test_sandbox_scenario_adapter_makes_two_runs_per_scenario():
    path = next((FIXTURES / "sandbox-scenarios").rglob("*.json"))
    trajectories = from_sandbox_scenario(path)
    assert len(trajectories) == 2
    branches = {events[0].payload["branch"] for events in trajectories}
    assert branches == {"naive", "sandboxed"}
    for events in trajectories:
        assert events[0].type == "run_start"
        assert events[-1].type == "run_end"
        assert [e.seq for e in events] == sorted(e.seq for e in events)
        facts = derive(events)
        assert facts.advertised_tools == {}  # the sandbox format does not publish schemas
        assert facts.status in {"completed", "blocked", "compromised"}


def test_discover_walks_every_format_in_one_tree():
    seen = {events[0].run_id for events in discover(FIXTURES)}
    assert any(rid.startswith("sandbox:") for rid in seen)
    assert "real-invoices-1" in seen
    assert len(seen) > 300


def test_discover_tolerates_a_json_array_of_events(tmp_path):
    path = tmp_path / "array.json"
    path.write_text(json.dumps([
        {"run_id": "a", "seq": 0, "ts": 1.0, "type": "run_start", "payload": {}},
        {"run_id": "a", "seq": 1, "ts": 2.0, "type": "run_end", "payload": {"status": "ok"}},
    ]))
    trajectories = list(discover(path))
    assert len(trajectories) == 1 and len(trajectories[0]) == 2


def test_interleaved_runs_in_one_file_are_split(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = []
    for seq, run in enumerate(["a", "b", "a", "b"]):
        rows.append({"run_id": run, "seq": seq, "ts": float(seq), "type": "run_start",
                     "payload": {}})
    path.write_text("\n".join(json.dumps(r) for r in rows))
    trajectories = from_jsonl_file(path)
    assert {t[0].run_id for t in trajectories} == {"a", "b"}
    assert all(len(t) == 2 for t in trajectories)


def test_event_get_walks_nested_paths_without_raising():
    event = Event("r", 0, 0.0, "llm_result", {"response": {"usage": {"prompt_tokens": 3}}})
    assert event.get("response", "usage", "prompt_tokens") == 3
    assert event.get("response", "missing", "deep", default="x") == "x"
