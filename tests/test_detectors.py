"""Each detector against hand-built minimal cases, then all of them against the labelled corpus."""

from __future__ import annotations

import pytest

from observatory.derive import derive
from observatory.detectors import (
    PATHOLOGIES,
    REGISTRY,
    detect_context_exhaustion,
    detect_hallucinated_tool_args,
    detect_oscillation,
    detect_premature_termination,
    detect_retry_loop,
    detect_silent_truncation,
    run_detectors,
)
from observatory.events import normalise
from observatory.validate import evaluate_detectors

TOOLS = [{"name": "search", "parameters": {"type": "object",
                                           "properties": {"query": {"type": "string"},
                                                          "top_k": {"type": "integer"}},
                                           "required": ["query"]}},
         {"name": "submit", "parameters": {"type": "object",
                                           "properties": {"score": {"type": "number"}},
                                           "required": ["score"]}}]


class Build:
    """Minimal trajectory builder, so each detector test states exactly its own premise."""

    def __init__(self, *, plan_steps: int = 6, terminal: str | None = "submit",
                 context_limit: int | None = None, tools: list | None = None) -> None:
        self.rows: list[dict] = []
        self.seq = 0
        self.ts = 0.0
        payload: dict = {"step": 0, "model": "gemini-2.5-flash",
                         "plan": [{"step": i, "tools": tools if tools is not None else TOOLS,
                                   "terminal_tool": terminal} for i in range(plan_steps)]}
        if context_limit:
            payload["context_limit"] = context_limit
        self._add("run_start", payload)

    def _add(self, type_: str, payload: dict, dt: float = 0.1) -> Build:
        self.ts += dt
        self.rows.append({"run_id": "t", "seq": self.seq, "ts": self.ts, "type": type_,
                          "payload": payload})
        self.seq += 1
        return self

    def llm(self, step: int, *, tokens_in: int = 200, tokens_out: int = 30,
            finish: str = "STOP", text: str | None = None) -> Build:
        self._add("llm_call", {"step": step, "model": "gemini-2.5-flash"})
        payload = {"step": step, "model": "gemini-2.5-flash", "finish_reason": finish,
                   "usage": {"prompt_tokens": tokens_in, "candidates_tokens": tokens_out}}
        if text is not None:
            payload["text"] = text
        return self._add("llm_result", payload)

    def tool(self, step: int, name: str, args: dict | None = None, *, ok: bool = True,
             error: str = "boom", result_text: str | None = None) -> Build:
        self._add("tool_call", {"step": step, "tool": name,
                                "args": args if args is not None else {"query": "q"}})
        result = {"status": "ok"} if ok else {"status": "error", "message": error}
        if result_text is not None:
            result["text"] = result_text
        return self._add("tool_result", {"step": step, "tool": name, "result": result})

    def decision(self, step: int, value: str) -> Build:
        return self._add("decision", {"step": step, "var": "route", "value": value})

    def error(self, step: int, message: str) -> Build:
        return self._add("error", {"step": step, "message": message})

    def end(self, status: str = "completed") -> Build:
        return self._add("run_end", {"status": status})

    def facts(self):
        return derive([normalise(r) for r in self.rows])


def test_registry_covers_exactly_the_named_pathologies():
    assert set(REGISTRY) == set(PATHOLOGIES)
    assert len(PATHOLOGIES) == 6


# --- retry loop ------------------------------------------------------------------------------

def test_retry_loop_fires_on_the_same_failing_call_repeated():
    b = Build()
    for step in range(4):
        b.tool(step, "search", {"query": "same"}, ok=False, error="503 upstream")
    finding = detect_retry_loop(b.end("failed").facts())
    assert finding is not None
    assert finding.evidence["tool"] == "search"
    assert finding.evidence["repeats"] == 4
    assert finding.evidence["failures"] == 4


def test_retry_loop_ignores_the_same_tool_with_different_arguments():
    b = Build()
    for step, q in enumerate(["a", "b", "c", "d"]):
        b.tool(step, "search", {"query": q}, ok=False)
    assert detect_retry_loop(b.end("failed").facts()) is None


def test_retry_loop_ignores_a_tool_called_thrice_successfully_and_apart():
    b = Build()
    b.tool(0, "search", {"query": "x"}).tool(1, "submit", {"score": 1.0})
    b.tool(2, "search", {"query": "x"}).tool(3, "submit", {"score": 1.0})
    b.tool(4, "search", {"query": "x"})
    assert detect_retry_loop(b.end().facts()) is None


def test_retry_loop_fires_on_contiguous_identical_calls_even_when_they_succeed():
    """An agent making the same call back to back is not reading its own results."""
    b = Build()
    for step in range(4):
        b.tool(step, "search", {"query": "same"})
    finding = detect_retry_loop(b.end().facts())
    assert finding is not None and finding.evidence["failures"] == 0


def test_retry_loop_skips_compensating_calls():
    rows = Build().rows
    for seq_offset, step in enumerate(range(3)):
        rows.append({"run_id": "t", "seq": 100 + seq_offset, "ts": 10.0 + step,
                     "type": "compensation_step_done",
                     "payload": {"step": step, "action": "refund",
                                 "result": {"status": "ok"}}})
    rows.append({"run_id": "t", "seq": 200, "ts": 30.0, "type": "run_end",
                 "payload": {"status": "compensated"}})
    assert detect_retry_loop(derive([normalise(r) for r in rows])) is None


# --- oscillation -----------------------------------------------------------------------------

def test_oscillation_fires_on_alternating_tools():
    b = Build()
    for step, name in enumerate(["search", "submit"] * 4):
        b.tool(step, name, {"query": "q"} if name == "search" else {"score": 1.0})
    finding = detect_oscillation(b.end("exhausted").facts())
    assert finding is not None
    assert finding.evidence["channel"] == "tool"
    assert finding.evidence["states"] == ["search", "submit"]
    assert finding.evidence["cycles"] >= 3


def test_oscillation_needs_coverage_not_a_coincidence():
    """Four alternating calls buried in a long run is not an oscillating run."""
    b = Build()
    names = ["search", "submit", "search", "submit"] + ["search"] * 14
    for step, name in enumerate(names):
        b.tool(step, name, {"query": "q"} if name == "search" else {"score": 1.0})
    assert detect_oscillation(b.end("failed").facts()) is None


def test_oscillation_ignores_three_states_cycling():
    b = Build()
    for step, name in enumerate(["a", "b", "c"] * 4):
        b.tool(step, name, {})
    assert detect_oscillation(b.end().facts()) is None


def test_oscillation_can_fire_on_the_decision_channel():
    b = Build()
    for step, value in enumerate(["auto", "manual"] * 5):
        b.decision(step, value)
    finding = detect_oscillation(b.end("exhausted").facts())
    assert finding is not None and finding.evidence["channel"] == "decision"


# --- silent truncation -----------------------------------------------------------------------

def test_silent_truncation_fires_when_output_is_capped_and_nobody_complains():
    b = Build().llm(0, finish="MAX_TOKENS", text="cut off mid sen").tool(1, "submit",
                                                                        {"score": 1.0})
    finding = detect_silent_truncation(b.end("completed").facts())
    assert finding is not None
    assert finding.evidence["finish_reasons"] == ["MAX_TOKENS"]


def test_silent_truncation_does_not_fire_when_the_run_raised():
    b = Build().llm(0, finish="MAX_TOKENS").error(1, "output incomplete")
    assert detect_silent_truncation(b.end("failed").facts()) is None


def test_silent_truncation_does_not_fire_when_the_run_did_not_finish_cleanly():
    b = Build().llm(0, finish="MAX_TOKENS")
    assert detect_silent_truncation(b.end("failed").facts()) is None


def test_silent_truncation_sees_a_truncated_tool_result():
    b = Build().llm(0).tool(1, "search", result_text="row 1\n...[truncated 4180 bytes]")
    b.tool(2, "submit", {"score": 1.0})
    finding = detect_silent_truncation(b.end("completed").facts())
    assert finding is not None and finding.evidence["truncation_markers"] >= 1


# --- premature termination -------------------------------------------------------------------

def test_premature_termination_fires_when_the_terminal_tool_is_never_called():
    b = Build(plan_steps=8).llm(0).tool(1, "search").llm(2).tool(3, "search")
    b.llm(4).tool(5, "search").llm(6).tool(7, "search")
    finding = detect_premature_termination(b.end("completed").facts())
    assert finding is not None
    assert any("submit" in reason for reason in finding.evidence["reasons"])


def test_premature_termination_fires_on_low_plan_coverage():
    b = Build(plan_steps=20, terminal=None).llm(0).tool(1, "search")
    finding = detect_premature_termination(b.end("completed").facts())
    assert finding is not None
    assert any("planned steps" in reason for reason in finding.evidence["reasons"])


def test_premature_termination_does_not_fire_on_a_completed_run():
    b = Build(plan_steps=4).llm(0).tool(1, "search").llm(2).tool(3, "submit", {"score": 1.0})
    assert detect_premature_termination(b.end("completed").facts()) is None


def test_premature_termination_does_not_fire_when_the_run_errored():
    """A run that says it broke has not silently given up."""
    b = Build(plan_steps=20).llm(0).error(1, "processor outage")
    assert detect_premature_termination(b.end("failed").facts()) is None


def test_premature_termination_does_not_fire_on_an_unterminated_run():
    b = Build(plan_steps=20).llm(0)
    assert detect_premature_termination(b.facts()) is None


# --- hallucinated tool args ------------------------------------------------------------------

def test_hallucinated_args_catches_a_tool_that_was_never_offered():
    b = Build().llm(0).tool(1, "query_database", {"sql": "select 1"}, ok=False)
    finding = detect_hallucinated_tool_args(b.end("failed").facts())
    assert finding is not None
    assert finding.evidence["kinds"] == ["unknown_tool"]


def test_hallucinated_args_catches_an_argument_outside_the_schema():
    b = Build().llm(0).tool(1, "search", {"query": "q", "dry_run": True}, ok=False)
    finding = detect_hallucinated_tool_args(b.end("failed").facts())
    assert finding is not None and finding.evidence["kinds"] == ["unknown_arg"]


def test_hallucinated_args_catches_a_missing_required_argument():
    b = Build().llm(0).tool(1, "search", {"top_k": 3}, ok=False)
    finding = detect_hallucinated_tool_args(b.end("failed").facts())
    assert finding is not None and finding.evidence["kinds"] == ["missing_required"]


def test_hallucinated_args_catches_a_type_mismatch():
    b = Build().llm(0).tool(1, "search", {"query": 12345}, ok=False)
    finding = detect_hallucinated_tool_args(b.end("failed").facts())
    assert finding is not None and finding.evidence["kinds"] == ["wrong_type"]


def test_hallucinated_args_does_not_fire_on_a_valid_call():
    b = Build().llm(0).tool(1, "search", {"query": "q", "top_k": 3})
    b.tool(2, "submit", {"score": 0.5})
    assert detect_hallucinated_tool_args(b.end().facts()) is None


def test_hallucinated_args_cannot_fire_without_advertised_schemas():
    """No schema, no claim. The sandbox fixtures publish none, and silence is the right answer."""
    b = Build(tools=[]).llm(0).tool(1, "anything", {"whatever": 1})
    assert detect_hallucinated_tool_args(b.end().facts()) is None


def test_boolean_is_not_accepted_where_a_number_is_declared():
    b = Build().llm(0).tool(1, "submit", {"score": True}, ok=False)
    finding = detect_hallucinated_tool_args(b.end("failed").facts())
    assert finding is not None and finding.evidence["kinds"] == ["wrong_type"]


# --- context exhaustion ----------------------------------------------------------------------

def test_context_exhaustion_fires_when_the_prompt_climbs_to_the_declared_window():
    b = Build(context_limit=32_000)
    for step, tokens in enumerate([2_000, 9_000, 17_000, 25_000, 30_000]):
        b.llm(step, tokens_in=tokens)
    b.error(5, "context window exceeded")
    finding = detect_context_exhaustion(b.end("failed").facts())
    assert finding is not None
    assert finding.evidence["near_limit"] is True
    assert finding.evidence["limit_assumed"] is False


def test_context_exhaustion_does_not_fire_on_ordinary_history_growth():
    """Every multi-turn agent's prompt grows several-fold. That is not exhaustion."""
    b = Build()
    for step, tokens in enumerate([200, 400, 700, 1_100, 1_600]):
        b.llm(step, tokens_in=tokens)
    b.error(5, "tool failed")
    assert detect_context_exhaustion(b.end("failed").facts()) is None


def test_context_exhaustion_does_not_fire_when_the_run_ended_cleanly():
    b = Build(context_limit=32_000)
    for step, tokens in enumerate([2_000, 9_000, 17_000, 25_000, 30_000]):
        b.llm(step, tokens_in=tokens)
    b.tool(5, "submit", {"score": 1.0})
    assert detect_context_exhaustion(b.end("completed").facts()) is None


def test_context_exhaustion_needs_four_calls_to_see_a_trend():
    b = Build(context_limit=32_000).llm(0, tokens_in=2_000).llm(1, tokens_in=30_000)
    b.error(2, "context window exceeded")
    assert detect_context_exhaustion(b.end("failed").facts()) is None


def test_context_exhaustion_assumes_a_window_when_none_is_declared_and_says_so():
    b = Build()
    for step, tokens in enumerate([2_000, 7_000, 11_000, 14_000, 15_000]):
        b.llm(step, tokens_in=tokens)
    b.error(5, "too long")
    finding = detect_context_exhaustion(b.end("failed").facts())
    assert finding is not None and finding.evidence["limit_assumed"] is True


# --- against the labelled corpus -------------------------------------------------------------

def test_every_detector_has_measurable_precision_on_the_corpus(small_facts, small_corpus):
    truth = {label.run_id: label.pathologies for label in small_corpus[1]}
    scores = {s.pathology: s for s in evaluate_detectors(small_facts, truth)}
    assert set(scores) == set(PATHOLOGIES)
    for pathology, score in scores.items():
        assert score.fired > 0, f"{pathology} never fired on 420 runs"
        assert score.precision >= 0.75, (
            f"{pathology} precision {score.precision:.3f}, false positives "
            f"{score.examples_fp}")
        assert score.recall >= 0.85, f"{pathology} recall {score.recall:.3f}"


def test_no_detector_fires_on_a_clean_healthy_run():
    b = Build(plan_steps=4).llm(0).tool(1, "search", {"query": "q"}).llm(2)
    b.tool(3, "submit", {"score": 0.9})
    assert run_detectors(b.end("completed").facts()) == []


def test_findings_carry_the_evidence_that_triggered_them(small_facts):
    for facts in small_facts:
        for finding in run_detectors(facts):
            assert finding.evidence, f"{finding.pathology} fired with no evidence"
            assert 0.0 < finding.strength <= 1.0
            assert finding.run_id == facts.run_id


@pytest.mark.parametrize("pathology", PATHOLOGIES)
def test_detectors_tolerate_a_one_event_run(pathology):
    facts = derive([normalise({"run_id": "x", "seq": 0, "ts": 0.0, "type": "run_start",
                               "payload": {}})])
    assert REGISTRY[pathology](facts) is None
