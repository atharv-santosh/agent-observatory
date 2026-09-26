"""Detectors for six named pathologies, each with the evidence that made it fire.

These are the opposite of the clustering: hand-specified rules for failures that already have names.
They are here because a derived taxonomy tells you what is going wrong at population level, and a
detector tells you why this particular run did it, and the two answers are worth having side by side.

Every detector returns the evidence that triggered it, because a detector you cannot audit is a
detector whose precision you cannot measure, and precision is the only number that makes one of
these worth shipping.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .derive import TRUNCATION_FINISH, RunFacts

IDENTIFIERISH = re.compile(r"^[A-Za-z_][\w./:-]{5,}$")

PATHOLOGIES = (
    "retry_loop",
    "oscillation",
    "silent_truncation",
    "premature_termination",
    "hallucinated_tool_args",
    "context_exhaustion",
)


@dataclass(slots=True)
class Finding:
    pathology: str
    run_id: str
    evidence: dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0


Detector = Callable[[RunFacts], Finding | None]


def detect_retry_loop(facts: RunFacts, *, min_repeats: int = 3) -> Finding | None:
    """The same call, with the same arguments, made again after it did not work."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, call in enumerate(facts.tool_calls):
        if call.compensating:
            continue
        groups.setdefault((call.tool, call.arg_signature), []).append(i)

    best: Finding | None = None
    for (tool, sig), idxs in groups.items():
        if len(idxs) < min_repeats:
            continue
        failures = sum(1 for i in idxs if facts.tool_calls[i].ok is False)
        # Either the repeats visibly failed, or they were back to back with nothing between them,
        # which is the shape of a loop that is not even reading its own results.
        contiguous = max(_longest_run(idxs), default=1)
        if failures < 2 and contiguous < len(idxs):
            continue
        strength = min(1.0, 0.4 + 0.15 * len(idxs) + 0.1 * failures)
        finding = Finding("retry_loop", facts.run_id, {
            "tool": tool,
            "repeats": len(idxs),
            "failures": failures,
            "contiguous_run": contiguous,
            "arg_signature": sig[:160],
            "at_calls": idxs[:12],
        }, strength)
        if best is None or finding.evidence["repeats"] > best.evidence["repeats"]:
            best = finding
    return best


def _longest_run(idxs: list[int]) -> list[int]:
    best = run = 1
    for a, b in zip(idxs, idxs[1:], strict=False):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return [best]


def detect_oscillation(facts: RunFacts, *, min_cycles: int = 2,
                       min_coverage: float = 0.5) -> Finding | None:
    """Two states, alternating, with neither winning: A B A B rather than A A A.

    The coverage floor is what separates an oscillating run from a run that happened to alternate
    twice in the middle of doing something else. Four elements of ABAB turn up by chance often
    enough that without it the decision channel alone produced a quarter of the firings.
    """
    for channel, seq, cycles_needed in (("tool", [c.tool for c in facts.tool_calls], min_cycles),
                                        ("decision", facts.decisions, min_cycles + 1)):
        window = _longest_alternation(seq)
        if window is None:
            continue
        start, end = window
        members = sorted(set(seq[start:end]))
        span = end - start
        cycles = span // 2
        coverage = span / len(seq) if seq else 0.0
        if len(members) == 2 and cycles >= cycles_needed and coverage >= min_coverage:
            return Finding("oscillation", facts.run_id, {
                "channel": channel,
                "states": members,
                "cycles": cycles,
                "span": span,
                "coverage": round(coverage, 3),
                "at": start,
            }, min(1.0, 0.5 + 0.12 * cycles))
    return None


def _longest_alternation(seq: list[str]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    for i in range(len(seq)):
        j = i + 2
        while j < len(seq) and seq[j] == seq[j - 2] and seq[j] != seq[j - 1]:
            j += 1
        if j - i >= 4 and (best is None or (j - i) > (best[1] - best[0])):
            best = (i, j)
    return best


def detect_silent_truncation(facts: RunFacts) -> Finding | None:
    """Output cut off at the token cap, or a truncated tool result, and nothing raised."""
    capped = [c for c in facts.llm_calls if c.finish_reason in TRUNCATION_FINISH]
    if not capped and not facts.truncation_markers:
        return None
    if facts.errors:
        return None  # It was not silent; something complained.
    if not facts.terminated_ok:
        return None
    return Finding("silent_truncation", facts.run_id, {
        "capped_calls": len(capped),
        "truncation_markers": facts.truncation_markers,
        "finish_reasons": sorted({c.finish_reason for c in capped if c.finish_reason}),
        "status": facts.status,
    }, 0.6 + 0.1 * min(len(capped) + facts.truncation_markers, 4))


def detect_premature_termination(facts: RunFacts, *, completion_floor: float = 0.6
                                 ) -> Finding | None:
    """Stopped without finishing and without complaining."""
    if facts.errors or facts.status == "unterminated":
        return None
    reasons: list[str] = []
    seq = [c.tool for c in facts.tool_calls]
    if facts.terminal_tool and facts.terminal_tool not in seq:
        reasons.append(f"terminal tool {facts.terminal_tool!r} never called")
    if facts.plan_steps:
        completion = facts.steps_seen / facts.plan_steps
        if completion < completion_floor:
            reasons.append(f"covered {facts.steps_seen}/{facts.plan_steps} planned steps")
    if facts.advertised_tools and not seq:
        reasons.append(f"{len(facts.advertised_tools)} tools offered, none called")
    if not reasons:
        return None
    return Finding("premature_termination", facts.run_id, {
        "reasons": reasons,
        "status": facts.status,
        "steps_seen": facts.steps_seen,
        "plan_steps": facts.plan_steps,
        "tool_calls": len(seq),
    }, min(1.0, 0.5 + 0.2 * len(reasons)))


def detect_hallucinated_tool_args(facts: RunFacts) -> Finding | None:
    """A call the advertised schema does not allow: wrong tool, wrong key, wrong type, invented id."""
    if not facts.advertised_tools:
        return None
    violations: list[dict[str, Any]] = []
    seen_strings = _observed_strings(facts)

    for i, call in enumerate(facts.tool_calls):
        if call.compensating:
            continue
        spec = facts.advertised_tools.get(call.tool)
        if spec is None:
            violations.append({"call": i, "tool": call.tool, "kind": "unknown_tool"})
            continue
        params = spec.get("parameters") if isinstance(spec, dict) else None
        props = params.get("properties") if isinstance(params, dict) else None
        required = params.get("required") if isinstance(params, dict) else None
        if not isinstance(props, dict):
            continue
        extra = sorted(set(call.args) - set(props))
        if extra:
            violations.append({"call": i, "tool": call.tool, "kind": "unknown_arg",
                               "keys": extra})
            continue
        if isinstance(required, list):
            missing = sorted(set(required) - set(call.args))
            if missing:
                violations.append({"call": i, "tool": call.tool, "kind": "missing_required",
                                   "keys": missing})
                continue
        bad_types = [k for k, v in call.args.items()
                     if not _type_ok(v, (props.get(k) or {}).get("type"))]
        if bad_types:
            violations.append({"call": i, "tool": call.tool, "kind": "wrong_type",
                               "keys": sorted(bad_types)})
            continue
        phantom = [k for k, v in call.args.items()
                   if isinstance(v, str) and IDENTIFIERISH.match(v) and v not in seen_strings]
        if phantom:
            violations.append({"call": i, "tool": call.tool, "kind": "phantom_value",
                               "keys": sorted(phantom)})

    if not violations:
        return None
    kinds = sorted({v["kind"] for v in violations})
    # A phantom value on its own is the weakest of the four signals, so it is reported at lower
    # strength rather than being silently treated as equivalent to a schema violation.
    strength = 0.55 if kinds == ["phantom_value"] else min(1.0, 0.7 + 0.1 * len(violations))
    return Finding("hallucinated_tool_args", facts.run_id, {
        "violations": violations[:10],
        "n_violations": len(violations),
        "kinds": kinds,
    }, strength)


def _observed_strings(facts: RunFacts) -> set[str]:
    """Every string the run could have copied an argument from, in call order."""
    out: set[str] = set(facts.advertised_tools)
    for name, spec in facts.advertised_tools.items():
        params = (spec or {}).get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict):
            out.update(props)
        del name
    for call in facts.tool_calls:
        blob = json.dumps(call.args, default=str)
        out.update(re.findall(r"[A-Za-z_][\w./:-]{5,}", blob))
    for label in facts.labels.values():
        if isinstance(label, str):
            out.update(re.findall(r"[A-Za-z_][\w./:-]{5,}", label))
    return out


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_ok(value: Any, declared: Any) -> bool:
    if not isinstance(declared, str) or declared not in _JSON_TYPES:
        return True
    if declared in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, _JSON_TYPES[declared])


def detect_context_exhaustion(facts: RunFacts, *, growth: float = 4.0,
                              limit_fraction: float = 0.8,
                              assumed_limit: int = 16_000) -> Finding | None:
    """The prompt growing every turn until something gives.

    The first version fired on growth ratio alone, which is useless: every multi-turn agent's prompt
    grows several-fold because history accumulates, so the detector ran at 0.42 precision. Exhaustion
    is about approaching a limit, not about growing, so an absolute size floor is required. When the
    run declares its context window that floor comes from the declaration; otherwise a conservative
    assumed window is used and the finding records that it was assumed.
    """
    prompts = [c.tokens_in for c in facts.llm_calls if c.tokens_in]
    if len(prompts) < 4:
        return None
    rising = sum(1 for a, b in zip(prompts, prompts[1:], strict=False) if b > a)
    monotone = rising / (len(prompts) - 1)
    ratio = prompts[-1] / prompts[0] if prompts[0] else 1.0
    limit = facts.context_limit or assumed_limit
    near_limit = max(prompts) >= limit_fraction * limit
    if monotone < 0.8 or ratio < growth or not near_limit:
        return None
    ended_badly = bool(facts.errors) or not facts.terminated_ok or any(
        c.finish_reason in TRUNCATION_FINISH for c in facts.llm_calls)
    if not ended_badly:
        return None
    return Finding("context_exhaustion", facts.run_id, {
        "limit_assumed": facts.context_limit is None,
        "prompt_tokens_first": prompts[0],
        "prompt_tokens_last": prompts[-1],
        "growth_ratio": round(ratio, 2),
        "monotone_fraction": round(monotone, 3),
        "context_limit": limit,
        "near_limit": near_limit,
        "status": facts.status,
    }, min(1.0, 0.6 + 0.1 * min(ratio, 4)))


REGISTRY: dict[str, Detector] = {
    "retry_loop": detect_retry_loop,
    "oscillation": detect_oscillation,
    "silent_truncation": detect_silent_truncation,
    "premature_termination": detect_premature_termination,
    "hallucinated_tool_args": detect_hallucinated_tool_args,
    "context_exhaustion": detect_context_exhaustion,
}


def run_detectors(facts: RunFacts, *, only: list[str] | None = None) -> list[Finding]:
    """Run every detector (or a named subset) over one run."""
    names = only or list(REGISTRY)
    out = []
    for name in names:
        finding = REGISTRY[name](facts)
        if finding is not None:
            out.append(finding)
    return out


def findings_table(all_facts: list[RunFacts]) -> dict[str, list[Finding]]:
    """Findings grouped by pathology across a population."""
    out: dict[str, list[Finding]] = {name: [] for name in REGISTRY}
    for facts in all_facts:
        for finding in run_detectors(facts):
            out[finding.pathology].append(finding)
    return out
