"""Turning a trajectory into a vector, without deciding in advance what the failures are.

The design constraint is that no feature may encode a taxonomy. `is_retry_loop` would be a hand
label smuggled into the input, and a cluster built on it would only rediscover the guess. So the
features are structural: what the tool sequence looks like, how periodic it is, what error text
shapes recur, how the token curve behaves, how the run stopped. A retry loop and an oscillation are
both "a periodic tool sequence" here, separated only by the period the data happens to have.

Every feature belongs to a named block, and each block is L2-normalised separately before being
scaled by a block weight. Without that the tool n-gram block, which has by far the most columns,
quietly decides every distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .derive import TRUNCATION_FINISH, RunFacts

BLOCKS = ("tool", "err", "loop", "term", "tok", "size", "time")

# Chosen by grid search on one synthetic corpus seed and then measured on three other seeds that
# had no part in choosing them; see MEASUREMENTS.md for both numbers. The n-gram block sits below
# 1.0 because tool identity varies a lot inside a single failure mode, and the size block low
# because step and token counts are mostly nuisance variation.
DEFAULT_BLOCK_WEIGHTS: dict[str, float] = {
    "tool": 0.70,
    "err": 1.00,
    "loop": 1.00,
    "term": 1.20,
    "tok": 2.20,
    "size": 0.35,
    "time": 1.10,
}

# Which block each scalar belongs to. The split between "size" and "time" is deliberate and was
# the single biggest quality win in tuning: step and token counts are mostly nuisance variation
# that fragments the healthy population, while tool latency is a real signal, and lumping them in
# one block forced one weight onto both.
SCALAR_BLOCKS: dict[str, str] = {
    "cycle_period": "loop",
    "cycle_repeats": "loop",
    "max_consecutive_repeat": "loop",
    "max_tool_multiplicity": "loop",
    "distinct_tool_ratio": "loop",
    "alternation_span": "loop",
    "fail_ratio": "err",
    "n_error_events": "err",
    "n_distinct_error_shapes": "err",
    "unknown_tool_ratio": "err",
    "bad_arg_ratio": "err",
    "plan_completion": "term",
    "n_tool_calls": "size",
    "n_llm_calls": "size",
    "steps_seen": "size",
    "tokens_in_total": "size",
    "tokens_out_total": "size",
    "mean_output_tokens": "size",
    "duration_s": "time",
    "max_tool_latency_s": "time",
    "max_prompt_tokens": "tok",
    "prompt_growth_ratio": "tok",
    "cap_finish_ratio": "tok",
    "truncation_markers": "tok",
}

N_BUCKETS = 6


@dataclass(slots=True)
class RawFeatures:
    """Pre-vectorisation features for one run: sparse counts, raw scalars, boolean flags."""

    run_id: str
    counts: dict[tuple[str, str], float] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)
    flags: dict[tuple[str, str], bool] = field(default_factory=dict)


def _cycle_signature(seq: list[str]) -> tuple[int, int, int]:
    """Longest periodic run in the sequence as (period, repeats, span).

    A retry loop is this with period 1 and an oscillation is period 2. Deriving both from one
    measurement is deliberate: the detectors get to name them, the features do not.
    """
    best = (0, 0, 0)
    n = len(seq)
    for period in range(1, 5):
        if n <= period:
            break
        i = period
        while i < n:
            if seq[i] != seq[i - period]:
                i += 1
                continue
            start = i - period
            j = i
            while j < n and seq[j] == seq[j - period]:
                j += 1
            span = j - start
            repeats = span // period
            if (repeats, -period) > (best[1], -best[0]):
                best = (period, repeats, span)
            i = j + 1
    return best


def _max_consecutive_repeat(seq: list[str]) -> int:
    best = run = 1 if seq else 0
    for a, b in zip(seq, seq[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _alternation_span(seq: list[str]) -> int:
    """Longest ABAB-style window with two distinct alternating members."""
    best = 0
    for i in range(len(seq)):
        j = i + 2
        while j < len(seq) and seq[j] == seq[j - 2] and seq[j] != seq[j - 1]:
            j += 1
        if j - i >= 3:
            best = max(best, j - i)
    return best


def _arg_violations(facts: RunFacts) -> tuple[float, float]:
    """(unknown tool ratio, bad argument ratio) against whatever schemas the run advertised."""
    if not facts.advertised_tools or not facts.tool_calls:
        return (0.0, 0.0)
    unknown = bad = 0
    real = [c for c in facts.tool_calls if not c.compensating]
    for call in real:
        spec = facts.advertised_tools.get(call.tool)
        if spec is None:
            unknown += 1
            continue
        params = spec.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        required = params.get("required") if isinstance(params, dict) else None
        if isinstance(props, dict) and call.args:
            if set(call.args) - set(props):
                bad += 1
                continue
        if isinstance(required, list) and set(required) - set(call.args):
            bad += 1
    total = float(len(real)) or 1.0
    return (unknown / total, bad / total)


def featurize(facts: RunFacts) -> RawFeatures:
    """Extract structural features for one run."""
    raw = RawFeatures(run_id=facts.run_id)
    seq = [c.tool for c in facts.tool_calls]

    for tool in seq:
        raw.counts[("tool", f"t:{tool}")] = raw.counts.get(("tool", f"t:{tool}"), 0.0) + 1.0
    for a, b in zip(seq, seq[1:], strict=False):
        key = ("tool", f"t2:{a}>{b}")
        raw.counts[key] = raw.counts.get(key, 0.0) + 1.0
    for a, b, c in zip(seq, seq[1:], seq[2:], strict=False):
        key = ("tool", f"t3:{a}>{b}>{c}")
        raw.counts[key] = raw.counts.get(key, 0.0) + 1.0
    for call in facts.tool_calls:
        if call.ok is False:
            key = ("tool", f"tf:{call.tool}")
            raw.counts[key] = raw.counts.get(key, 0.0) + 1.0

    for shape in facts.error_shapes:
        key = ("err", f"e:{shape}")
        raw.counts[key] = raw.counts.get(key, 0.0) + 1.0

    raw.counts[("term", f"term:{facts.status}")] = 1.0
    for outcome in facts.gates:
        key = ("term", f"gate:{outcome}")
        raw.counts[key] = raw.counts.get(key, 0.0) + 1.0
    for decision in facts.decisions[:40]:
        key = ("term", f"d:{decision}")
        raw.counts[key] = raw.counts.get(key, 0.0) + 1.0
    for call in facts.llm_calls:
        if call.finish_reason:
            key = ("tok", f"fin:{call.finish_reason}")
            raw.counts[key] = raw.counts.get(key, 0.0) + 1.0

    period, repeats, span = _cycle_signature(seq)
    prompts = [c.tokens_in for c in facts.llm_calls if c.tokens_in]
    outputs = [c.tokens_out for c in facts.llm_calls]
    latencies = [c.latency_s for c in facts.tool_calls if c.latency_s is not None]
    unknown_ratio, bad_arg_ratio = _arg_violations(facts)
    n_tools = float(len(seq)) or 1.0
    cap_finishes = sum(1 for c in facts.llm_calls if c.finish_reason in TRUNCATION_FINISH)

    raw.scalars = {
        "cycle_period": float(period),
        "cycle_repeats": float(repeats),
        "max_consecutive_repeat": float(_max_consecutive_repeat(seq)),
        "max_tool_multiplicity": (max((seq.count(t) for t in set(seq)), default=0) / n_tools),
        "distinct_tool_ratio": len(set(seq)) / n_tools,
        "alternation_span": float(_alternation_span(seq)),
        "fail_ratio": facts.failed_tool_calls / n_tools,
        "n_error_events": float(len(facts.errors)),
        "n_distinct_error_shapes": float(len(set(facts.error_shapes))),
        "n_tool_calls": float(len(seq)),
        "n_llm_calls": float(len(facts.llm_calls)),
        "steps_seen": float(facts.steps_seen),
        "plan_completion": (facts.steps_seen / facts.plan_steps) if facts.plan_steps else 1.0,
        "duration_s": facts.duration_s,
        "max_tool_latency_s": max(latencies, default=0.0),
        "tokens_in_total": float(facts.tokens_in),
        "tokens_out_total": float(facts.tokens_out),
        "max_prompt_tokens": float(max(prompts, default=0)),
        "prompt_growth_ratio": ((prompts[-1] / prompts[0])
                                if len(prompts) > 1 and prompts[0] else 1.0),
        "cap_finish_ratio": cap_finishes / (len(facts.llm_calls) or 1),
        "truncation_markers": float(facts.truncation_markers),
        "mean_output_tokens": (sum(outputs) / len(outputs)) if outputs else 0.0,
        "unknown_tool_ratio": unknown_ratio,
        "bad_arg_ratio": bad_arg_ratio,
    }
    del span

    raw.flags = {
        ("term", "f:no_run_end"): facts.status == "unterminated",
        ("term", "f:terminal_tool_missing"): bool(
            facts.terminal_tool and facts.terminal_tool not in seq),
        ("term", "f:gate_unresolved"): bool(facts.gates) and all(g == "wait" for g in facts.gates),
        ("err", "f:no_tool_calls"): not seq,
        ("err", "f:all_tools_failed"): bool(seq) and all(c.ok is False for c in facts.tool_calls),
        ("tok", "f:zero_usage"): facts.tokens_in == 0 and facts.tokens_out == 0,
        ("tok", "f:output_capped"): cap_finishes > 0,
    }
    return raw


class Vectorizer:
    """Fits a vocabulary and per-scalar quantile buckets over a corpus, then vectorises it."""

    def __init__(self, *, min_df: int = 3, block_weights: dict[str, float] | None = None,
                 n_buckets: int = N_BUCKETS) -> None:
        self.min_df = min_df
        self.block_weights = dict(block_weights or DEFAULT_BLOCK_WEIGHTS)
        self.n_buckets = n_buckets
        self.columns: list[tuple[str, str]] = []
        self.index: dict[tuple[str, str], int] = {}
        self.edges: dict[str, list[float]] = {}
        self._fitted = False

    def fit(self, corpus: list[RawFeatures]) -> Vectorizer:
        if not corpus:
            raise ValueError("cannot fit a vectorizer on an empty corpus")
        df: dict[tuple[str, str], int] = {}
        for raw in corpus:
            for key in raw.counts:
                df[key] = df.get(key, 0) + 1
            for key, on in raw.flags.items():
                if on:
                    df[key] = df.get(key, 0) + 1
        min_df = min(self.min_df, max(1, len(corpus) // 50))
        kept = sorted(k for k, n in df.items() if n >= min_df)

        for name, block in sorted(SCALAR_BLOCKS.items()):
            values = np.array([r.scalars.get(name, 0.0) for r in corpus], dtype=np.float64)
            qs = np.linspace(0, 100, self.n_buckets + 1)[1:-1]
            edges = sorted(set(np.percentile(values, qs).tolist()))
            self.edges[name] = edges
            for b in range(len(edges) + 1):
                kept.append((block, f"s:{name}:b{b}"))

        self.columns = kept
        self.index = {k: i for i, k in enumerate(kept)}
        self._fitted = True
        return self

    def transform(self, corpus: list[RawFeatures]) -> np.ndarray:
        """Vectorise to a dense, block-normalised, L2-unit matrix of shape (n_runs, n_columns)."""
        if not self._fitted:
            raise RuntimeError("Vectorizer.transform before fit")
        matrix = np.zeros((len(corpus), len(self.columns)), dtype=np.float64)
        for row, raw in enumerate(corpus):
            for key, count in raw.counts.items():
                col = self.index.get(key)
                if col is not None:
                    matrix[row, col] = 1.0 + math.log(count) if count > 1 else 1.0
            for key, on in raw.flags.items():
                col = self.index.get(key)
                if on and col is not None:
                    matrix[row, col] = 1.0
            for name, block in SCALAR_BLOCKS.items():
                bucket = self._bucket(name, raw.scalars.get(name, 0.0))
                # Cumulative rather than one-hot: bucket b lights columns 0..b. A scalar's buckets
                # are ordered, and one-hot throws that away, making "two repeats" exactly as far
                # from "one repeat" as from "seven". That single change was worth more to cluster
                # quality than any weight tuning, because it stopped incidental variation inside the
                # healthy population reading as a categorical difference.
                for b in range(bucket + 1):
                    col = self.index.get((block, f"s:{name}:b{b}"))
                    if col is not None:
                        matrix[row, col] = 1.0
        return self._normalise(matrix)

    def fit_transform(self, corpus: list[RawFeatures]) -> np.ndarray:
        return self.fit(corpus).transform(corpus)

    def _bucket(self, name: str, value: float) -> int:
        edges = self.edges.get(name, [])
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if value <= edges[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def _normalise(self, matrix: np.ndarray) -> np.ndarray:
        for block, weight in self.block_weights.items():
            cols = [i for i, (b, _) in enumerate(self.columns) if b == block]
            if not cols:
                continue
            sub = matrix[:, cols]
            norms = np.linalg.norm(sub, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix[:, cols] = sub / norms * weight
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def column_names(self) -> list[str]:
        return [f"{b}/{n}" for b, n in self.columns]
