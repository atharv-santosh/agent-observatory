"""Cost, token and time attribution rolled up the span tree.

Totals are the easy part and the useless part. The question worth answering is which subtree is
burning the budget, which means every span needs both an exclusive figure (what it spent itself)
and an inclusive figure (what it and everything under it spent). A step span has no tokens of its
own; all of its cost is inherited from the model calls beneath it, and that is exactly the number
you want when deciding which step to cut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .spans import Span, Status, Trace

# USD per million tokens. Public list prices as of 2026-09; override with a JSON file so the
# numbers in this repo are not silently wrong six months from now.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.6-flash": (0.30, 2.50),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "_default": (0.30, 2.50),
}


class PriceTable:
    """Token prices, with longest-prefix matching so model version suffixes still resolve."""

    def __init__(self, prices: dict[str, tuple[float, float]] | None = None) -> None:
        self.prices = dict(prices or DEFAULT_PRICES)

    @classmethod
    def from_file(cls, path: Path) -> PriceTable:
        raw = json.loads(Path(path).read_text())
        return cls({k: (float(v[0]), float(v[1])) for k, v in raw.items()})

    def rate(self, model: str | None) -> tuple[float, float]:
        if model:
            if model in self.prices:
                return self.prices[model]
            best = max((k for k in self.prices if k != "_default" and model.startswith(k)),
                       key=len, default=None)
            if best:
                return self.prices[best]
        return self.prices.get("_default", (0.0, 0.0))

    def cost(self, model: str | None, tok_in: int, tok_out: int) -> float:
        rin, rout = self.rate(model)
        return (tok_in * rin + tok_out * rout) / 1_000_000


@dataclass(slots=True)
class SpanCost:
    span_id: str
    name: str
    depth: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    self_ms: float = 0.0
    total_ms: float = 0.0
    incl_tokens_in: int = 0
    incl_tokens_out: int = 0
    incl_cost_usd: float = 0.0

    @property
    def incl_tokens(self) -> int:
        return self.incl_tokens_in + self.incl_tokens_out


@dataclass(slots=True)
class RunAttribution:
    """Per-span and grouped attribution for one run."""

    run_id: str
    spans: dict[str, SpanCost]
    order: list[str]
    by_tool: dict[str, dict[str, float]] = field(default_factory=dict)
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    by_step: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def total_cost_usd(self) -> float:
        root = self.spans[self.order[0]]
        return root.incl_cost_usd

    @property
    def total_tokens(self) -> int:
        return self.spans[self.order[0]].incl_tokens

    @property
    def wall_ms(self) -> float:
        return self.spans[self.order[0]].total_ms

    def hottest(self, n: int = 5, *, exclude_root: bool = True) -> list[SpanCost]:
        """Spans ranked by inclusive cost, which is the 'which step is burning budget' answer."""
        items = [self.spans[i] for i in self.order]
        if exclude_root:
            items = [s for s in items if s.depth > 0]
        return sorted(items, key=lambda s: (-s.incl_cost_usd, s.name))[:n]

    def slowest_tools(self, n: int = 5) -> list[tuple[str, float]]:
        ranked = sorted(self.by_tool.items(), key=lambda kv: -kv[1]["total_ms"])
        return [(k, v["total_ms"]) for k, v in ranked[:n]]


def attribute(trace: Trace, prices: PriceTable | None = None) -> RunAttribution:
    """Compute exclusive and inclusive cost/token/time figures for every span in a trace."""
    prices = prices or PriceTable()
    walk = trace.walk()
    depth_of = {span.span_id: depth for depth, span in walk}
    kids = trace.children()

    costs: dict[str, SpanCost] = {}
    for depth, span in walk:
        tok_in = int(span.attributes.get("gen_ai.usage.input_tokens") or 0)
        tok_out = int(span.attributes.get("gen_ai.usage.output_tokens") or 0)
        model = span.attributes.get("gen_ai.response.model") or span.attributes.get(
            "gen_ai.request.model")
        child_ms = sum(c.duration_ms for c in kids.get(span.span_id, []))
        costs[span.span_id] = SpanCost(
            span_id=span.span_id,
            name=span.name,
            depth=depth,
            tokens_in=tok_in,
            tokens_out=tok_out,
            cost_usd=prices.cost(model if isinstance(model, str) else None, tok_in, tok_out),
            self_ms=max(span.duration_ms - child_ms, 0.0),
            total_ms=span.duration_ms,
        )

    # Roll up deepest-first so a parent always sees finished children.
    for depth, span in sorted(walk, key=lambda ds: -ds[0]):
        sc = costs[span.span_id]
        sc.incl_tokens_in = sc.tokens_in
        sc.incl_tokens_out = sc.tokens_out
        sc.incl_cost_usd = sc.cost_usd
        for child in kids.get(span.span_id, []):
            cc = costs[child.span_id]
            sc.incl_tokens_in += cc.incl_tokens_in
            sc.incl_tokens_out += cc.incl_tokens_out
            sc.incl_cost_usd += cc.incl_cost_usd
    del depth_of

    attribution = RunAttribution(run_id=trace.run_id, spans=costs,
                                 order=[s.span_id for _, s in walk])
    _group(trace, costs, attribution)
    return attribution


def _bucket() -> dict[str, float]:
    return {"calls": 0.0, "errors": 0.0, "tokens_in": 0.0, "tokens_out": 0.0,
            "cost_usd": 0.0, "total_ms": 0.0, "max_ms": 0.0}


def _group(trace: Trace, costs: dict[str, SpanCost], out: RunAttribution) -> None:
    for span in trace.spans:
        sc = costs[span.span_id]
        tool = span.attributes.get("gen_ai.tool.name")
        model = span.attributes.get("gen_ai.response.model") or span.attributes.get(
            "gen_ai.request.model")
        step = span.attributes.get("agent.step.index")
        if isinstance(tool, str):
            _add(out.by_tool.setdefault(tool, _bucket()), span, sc)
        if isinstance(model, str) and span.attributes.get("gen_ai.operation.name") == "chat":
            _add(out.by_model.setdefault(model, _bucket()), span, sc)
        if step is not None:
            _add(out.by_step.setdefault(str(step), _bucket()), span, sc, inclusive=True)


def _add(bucket: dict[str, float], span: Span, sc: SpanCost, *, inclusive: bool = False) -> None:
    bucket["calls"] += 1
    if span.status is Status.ERROR:
        bucket["errors"] += 1
    bucket["tokens_in"] += sc.incl_tokens_in if inclusive else sc.tokens_in
    bucket["tokens_out"] += sc.incl_tokens_out if inclusive else sc.tokens_out
    bucket["cost_usd"] += sc.incl_cost_usd if inclusive else sc.cost_usd
    bucket["total_ms"] += sc.total_ms
    bucket["max_ms"] = max(bucket["max_ms"], sc.total_ms)


def aggregate(attributions: list[RunAttribution]) -> dict[str, dict[str, dict[str, float]]]:
    """Sum per-tool / per-model / per-step buckets across a population of runs."""
    out: dict[str, dict[str, dict[str, float]]] = {"by_tool": {}, "by_model": {}, "by_step": {}}
    for attribution in attributions:
        for group in out:
            for key, bucket in getattr(attribution, group).items():
                target = out[group].setdefault(key, _bucket())
                for field_name, value in bucket.items():
                    if field_name == "max_ms":
                        target[field_name] = max(target[field_name], value)
                    else:
                        target[field_name] += value
    for group in out:
        for bucket in out[group].values():
            calls = bucket["calls"] or 1.0
            bucket["mean_ms"] = bucket["total_ms"] / calls
            bucket["error_rate"] = bucket["errors"] / calls
    return out
