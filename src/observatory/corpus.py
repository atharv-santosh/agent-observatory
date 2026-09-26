"""A synthetic corpus with failure modes planted on purpose, so recovery can be measured.

This is the load-bearing part of the repo. Clustering a pile of real trajectories produces clusters
that look plausible and cannot be checked, which is how unvalidated clustering stays unvalidated.
Here the generator knows what it planted, the clusterer is never shown it, and purity against the
plant is a real number.

Two rules keep the exercise honest. The ground truth is written to a sidecar file and never into the
event stream, so no feature can read it even by accident. And the modes are deliberately not cleanly
separable: they share a tool vocabulary, their step counts and token budgets overlap, and a sampled
fraction of runs in every mode carry an incidental second pathology that is recorded truthfully in
the labels. A corpus where every mode is obvious would prove nothing.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODES = (
    "healthy",
    "retry_storm",
    "oscillation",
    "silent_truncation",
    "premature_stop",
    "arg_hallucination",
    "context_exhaustion",
    "tool_outage",
    "gate_abandoned",
    "slow_tool",
)

BASELINE_WEIGHTS: dict[str, float] = {
    "healthy": 0.40,
    "retry_storm": 0.10,
    "oscillation": 0.07,
    "silent_truncation": 0.07,
    "premature_stop": 0.08,
    "arg_hallucination": 0.08,
    "context_exhaustion": 0.06,
    "tool_outage": 0.07,
    "gate_abandoned": 0.04,
    "slow_tool": 0.03,
}

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_docs": {"properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"]},
    "read_file": {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    "write_file": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]},
    "list_files": {"properties": {}, "required": []},
    "run_query": {"properties": {"sql": {"type": "string"}, "limit": {"type": "integer"}},
                  "required": ["sql"]},
    "fetch_url": {"properties": {"url": {"type": "string"}}, "required": ["url"]},
    "charge": {"properties": {"account": {"type": "string"}, "amount": {"type": "number"}},
               "required": ["account", "amount"]},
    "refund": {"properties": {"account": {"type": "string"}, "amount": {"type": "number"}},
               "required": ["account", "amount"]},
    "send_email": {"properties": {"to": {"type": "string"}, "subject": {"type": "string"},
                                  "body": {"type": "string"}}, "required": ["to", "body"]},
    "create_ticket": {"properties": {"title": {"type": "string"}, "priority": {"type": "integer"}},
                      "required": ["title"]},
    "lookup_customer": {"properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"]},
    "update_record": {"properties": {"record_id": {"type": "string"}, "fields": {"type": "object"}},
                      "required": ["record_id", "fields"]},
    "submit_report": {"properties": {"summary": {"type": "string"}, "score": {"type": "number"}},
                      "required": ["summary"]},
    "escalate": {"properties": {"reason": {"type": "string"}}, "required": ["reason"]},
}

TASKS = ("triage_invoice", "reconcile_ledger", "answer_support_ticket", "audit_config",
         "summarise_incident", "migrate_record", "verify_refund")

MODELS = ("gemini-2.5-flash-lite", "gemini-2.5-flash", "claude-haiku-4-5")

TRANSIENT_ERRORS = (
    "upstream 503 from payments processor",
    "connection reset by peer while calling ledger",
    "rate limit exceeded on search backend",
)

CONTEXT_LIMIT = 32_000


@dataclass(slots=True)
class RunLabel:
    """What the generator planted, kept out of the event stream on purpose."""

    run_id: str
    mode: str
    pathologies: list[str] = field(default_factory=list)
    cohort: str = "baseline"
    task: str = ""
    model: str = ""
    incidental: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "mode": self.mode, "pathologies": sorted(self.pathologies),
                "cohort": self.cohort, "task": self.task, "model": self.model,
                "incidental": sorted(self.incidental)}


class _Writer:
    """Accumulates events for one run, keeping seq and the clock consistent."""

    def __init__(self, run_id: str, t0: float) -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.ts = t0
        self.seq = 0

    def add(self, type_: str, payload: dict[str, Any], *, dt: float = 0.0) -> None:
        self.ts += dt
        self.events.append({"run_id": self.run_id, "seq": self.seq, "ts": round(self.ts, 6),
                            "type": type_, "payload": payload})
        self.seq += 1


def _tool_args(tool: str, rng: random.Random, n: int) -> dict[str, Any]:
    if tool == "search_docs":
        return {"query": rng.choice(("payment failure", "refund policy", "sla breach",
                                     "incident 88")), "top_k": rng.choice((3, 5, 10))}
    if tool in ("read_file", "write_file"):
        args: dict[str, Any] = {"path": f"workspace/{rng.choice(('notes', 'ledger', 'cfg'))}"
                                        f"/{n}.txt"}
        if tool == "write_file":
            args["content"] = "x" * rng.randint(20, 400)
        return args
    if tool == "list_files":
        return {}
    if tool == "run_query":
        return {"sql": f"select * from invoices where id = {rng.randint(1, 900)}",
                "limit": rng.choice((10, 50))}
    if tool == "fetch_url":
        return {"url": f"https://internal.example/api/v1/item/{rng.randint(1, 500)}"}
    if tool in ("charge", "refund"):
        return {"account": "operating_account", "amount": float(rng.randint(50, 4000))}
    if tool == "send_email":
        return {"to": "ops@example.com", "subject": "status", "body": "all clear"}
    if tool == "create_ticket":
        return {"title": f"follow up {n}", "priority": rng.choice((1, 2, 3))}
    if tool == "lookup_customer":
        return {"customer_id": f"cust-{rng.randint(1000, 9999)}"}
    if tool == "update_record":
        return {"record_id": f"rec-{rng.randint(100, 999)}", "fields": {"state": "reviewed"}}
    if tool == "submit_report":
        return {"summary": "reviewed and routed", "score": round(rng.random(), 3)}
    if tool == "escalate":
        return {"reason": "needs human judgement"}
    return {}


def _advertised(tools: list[str]) -> list[dict[str, Any]]:
    return [{"name": t, "description": f"{t} tool",
             "parameters": {"type": "object", **TOOL_SCHEMAS[t]}} for t in tools]


class Generator:
    """Deterministic generator for one cohort of runs."""

    def __init__(self, *, seed: int = 0, cohort: str = "baseline",
                 weights: dict[str, float] | None = None, t0: float = 1_780_000_000.0) -> None:
        self.seed = seed
        self.cohort = cohort
        self.weights = dict(weights or BASELINE_WEIGHTS)
        self.t0 = t0

    def _pick_mode(self, rng: random.Random) -> str:
        names = list(self.weights)
        return rng.choices(names, weights=[self.weights[n] for n in names], k=1)[0]

    def generate(self, n_runs: int) -> tuple[list[list[dict[str, Any]]], list[RunLabel]]:
        streams: list[list[dict[str, Any]]] = []
        labels: list[RunLabel] = []
        for i in range(n_runs):
            rng = random.Random(f"{self.seed}:{self.cohort}:{i}")
            mode = self._pick_mode(rng)
            run_id = f"{self.cohort}-{i:05d}"
            events, label = self._one(run_id, mode, rng)
            streams.append(events)
            labels.append(label)
        return streams, labels

    def _one(self, run_id: str, mode: str, rng: random.Random
             ) -> tuple[list[dict[str, Any]], RunLabel]:
        task = rng.choice(TASKS)
        model = rng.choice(MODELS)
        n_tools = rng.randint(4, 8)
        tools = rng.sample(sorted(TOOL_SCHEMAS), n_tools)
        terminal = "submit_report"
        if terminal not in tools:
            tools[-1] = terminal
        n_steps = {
            "healthy": rng.randint(4, 12),
            "retry_storm": rng.randint(5, 11),
            "oscillation": rng.randint(6, 14),
            "silent_truncation": rng.randint(4, 10),
            "premature_stop": rng.randint(6, 14),
            "arg_hallucination": rng.randint(4, 10),
            "context_exhaustion": rng.randint(10, 20),
            "tool_outage": rng.randint(5, 12),
            "gate_abandoned": rng.randint(5, 12),
            "slow_tool": rng.randint(4, 11),
        }[mode]

        w = _Writer(run_id, self.t0 + rng.random() * 86_400 * 30)
        plan = [{"step": s, "type": "llm" if s % 2 == 0 else "tool",
                 "tools": _advertised(tools), "terminal_tool": terminal}
                for s in range(n_steps)]
        w.add("run_start", {"step": 0, "task_id": task, "model": model, "plan": plan,
                            "cohort": self.cohort, "context_limit": CONTEXT_LIMIT,
                            "tools": _advertised(tools)})

        label = RunLabel(run_id=run_id, mode=mode, cohort=self.cohort, task=task, model=model)
        builder = getattr(self, f"_mode_{mode}")
        builder(w, rng, tools, terminal, n_steps, model, label)
        self._maybe_incidental(w, rng, tools, model, label)
        return w.events, label

    # --- shared building blocks -------------------------------------------------------------

    def _llm_turn(self, w: _Writer, rng: random.Random, step: int, model: str, *,
                  turn: int = 0, prompt_tokens: int | None = None,
                  finish: str = "STOP", called_tool: str | None = None,
                  text: str | None = None) -> None:
        w.add("llm_call", {"step": step, "model": model, "turn": turn}, dt=rng.uniform(0.01, 0.08))
        tok_in = prompt_tokens if prompt_tokens is not None else 180 + 110 * turn + rng.randint(
            0, 90)
        payload: dict[str, Any] = {
            "step": step, "model": model,
            "usage": {"prompt_tokens": tok_in, "candidates_tokens": rng.randint(18, 160)},
            "finish_reason": finish,
        }
        if called_tool:
            payload["function_call"] = {"name": called_tool}
        if text is not None:
            payload["text"] = text
        w.add("llm_result", payload, dt=rng.uniform(0.3, 2.4))

    def _tool_turn(self, w: _Writer, rng: random.Random, step: int, tool: str, *,
                   args: dict[str, Any] | None = None, ok: bool = True,
                   error: str | None = None, latency: float | None = None,
                   result_text: str | None = None) -> None:
        args = args if args is not None else _tool_args(tool, rng, step)
        w.add("tool_call", {"step": step, "tool": tool, "args": args,
                            "idempotency_key": f"{w.run_id}:{step}"}, dt=rng.uniform(0.0, 0.03))
        dt = latency if latency is not None else rng.uniform(0.02, 0.9)
        result: dict[str, Any] = {"status": "ok"} if ok else {
            "status": "error", "message": error or "tool failed"}
        if result_text is not None:
            result["text"] = result_text
        w.add("tool_result", {"step": step, "tool": tool, "result": result,
                              "idempotency_key": f"{w.run_id}:{step}"}, dt=dt)

    def _decision(self, w: _Writer, rng: random.Random, step: int, var: str) -> None:
        w.add("decision", {"step": step, "var": var,
                           "value": rng.choice(("auto", "manual", "retry"))},
              dt=rng.uniform(0.0, 0.02))

    def _body(self, w: _Writer, rng: random.Random, tools: list[str], model: str,
              steps: range, *, prompt_base: int | None = None) -> int:
        """Ordinary alternating work, returning the next free step index."""
        turn = 0
        step = steps.start
        for step in steps:
            if step % 3 == 2:
                self._decision(w, rng, step, f"route_{step}")
                continue
            if step % 2 == 0:
                tokens = None if prompt_base is None else prompt_base + 240 * turn
                self._llm_turn(w, rng, step, model, turn=turn, prompt_tokens=tokens)
                turn += 1
            else:
                self._tool_turn(w, rng, step, rng.choice(tools))
        return step + 1

    def _finish_ok(self, w: _Writer, rng: random.Random, step: int, terminal: str, model: str,
                   status: str = "completed") -> None:
        self._llm_turn(w, rng, step, model, turn=9, called_tool=terminal)
        self._tool_turn(w, rng, step, terminal)
        w.add("run_end", {"status": status}, dt=rng.uniform(0.0, 0.05))

    # --- one builder per planted mode -------------------------------------------------------

    def _mode_healthy(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, n_steps))
        self._finish_ok(w, rng, step, terminal, model)

    def _mode_retry_storm(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps // 2)))
        stuck = rng.choice([t for t in tools if t != terminal])
        args = _tool_args(stuck, rng, step)
        message = rng.choice(TRANSIENT_ERRORS)
        for _ in range(rng.randint(3, 7)):
            self._llm_turn(w, rng, step, model, turn=rng.randint(1, 6), called_tool=stuck)
            self._tool_turn(w, rng, step, stuck, args=dict(args), ok=False, error=message)
            step += 1
        w.add("error", {"step": step, "message": message}, dt=rng.uniform(0.0, 0.1))
        w.add("run_end", {"status": "failed"}, dt=0.01)
        label.pathologies.append("retry_loop")

    def _mode_oscillation(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps // 3)))
        pair = rng.sample([t for t in tools if t != terminal], 2)
        for i in range(rng.randint(5, 9)):
            tool = pair[i % 2]
            self._llm_turn(w, rng, step, model, turn=i, called_tool=tool)
            self._tool_turn(w, rng, step, tool)
            step += 1
        w.add("run_end", {"status": "exhausted"}, dt=0.01)
        label.pathologies.append("oscillation")
        # An oscillating run that gives up without calling its terminal tool really has terminated
        # prematurely as well. The labels record every pathology actually planted, not one per run,
        # because the detectors are not mutually exclusive and pretending otherwise would charge
        # premature_termination for false positives it did not commit.
        label.pathologies.append("premature_termination")

    def _mode_silent_truncation(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps - 2)))
        self._llm_turn(w, rng, step, model, turn=6, finish="MAX_TOKENS",
                       text="partial analysis of the ledger that stops mid sen")
        step += 1
        if rng.random() < 0.5:
            self._tool_turn(w, rng, step, rng.choice(tools),
                            result_text="row 1\nrow 2\n...[truncated 4180 bytes]")
            step += 1
        self._finish_ok(w, rng, step, terminal, model)
        label.pathologies.append("silent_truncation")

    def _mode_premature_stop(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        covered = max(1, int(n_steps * rng.uniform(0.15, 0.45)))
        self._body(w, rng, tools, model, range(0, covered))
        w.add("run_end", {"status": "completed"}, dt=rng.uniform(0.0, 0.05))
        label.pathologies.append("premature_termination")

    def _mode_arg_hallucination(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps // 2)))
        victim = rng.choice([t for t in tools if TOOL_SCHEMAS[t]["properties"]])
        args = _tool_args(victim, rng, step)
        kind = rng.choice(("unknown_arg", "wrong_type", "unknown_tool"))
        if kind == "unknown_arg":
            args[rng.choice(("filters", "options", "dry_run", "namespace"))] = "true"
            call_tool = victim
        elif kind == "wrong_type":
            key = sorted(TOOL_SCHEMAS[victim]["properties"])[0]
            declared = TOOL_SCHEMAS[victim]["properties"][key]["type"]
            args[key] = 12345 if declared == "string" else "not-a-number"
            call_tool = victim
        else:
            call_tool = rng.choice(("query_database", "get_invoice_status", "list_deployments"))
            args = {"id": rng.randint(1, 99)}
        attempts = rng.randint(1, 4)
        for _ in range(attempts):
            self._llm_turn(w, rng, step, model, turn=rng.randint(1, 5), called_tool=call_tool)
            self._tool_turn(w, rng, step, call_tool, args=dict(args), ok=False,
                            error=f"invalid arguments for {call_tool}")
            step += 1
        w.add("error", {"step": step, "message": f"invalid arguments for {call_tool}"}, dt=0.02)
        w.add("run_end", {"status": "failed"}, dt=0.01)
        label.pathologies.append("hallucinated_tool_args")
        if attempts >= 3:
            # Making the same rejected call three times is a retry loop by any definition.
            label.pathologies.append("retry_loop")

    def _mode_context_exhaustion(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        base = rng.randint(900, 1800)
        growth = rng.randint(1900, 3200)
        step = 0
        turn = 0
        prompt = base
        while prompt < CONTEXT_LIMIT * rng.uniform(0.86, 0.99) and turn < 24:
            self._llm_turn(w, rng, step, model, turn=turn, prompt_tokens=prompt)
            self._tool_turn(w, rng, step, rng.choice(tools))
            prompt += growth + rng.randint(0, 400)
            turn += 1
            step += 1
        w.add("error", {"step": step, "message":
                        f"context window exceeded: {prompt} tokens > {CONTEXT_LIMIT}"}, dt=0.05)
        w.add("run_end", {"status": "failed"}, dt=0.01)
        label.pathologies.append("context_exhaustion")

    def _mode_tool_outage(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps // 2)))
        down = rng.choice([t for t in tools if t != terminal])
        message = "upstream 503 from payments processor"
        self._llm_turn(w, rng, step, model, turn=3, called_tool=down)
        self._tool_turn(w, rng, step, down, ok=False, error=message)
        step += 1
        w.add("error", {"step": step, "message": message}, dt=0.02)
        w.add("compensation_start", {"count": 2}, dt=0.01)
        for i in range(2):
            w.add("compensation_step_done", {"step": step - i - 1, "action": "refund",
                                             "result": {"status": "ok"}}, dt=rng.uniform(0.01, 0.2))
        w.add("compensation_end", {}, dt=0.01)
        w.add("run_end", {"status": "compensated"}, dt=0.01)

    def _mode_gate_abandoned(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps - 1)))
        w.add("human_gate_wait", {"step": step, "gate_id": f"approval_{step}",
                                  "prompt": "approve payment?"}, dt=rng.uniform(0.0, 0.2))
        # No approval, no run_end: the run is still sitting there.

    def _mode_slow_tool(self, w, rng, tools, terminal, n_steps, model, label) -> None:
        step = self._body(w, rng, tools, model, range(0, max(2, n_steps - 1)))
        slow = rng.choice([t for t in tools if t != terminal])
        self._llm_turn(w, rng, step, model, turn=4, called_tool=slow)
        self._tool_turn(w, rng, step, slow, latency=rng.uniform(25.0, 95.0))
        step += 1
        self._finish_ok(w, rng, step, terminal, model)

    # --- incidental noise ------------------------------------------------------------------

    def _maybe_incidental(self, w: _Writer, rng: random.Random, tools: list[str], model: str,
                          label: RunLabel) -> None:
        """Occasionally add a second, truthfully-labelled pathology on top of the planted mode."""
        if rng.random() > 0.08:
            return
        kind = rng.choice(("retry", "truncation"))
        step = 900
        if kind == "retry":
            tool = rng.choice(tools)
            args = _tool_args(tool, rng, step)
            for _ in range(3):
                self._tool_turn(w, rng, step, tool, args=dict(args), ok=False,
                                error="transient backend error")
            if "retry_loop" not in label.pathologies:
                label.pathologies.append("retry_loop")
                label.incidental.append("retry_loop")
        else:
            self._llm_turn(w, rng, step, model, turn=7, finish="MAX_TOKENS",
                           text="cut off mid sen")
            if "silent_truncation" not in label.pathologies and label.mode not in (
                    "retry_storm", "arg_hallucination", "context_exhaustion", "tool_outage"):
                label.pathologies.append("silent_truncation")
                label.incidental.append("silent_truncation")


def shifted_weights(base: dict[str, float] | None = None, *, mode: str = "retry_storm",
                    multiplier: float = 2.5) -> dict[str, float]:
    """Baseline weights with one mode made more common, for the regression view's after cohort."""
    weights = dict(base or BASELINE_WEIGHTS)
    if mode not in weights:
        raise KeyError(mode)
    weights[mode] *= multiplier
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def write_corpus(out_dir: Path, *, n_runs: int = 2000, seed: int = 0, cohort: str = "baseline",
                 weights: dict[str, float] | None = None, shard_size: int = 250) -> dict[str, Any]:
    """Write a cohort to sharded JSONL plus a labels sidecar; returns a summary."""
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs" / cohort
    runs_dir.mkdir(parents=True, exist_ok=True)
    for stale in runs_dir.glob("*.jsonl"):
        stale.unlink()

    streams, labels = Generator(seed=seed, cohort=cohort, weights=weights).generate(n_runs)
    for shard_index in range(0, len(streams), shard_size):
        shard = streams[shard_index:shard_index + shard_size]
        path = runs_dir / f"shard-{shard_index // shard_size:04d}.jsonl"
        with path.open("w") as fh:
            for stream in shard:
                for event in stream:
                    fh.write(json.dumps(event, separators=(",", ":")) + "\n")

    labels_path = out_dir / "labels" / f"{cohort}.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps([lab.to_dict() for lab in labels], indent=1))

    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab.mode] = counts.get(lab.mode, 0) + 1
    return {
        "cohort": cohort,
        "n_runs": len(streams),
        "n_events": sum(len(s) for s in streams),
        "runs_dir": str(runs_dir),
        "labels_path": str(labels_path),
        "mode_counts": dict(sorted(counts.items())),
    }


def load_labels(path: Path) -> dict[str, RunLabel]:
    """Read a labels sidecar back into RunLabel objects keyed by run id."""
    out: dict[str, RunLabel] = {}
    for row in json.loads(Path(path).read_text()):
        out[row["run_id"]] = RunLabel(
            run_id=row["run_id"], mode=row["mode"], pathologies=list(row.get("pathologies") or []),
            cohort=row.get("cohort", ""), task=row.get("task", ""), model=row.get("model", ""),
            incidental=list(row.get("incidental") or []))
    return out
