"""Optional prose narration of a cluster, off by default and never on the tested path.

Everything else in this repo runs on recorded data with no credentials, and that property is worth
more than a paragraph of generated prose, so this module is arranged so the offline answer is the
default and the model call is the opt-in. The deterministic summary is assembled from the centroid
features and the representative runs; the model, if you have one configured, only rewrites it.
"""

from __future__ import annotations

import os
from typing import Any

from .store import Store

_PROMPT = """You are looking at a cluster of agent runs that were grouped automatically by
structural similarity, not by any human label. Below are the features that distinguish this cluster
from the rest of the population, and a description of some representative runs.

Write three sentences: what these runs have in common, the most likely single cause, and what an
engineer should look at first. Do not invent detail that is not in the evidence.

{evidence}
"""


def build_evidence(store: Store, cluster: dict[str, Any], *, n_examples: int = 3) -> str:
    """Assemble the evidence block: centroid features plus a sketch of representative runs."""
    lines = [f"cluster {cluster['slug']}: {cluster['name']}",
             f"size: {cluster['size']} runs, cohesion {cluster['cohesion']:.2f}",
             "",
             "distinguishing features (centroid lift over population mean):"]
    for name, weight in cluster["top_features"][:8]:
        lines.append(f"  {weight:+.3f}  {name}")
    lines.append("")
    lines.append("representative runs:")
    for run_id in cluster["representatives"][:n_examples]:
        row = store.get_run(run_id)
        if row is None:
            continue
        lines.append(f"  {run_id}: status={row['status']} events={row['n_events']} "
                     f"tools={row['n_tool_calls']} errors={row['n_errors']} "
                     f"cost=${row['cost_usd']:.5f}")
        for finding in row["findings"]:
            lines.append(f"    detector {finding['pathology']} "
                         f"(strength {finding['strength']:.2f})")
        events = store.get_events(run_id)
        tools = [e.payload.get("tool") for e in events if e.type == "tool_call"]
        if tools:
            lines.append(f"    tool sequence: {' -> '.join(str(t) for t in tools[:14])}")
        errors = [str(e.payload.get("message"))[:90] for e in events if e.type == "error"]
        for message in errors[:2]:
            lines.append(f"    error: {message}")
    return "\n".join(lines)


def deterministic_summary(cluster: dict[str, Any]) -> str:
    """A summary derived from the centroid alone, which is what the tests and the report use."""
    fragments = [n.split("/", 1)[-1] for n, _ in cluster["top_features"][:4]]
    quality = cluster.get("quality") or {}
    parts = [f"{cluster['size']} runs sharing {', '.join(fragments) or 'no distinguishing feature'}",
             f"named {cluster['name']!r} from its own centroid",
             f"internal cohesion {cluster['cohesion']:.2f}"]
    if quality:
        parts.append(f"dominant hand label {quality['dominant_label']} "
                     f"at {quality['dominant_share']:.0%}"
                     + (" (incoherent)" if quality.get("incoherent") else ""))
    return ". ".join(parts) + "."


def summarise_cluster(store: Store, cluster: dict[str, Any], *, model: str = "gemini-2.5-flash",
                      offline: bool = False) -> str:
    """Narrate a cluster. Falls back to the deterministic summary whenever a model is unavailable."""
    evidence = build_evidence(store, cluster)
    baseline = deterministic_summary(cluster)
    if offline:
        return f"{baseline}\n\n{evidence}"
    text = _try_model(_PROMPT.format(evidence=evidence), model)
    if text is None:
        return (f"{baseline}\n\nNo model configured (set GOOGLE_CLOUD_PROJECT and run "
                f"`gcloud auth application-default login`, or pass --offline).\n\n{evidence}")
    return f"{text.strip()}\n\n{baseline}"


def _try_model(prompt: str, model: str) -> str | None:
    """Return generated prose, or None if no model is reachable. Never raises."""
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return None
    try:
        from google import genai  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                              location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
        response = client.models.generate_content(model=model, contents=prompt)
        return getattr(response, "text", None)
    except Exception:  # noqa: BLE001 - an optional extra must never break the CLI
        return None
