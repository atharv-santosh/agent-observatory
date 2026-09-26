"""The whole pipeline in one place: ingest, spans, cost, features, clusters, detectors, persist."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import discover
from .attribution import PriceTable, RunAttribution, aggregate, attribute
from .cluster import Clustering, cluster, project_2d
from .derive import RunFacts, derive
from .detectors import Finding, run_detectors
from .events import Event
from .features import RawFeatures, Vectorizer, featurize
from .spans import Trace, build_trace
from .store import Store, compress_events
from .validate import TaxonomyReport, evaluate_clustering, stability


@dataclass(slots=True)
class Analysis:
    """Everything derived from one corpus, in memory."""

    run_ids: list[str]
    facts: dict[str, RunFacts]
    traces: dict[str, Trace]
    attributions: dict[str, RunAttribution]
    findings: dict[str, list[Finding]]
    raw_features: list[RawFeatures]
    vectorizer: Vectorizer
    vectors: np.ndarray
    clustering: Clustering
    cohorts: dict[str, str]
    sources: dict[str, str]
    taxonomy: TaxonomyReport | None = None
    cluster_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    @property
    def total_cost_usd(self) -> float:
        return sum(a.total_cost_usd for a in self.attributions.values())

    @property
    def total_tokens(self) -> int:
        return sum(a.total_tokens for a in self.attributions.values())

    def cohort_members(self, cohort: str) -> set[str]:
        return {rid for rid, c in self.cohorts.items() if c == cohort}

    def overview(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for facts in self.facts.values():
            statuses[facts.status] = statuses.get(facts.status, 0) + 1
        return {
            "n_runs": len(self.run_ids),
            "n_events": sum(f.n_events for f in self.facts.values()),
            "n_spans": sum(len(t.spans) for t in self.traces.values()),
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "k": self.clustering.k,
            "silhouette": self.clustering.silhouette,
            "noise_fraction": self.clustering.noise_fraction,
            "statuses": dict(sorted(statuses.items(), key=lambda kv: -kv[1])),
            "cohorts": sorted(set(self.cohorts.values())),
            "n_findings": sum(len(v) for v in self.findings.values()),
            "feature_columns": len(self.vectorizer.columns),
        }


def load_trajectories(paths: Iterable[Path]) -> list[list[Event]]:
    """Ingest every trajectory under the given files or directories."""
    out: list[list[Event]] = []
    for path in paths:
        for events in discover(Path(path)):
            if events:
                out.append(events)
    return out


def _cohort_of(facts: RunFacts, source: str) -> str:
    value = facts.labels.get("cohort")
    if isinstance(value, str) and value:
        return value
    if "-" in facts.run_id:
        head = facts.run_id.rsplit("-", 1)[0]
        if head and not head.isdigit():
            return head
    return Path(source).parent.name or "default"


def analyse(trajectories: list[list[Event]], *, sources: dict[str, str] | None = None,
            prices: PriceTable | None = None, k: int | None = None,
            k_range: tuple[int, int] = (4, 18), seed: int = 0, min_df: int = 3,
            n_init: int = 8, noise_z: float = 3.5,
            truth: dict[str, str] | None = None, stability_reps: int = 0) -> Analysis:
    """Run the full pipeline over a list of event streams."""
    if not trajectories:
        raise ValueError("no trajectories to analyse")
    prices = prices or PriceTable()
    sources = sources or {}

    facts: dict[str, RunFacts] = {}
    traces: dict[str, Trace] = {}
    attributions: dict[str, RunAttribution] = {}
    findings: dict[str, list[Finding]] = {}
    raw: list[RawFeatures] = []
    run_ids: list[str] = []
    cohorts: dict[str, str] = {}

    for events in trajectories:
        f = derive(events)
        if f.run_id in facts:
            continue
        facts[f.run_id] = f
        traces[f.run_id] = build_trace(events)
        attributions[f.run_id] = attribute(traces[f.run_id], prices)
        findings[f.run_id] = run_detectors(f)
        raw.append(featurize(f))
        run_ids.append(f.run_id)
        cohorts[f.run_id] = _cohort_of(f, sources.get(f.run_id, ""))

    vectorizer = Vectorizer(min_df=min_df)
    vectors = vectorizer.fit_transform(raw)
    clustering = cluster(vectors, run_ids, vectorizer.column_names(), k=k, k_range=k_range,
                         n_init=n_init, seed=seed, noise_z=noise_z)

    analysis = Analysis(
        run_ids=run_ids, facts=facts, traces=traces, attributions=attributions,
        findings=findings, raw_features=raw, vectorizer=vectorizer, vectors=vectors,
        clustering=clustering, cohorts=cohorts, sources=sources,
    )
    if clustering.clusters:
        analysis.cluster_coords = project_2d(
            np.vstack([c.centroid for c in clustering.clusters]), seed=seed)
    if truth:
        analysis.taxonomy = evaluate_clustering(clustering, truth)
        if stability_reps:
            analysis.taxonomy.stability_ari = stability(
                vectors, run_ids, vectorizer.column_names(), k=clustering.k,
                reps=stability_reps, seed=seed)
            analysis.taxonomy.stability_reps = stability_reps
    return analysis


def persist(analysis: Analysis, store: Store, *, trajectories: dict[str, list[Event]],
            reset: bool = True) -> None:
    """Write an analysis into a SQLite store the web app and the report can read."""
    store.initialise(reset=reset)
    cluster_of = analysis.clustering.cluster_of()

    rows = []
    for run_id in analysis.run_ids:
        facts = analysis.facts[run_id]
        attribution = analysis.attributions[run_id]
        c = cluster_of.get(run_id)
        my_findings = analysis.findings[run_id]
        rows.append({
            "run_id": run_id,
            "source": analysis.sources.get(run_id),
            "cohort": analysis.cohorts.get(run_id),
            "status": facts.status,
            "n_events": facts.n_events,
            "duration_s": facts.duration_s,
            "n_tool_calls": len(facts.tool_calls),
            "n_llm_calls": len(facts.llm_calls),
            "n_errors": len(facts.errors),
            "tokens_in": facts.tokens_in,
            "tokens_out": facts.tokens_out,
            "cost_usd": attribution.total_cost_usd,
            "wall_ms": attribution.wall_ms,
            "cluster_cid": c.cid if c else None,
            "cluster_slug": c.slug if c else None,
            "cluster_name": c.name if c else None,
            "pathologies": json.dumps(sorted(f.pathology for f in my_findings)),
            "findings": json.dumps([
                {"pathology": f.pathology, "strength": round(f.strength, 3),
                 "evidence": f.evidence} for f in my_findings]),
            "hotspots": json.dumps([
                {"name": s.name, "depth": s.depth, "incl_cost_usd": round(s.incl_cost_usd, 8),
                 "incl_tokens": s.incl_tokens, "total_ms": round(s.total_ms, 3)}
                for s in attribution.hottest(6)]),
            "events_gz": compress_events(trajectories[run_id]),
        })
    store.put_runs(rows)

    quality_by_slug = {q.slug: q for q in (analysis.taxonomy.clusters if analysis.taxonomy else [])}
    coords = analysis.cluster_coords
    store.put_clusters([{
        "cid": c.cid, "slug": c.slug, "name": c.name, "size": c.size,
        "cohesion": round(c.cohesion, 6),
        "x": float(coords[i, 0]) if i < len(coords) else 0.0,
        "y": float(coords[i, 1]) if i < len(coords) else 0.0,
        "top_features": json.dumps([[n, round(w, 6)] for n, w in c.top_features]),
        "representatives": json.dumps(c.representatives),
        "quality": json.dumps({
            "dominant_label": q.dominant_label, "dominant_share": round(q.dominant_share, 4),
            "entropy": round(q.entropy, 4), "label_mix": q.label_mix,
            "incoherent": q.incoherent,
        }) if (q := quality_by_slug.get(c.slug)) else None,
    } for i, c in enumerate(analysis.clustering.clusters)])

    store.put_aggregate("overview", "summary", analysis.overview())
    store.put_aggregate("attribution", "population",
                        aggregate(list(analysis.attributions.values())))
    store.put_aggregate("cluster", "sweep", analysis.clustering.sweep)
    store.put_aggregate("cluster", "stability_sweep", analysis.clustering.stability_sweep)
    store.put_aggregate("cluster", "merge_tree", analysis.clustering.merge_tree)
    store.put_aggregate("cluster", "noise", analysis.clustering.noise[:500])
    pathology_counts: dict[str, int] = {}
    for fs in analysis.findings.values():
        for f in fs:
            pathology_counts[f.pathology] = pathology_counts.get(f.pathology, 0) + 1
    store.put_aggregate("detector", "counts", pathology_counts)
    if analysis.taxonomy is not None:
        t = analysis.taxonomy
        store.put_aggregate("validation", "taxonomy", {
            "n_runs": t.n_runs, "n_labelled": t.n_labelled, "k": t.k, "purity": t.purity,
            "inverse_purity": t.inverse_purity, "adjusted_rand": t.adjusted_rand, "nmi": t.nmi,
            "silhouette": t.silhouette, "noise_fraction": t.noise_fraction,
            "mode_coverage": t.mode_coverage, "mode_split": t.mode_split,
            "incoherent": t.incoherent, "stability_ari": t.stability_ari,
            "stability_reps": t.stability_reps,
        })
    store.set_meta(created_by="agent-observatory", version="0.1.0")
