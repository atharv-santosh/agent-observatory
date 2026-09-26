"""The store, the end-to-end pipeline, the corpus generator's own guarantees, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observatory.attribution import PriceTable
from observatory.corpus import BASELINE_WEIGHTS, MODES, Generator, load_labels, write_corpus
from observatory.events import normalise
from observatory.pipeline import analyse, load_trajectories, persist
from observatory.spans import build_trace
from observatory.store import Store, compress_events
from observatory.summarise import build_evidence, deterministic_summary, summarise_cluster

from .conftest import FIXTURES

# --- the corpus generator --------------------------------------------------------------------

def test_generator_is_deterministic_for_a_seed():
    a_streams, a_labels = Generator(seed=5, cohort="x").generate(40)
    b_streams, b_labels = Generator(seed=5, cohort="x").generate(40)
    assert a_streams == b_streams
    assert [label.to_dict() for label in a_labels] == [label.to_dict() for label in b_labels]


def test_different_seeds_give_different_corpora():
    a, _ = Generator(seed=5, cohort="x").generate(30)
    b, _ = Generator(seed=6, cohort="x").generate(30)
    assert a != b


def test_every_mode_is_reachable_and_every_stream_is_well_formed():
    streams, labels = Generator(seed=9, cohort="c").generate(500)
    assert {label.mode for label in labels} == set(MODES)
    assert set(BASELINE_WEIGHTS) == set(MODES)
    for stream, label in zip(streams, labels, strict=True):
        assert stream[0]["type"] == "run_start"
        assert [e["seq"] for e in stream] == list(range(len(stream)))
        timestamps = [e["ts"] for e in stream]
        assert timestamps == sorted(timestamps), "the clock must not go backwards"
        assert all(e["run_id"] == label.run_id for e in stream)


def test_ground_truth_is_not_written_into_the_event_stream():
    """The whole validation rests on this: no feature can read the label even by accident."""
    streams, labels = Generator(seed=11, cohort="c").generate(200)
    for stream, label in zip(streams, labels, strict=True):
        blob = json.dumps(stream)
        assert label.mode not in blob, f"{label.mode} leaked into {label.run_id}"
        for pathology in label.pathologies:
            assert pathology not in blob
        for key in ("ground_truth", "injected", "patholog", "failure_mode"):
            assert key not in blob
        # "mode" only as a JSON key; "model" is legitimately present.
        assert '"mode"' not in blob


def test_planted_pathologies_match_the_mode_they_belong_to():
    _, labels = Generator(seed=13, cohort="c").generate(400)
    expected = {
        "retry_storm": "retry_loop",
        "oscillation": "oscillation",
        "silent_truncation": "silent_truncation",
        "premature_stop": "premature_termination",
        "arg_hallucination": "hallucinated_tool_args",
        "context_exhaustion": "context_exhaustion",
    }
    for label in labels:
        if label.mode in expected:
            assert expected[label.mode] in label.pathologies
        if label.mode in ("healthy", "slow_tool", "tool_outage", "gate_abandoned"):
            # These plant no named pathology of their own; only an incidental one may appear.
            assert set(label.pathologies) <= set(label.incidental)


def test_modes_overlap_so_the_exercise_is_not_trivial():
    _, labels = Generator(seed=17, cohort="c").generate(600)
    assert any(label.incidental for label in labels), "no incidental label noise was planted"
    assert any(len(label.pathologies) > 1 for label in labels), "no run carries two pathologies"


def test_write_corpus_shards_and_writes_a_sidecar(tmp_path):
    summary = write_corpus(tmp_path, n_runs=120, seed=3, cohort="probe", shard_size=50)
    assert summary["n_runs"] == 120
    shards = sorted(Path(summary["runs_dir"]).glob("*.jsonl"))
    assert len(shards) == 3
    labels = load_labels(Path(summary["labels_path"]))
    assert len(labels) == 120
    assert sum(summary["mode_counts"].values()) == 120
    trajectories = load_trajectories([Path(summary["runs_dir"])])
    assert len(trajectories) == 120
    assert {t[0].run_id for t in trajectories} == set(labels)


def test_write_corpus_clears_stale_shards(tmp_path):
    write_corpus(tmp_path, n_runs=200, seed=3, cohort="probe", shard_size=50)
    summary = write_corpus(tmp_path, n_runs=50, seed=3, cohort="probe", shard_size=50)
    assert len(sorted(Path(summary["runs_dir"]).glob("*.jsonl"))) == 1


# --- the store -------------------------------------------------------------------------------

@pytest.fixture
def stored(tmp_path, analysis, small_corpus):
    store = Store(tmp_path / "obs.db")
    persist(analysis, store, trajectories={t[0].run_id: t for t in small_corpus[0]})
    return store


def test_store_round_trips_runs_clusters_and_aggregates(stored, analysis):
    assert stored.count_runs() == len(analysis.run_ids)
    assert len(stored.clusters()) == analysis.clustering.k
    assert stored.aggregate("overview", "summary")["n_runs"] == len(analysis.run_ids)
    assert stored.aggregate("attribution", "population")["by_tool"]
    assert stored.meta()["created_by"] == "agent-observatory"


def test_events_survive_compression_unchanged(stored, small_corpus):
    original = {t[0].run_id: t for t in small_corpus[0]}
    run_id = next(iter(original))
    restored = stored.get_events(run_id)
    assert [e.to_dict() for e in restored] == [e.to_dict() for e in original[run_id]]


def test_span_tree_can_be_rebuilt_from_the_store(stored, analysis):
    run_id = analysis.run_ids[0]
    rebuilt = build_trace(stored.get_events(run_id))
    original = analysis.traces[run_id]
    assert rebuilt.trace_id == original.trace_id
    assert [s.span_id for s in rebuilt.spans] == [s.span_id for s in original.spans]


def test_filters_narrow_the_run_list(stored, analysis):
    slug = analysis.clustering.clusters[0].slug
    rows = stored.list_runs(cluster=slug, limit=500)
    assert rows and all(r["cluster_slug"] == slug for r in rows)
    assert stored.list_runs(cluster="nope") == []
    by_pathology = stored.list_runs(pathology="retry_loop", limit=500)
    assert by_pathology and all("retry_loop" in r["pathologies"] for r in by_pathology)
    one = stored.list_runs(query=analysis.run_ids[3], limit=5)
    assert one and one[0]["run_id"] == analysis.run_ids[3]


def test_ordering_options_all_work(stored):
    for order in ("cost", "duration", "events", "run_id", "errors"):
        rows = stored.list_runs(order=order, limit=10)
        assert len(rows) == 10
    costs = [r["cost_usd"] for r in stored.list_runs(order="cost", limit=20)]
    assert costs == sorted(costs, reverse=True)


def test_pagination_does_not_repeat_or_skip(stored):
    first = stored.list_runs(order="run_id", limit=10, offset=0)
    second = stored.list_runs(order="run_id", limit=10, offset=10)
    assert {r["run_id"] for r in first}.isdisjoint({r["run_id"] for r in second})


def test_get_run_includes_findings_and_hotspots(stored, analysis):
    run_id = max(analysis.run_ids, key=lambda r: len(analysis.findings[r]))
    row = stored.get_run(run_id)
    assert row is not None
    assert len(row["findings"]) == len(analysis.findings[run_id])
    assert row["hotspots"]
    assert all("incl_cost_usd" in h for h in row["hotspots"])
    assert stored.get_run("no-such-run") is None
    assert stored.get_events("no-such-run") == []


def test_reset_truly_resets(tmp_path, analysis, small_corpus):
    store = Store(tmp_path / "r.db")
    trajectories = {t[0].run_id: t for t in small_corpus[0]}
    persist(analysis, store, trajectories=trajectories)
    before = store.count_runs()
    store.initialise(reset=True)
    assert store.count_runs() == 0
    persist(analysis, store, trajectories=trajectories)
    assert store.count_runs() == before


def test_cohorts_are_recorded(stored):
    assert stored.cohorts() == ["baseline"]


def test_compress_events_is_smaller_than_the_raw_json(small_corpus):
    events = small_corpus[0][0]
    raw = "\n".join(json.dumps(e.to_dict()) for e in events).encode()
    assert len(compress_events(events)) < len(raw)


# --- the pipeline ----------------------------------------------------------------------------

def test_analyse_produces_one_entry_per_run(analysis):
    n = len(analysis.run_ids)
    assert len(analysis.facts) == n
    assert len(analysis.traces) == n
    assert len(analysis.attributions) == n
    assert len(analysis.findings) == n
    assert analysis.vectors.shape[0] == n
    assert set(analysis.cohorts) == set(analysis.run_ids)


def test_overview_numbers_add_up(analysis):
    overview = analysis.overview()
    assert overview["n_runs"] == len(analysis.run_ids)
    assert overview["n_events"] == sum(f.n_events for f in analysis.facts.values())
    assert overview["n_spans"] == sum(len(t.spans) for t in analysis.traces.values())
    assert overview["total_cost_usd"] > 0
    assert overview["total_tokens"] > 0
    assert sum(overview["statuses"].values()) == overview["n_runs"]


def test_analyse_is_deterministic():
    streams, _ = Generator(seed=23, cohort="d").generate(150)
    trajectories = [[normalise(e) for e in s] for s in streams]
    a = analyse(trajectories, k=5, seed=0)
    b = analyse(trajectories, k=5, seed=0)
    assert [c.name for c in a.clustering.clusters] == [c.name for c in b.clustering.clusters]
    assert a.clustering.label_of() == b.clustering.label_of()


def test_analyse_deduplicates_a_run_seen_twice():
    streams, _ = Generator(seed=29, cohort="e").generate(60)
    trajectories = [[normalise(e) for e in s] for s in streams]
    analysis = analyse(trajectories + trajectories, k=5, seed=0)
    assert len(analysis.run_ids) == 60


def test_analyse_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="no trajectories"):
        analyse([])


def test_analyse_runs_over_the_real_sibling_fixtures():
    """The mixed real corpus has no labels, so nothing is claimed but that the pipeline runs."""
    trajectories = load_trajectories([FIXTURES])
    assert len(trajectories) > 300
    analysis = analyse(trajectories, k_range=(4, 9), seed=0, min_df=5)
    assert analysis.clustering.clusters
    assert analysis.taxonomy is None
    assert analysis.overview()["n_spans"] > len(trajectories)
    assert {"agent-sandbox", None} & {f.source for f in analysis.facts.values()}


def test_custom_price_table_changes_the_totals():
    streams, _ = Generator(seed=31, cohort="f").generate(40)
    trajectories = [[normalise(e) for e in s] for s in streams]
    cheap = analyse(trajectories, k=4, seed=0, prices=PriceTable({"_default": (0.0, 0.0)}))
    dear = analyse(trajectories, k=4, seed=0, prices=PriceTable({"_default": (10.0, 10.0)}))
    assert cheap.total_cost_usd == 0.0
    assert dear.total_cost_usd > 0.0


def test_cohorts_are_read_from_the_run_start_payload():
    before, _ = Generator(seed=37, cohort="before").generate(30)
    after, _ = Generator(seed=38, cohort="after").generate(30)
    trajectories = [[normalise(e) for e in s] for s in before + after]
    analysis = analyse(trajectories, k=4, seed=0)
    assert len(analysis.cohort_members("before")) == 30
    assert len(analysis.cohort_members("after")) == 30


def test_taxonomy_and_stability_are_computed_when_labels_are_given():
    streams, labels = Generator(seed=41, cohort="g").generate(240)
    trajectories = [[normalise(e) for e in s] for s in streams]
    analysis = analyse(trajectories, k=6, seed=0,
                       truth={label.run_id: label.mode for label in labels},
                       stability_reps=3)
    assert analysis.taxonomy is not None
    assert analysis.taxonomy.stability_ari is not None
    assert 0.0 <= analysis.taxonomy.stability_ari <= 1.0
    assert "purity" in analysis.taxonomy.summary()


def test_persisted_quality_marks_incoherent_clusters(stored):
    for row in stored.clusters():
        quality = row["quality"]
        if quality is not None:
            assert quality["incoherent"] == (quality["dominant_share"] < 0.5)
            assert sum(quality["label_mix"].values()) == row["size"]


# --- optional summarisation ------------------------------------------------------------------

def test_cluster_summary_works_with_no_model_and_no_credentials(stored):
    cluster_row = stored.clusters()[0]
    offline = summarise_cluster(stored, cluster_row, offline=True)
    assert cluster_row["slug"] in offline
    assert "distinguishing features" in offline
    assert deterministic_summary(cluster_row) in offline


def test_evidence_block_cites_real_runs(stored):
    cluster_row = stored.clusters()[0]
    evidence = build_evidence(stored, cluster_row)
    assert cluster_row["representatives"][0] in evidence
    assert "representative runs" in evidence


def test_summarise_without_credentials_falls_back_rather_than_failing(stored, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    text = summarise_cluster(stored, stored.clusters()[0], offline=False)
    assert "No model configured" in text
