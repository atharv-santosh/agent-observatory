"""The HTML report's self-containment, and the web app's API, over a real analysed store."""

from __future__ import annotations

import json
import re

import pytest

from observatory.cli import main
from observatory.pipeline import persist
from observatory.regression import compare_populations
from observatory.report import build_report, report_assets_are_local, write_report
from observatory.store import Store

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from observatory.web.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def store(tmp_path_factory, analysis, small_corpus):
    path = tmp_path_factory.mktemp("web") / "obs.db"
    s = Store(path)
    persist(analysis, s, trajectories={t[0].run_id: t for t in small_corpus[0]})
    report = compare_populations(
        analysis.clustering,
        set(analysis.run_ids[: len(analysis.run_ids) // 2]),
        set(analysis.run_ids[len(analysis.run_ids) // 2:]),
        all_facts=list(analysis.facts.values()), n_permutations=800, seed=0)
    from observatory.cli import _delta_dict
    s.put_aggregate("regression", "half->half", {
        "n_a": report.n_a, "n_b": report.n_b, "alpha": report.alpha,
        "label_a": report.label_a, "label_b": report.label_b,
        "clusters": [_delta_dict(d) for d in report.clusters],
        "pathologies": [_delta_dict(d) for d in report.pathologies],
    })
    s.put_aggregate("validation", "detectors", [{
        "pathology": "retry_loop", "fired": 10, "precision": 0.9, "recall": 1.0, "f1": 0.947,
        "precision_ci": [0.6, 0.99], "true_positives": 9, "false_positives": 1, "missed": 0,
        "examples_fp": [], "examples_fn": [],
    }])
    return s


@pytest.fixture(scope="module")
def client(store):
    return TestClient(create_app(store))


# --- the report -------------------------------------------------------------------------------

def test_report_is_self_contained(store):
    html = build_report(store)
    assert report_assets_are_local(html), "the report reaches outside itself"
    assert "<link" not in html.lower()
    assert "http://" not in html and "https://" not in html.replace(
        "https://opentelemetry.io", "")
    assert "@import" not in html
    assert "<style>" in html, "the stylesheet must be inlined"


def test_report_uses_the_same_stylesheet_as_the_web_app(store):
    from observatory.report import STATIC
    css = (STATIC / "styles.css").read_text()
    assert css in build_report(store)


def test_report_carries_the_visual_system_tokens(store):
    html = build_report(store)
    for token in ("#ffffff", "#16191c", "#6b7378", "#e2e5e7", "#14676f", "#a3322a", "#2b6446",
                  "#14171a", "#dfe3e6", "#8e979d", "#272c30", "#52aeb5", "#de7b71", "#66b48d"):
        assert token in html, f"missing palette token {token}"
    assert "prefers-color-scheme: dark" in html
    assert "font-size: 13px" in html
    assert "system-ui" in html
    assert "@font-face" not in html
    assert "box-shadow" not in html


def test_report_includes_every_section_it_has_data_for(store):
    html = build_report(store)
    for heading in ("Derived failure taxonomy", "Cluster map",
                    "Validation against planted ground truth", "Detector precision",
                    "Slowest tools", "Costliest steps", "Regression", "Termination"):
        assert heading in html, f"missing section {heading}"


def test_report_names_every_cluster_and_flags_incoherence(store):
    html = build_report(store)
    for row in store.clusters():
        assert row["slug"] in html
        quality = row["quality"] or {}
        if quality.get("incoherent"):
            assert "incoherent" in html


def test_report_escapes_content_it_did_not_write(tmp_path):
    store = Store(tmp_path / "x.db")
    store.initialise(reset=True)
    store.put_clusters([{
        "cid": 0, "slug": "c00", "name": "<script>alert(1)</script>", "size": 1,
        "cohesion": 1.0, "x": 0.0, "y": 0.0, "top_features": json.dumps([["a/b", 1.0]]),
        "representatives": json.dumps(["r"]), "quality": None,
    }])
    html = build_report(store)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_survives_an_empty_store(tmp_path):
    store = Store(tmp_path / "empty.db")
    store.initialise(reset=True)
    html = build_report(store)
    assert "agent-observatory" in html
    assert report_assets_are_local(html)


def test_write_report_creates_parent_directories(store, tmp_path):
    out = write_report(store, tmp_path / "deep" / "nested" / "report.html")
    assert out.exists() and out.stat().st_size > 5000


def test_cluster_map_is_inline_svg_with_one_circle_per_cluster(store):
    html = build_report(store)
    assert html.count("<circle") == len(store.clusters())
    assert "<svg" in html


# --- the web app ------------------------------------------------------------------------------

def test_index_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "agent-observatory" in page.text
    assert '<link rel="stylesheet" href="/static/styles.css">' in page.text
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_static_handler_refuses_a_path_escape(client):
    for bad in ("../app.py", "..%2Fapp.py", "nope.css"):
        assert client.get(f"/static/{bad}").status_code in (400, 404)


def test_the_page_loads_no_external_resource(client):
    page = client.get("/").text + client.get("/static/app.js").text
    page += client.get("/static/styles.css").text
    assert "http://" not in page and "https://" not in page
    assert "cdn" not in page.lower()
    assert "@font-face" not in page


def test_overview_endpoint_carries_what_the_views_need(client):
    body = client.get("/api/overview").json()
    assert body["summary"]["n_runs"] > 0
    assert body["validation"]["purity"] >= 0
    assert body["detectors"]
    assert body["attribution"]["by_tool"]
    assert body["sweep"] and body["stability_sweep"]
    assert body["cohorts"] == ["baseline"]
    assert "half->half" in body["regressions"]


def test_clusters_endpoint_includes_names_features_and_summaries(client, store):
    rows = client.get("/api/clusters").json()
    assert len(rows) == len(store.clusters())
    for row in rows:
        assert row["name"] and row["summary"]
        assert row["top_features"]
        assert row["representatives"]
        assert isinstance(row["x"], float) and isinstance(row["y"], float)


def test_runs_endpoint_filters_and_paginates(client, store):
    slug = store.clusters()[0]["slug"]
    rows = client.get("/api/runs", params={"cluster": slug, "limit": 5}).json()
    assert 0 < len(rows) <= 5
    assert all(r["cluster_slug"] == slug for r in rows)
    first = client.get("/api/runs", params={"order": "run_id", "limit": 3}).json()
    second = client.get("/api/runs", params={"order": "run_id", "limit": 3,
                                            "offset": 3}).json()
    assert {r["run_id"] for r in first}.isdisjoint({r["run_id"] for r in second})


def test_runs_endpoint_validates_its_limits(client):
    assert client.get("/api/runs", params={"limit": 0}).status_code == 422
    assert client.get("/api/runs", params={"limit": 9999}).status_code == 422
    assert client.get("/api/runs", params={"offset": -1}).status_code == 422


def test_run_detail_returns_a_walkable_span_tree(client, store):
    run_id = store.list_runs(order="events", limit=1)[0]["run_id"]
    body = client.get(f"/api/runs/{run_id}").json()
    spans = body["spans"]
    assert spans and spans[0]["depth"] == 0 and spans[0]["parent_id"] is None
    ids = {s["span_id"] for s in spans}
    for span in spans[1:]:
        assert span["parent_id"] in ids
        assert span["offset_ms"] >= 0
        assert span["duration_ms"] >= 0
        assert span["kind"] in ("internal", "client", "server")
    assert len(body["trace_id"]) == 32
    assert body["wall_ms"] >= max(s["duration_ms"] for s in spans) - 1e-6
    assert body["events"]


def test_run_detail_reports_inclusive_cost_up_the_tree(client, store):
    run_id = store.list_runs(order="cost", limit=1)[0]["run_id"]
    spans = client.get(f"/api/runs/{run_id}").json()["spans"]
    by_id = {s["span_id"]: s for s in spans}
    for span in spans:
        kids = [s for s in spans if s["parent_id"] == span["span_id"]]
        if kids:
            assert span["incl_cost_usd"] >= sum(k["incl_cost_usd"] for k in kids) - 1e-12
    assert by_id[spans[0]["span_id"]]["incl_cost_usd"] > 0


def test_unknown_run_is_a_404(client):
    assert client.get("/api/runs/not-a-run").status_code == 404


def test_api_docs_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_findings_reach_the_api_with_their_evidence(client, store):
    rows = store.list_runs(pathology="retry_loop", limit=1)
    body = client.get(f"/api/runs/{rows[0]['run_id']}").json()
    finding = next(f for f in body["findings"] if f["pathology"] == "retry_loop")
    assert finding["evidence"]["repeats"] >= 3


# --- the CLI ----------------------------------------------------------------------------------

def test_cli_end_to_end_gen_analyse_report_export(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    db = tmp_path / "obs.db"
    assert main(["gen", str(corpus), "-n", "150", "--seed", "3",
                 "--with-regression", "--shard-size", "75"]) == 0
    printed = capsys.readouterr().out
    assert "before" in printed and "after" in printed
    assert not (corpus / "runs" / "baseline").exists(), "no stale cohort"

    assert main(["--db", str(db), "analyse", str(corpus / "runs"),
                 "--labels", str(corpus / "labels" / "before.json"),
                 "--labels", str(corpus / "labels" / "after.json"),
                 "--k", "6"]) == 0
    printed = capsys.readouterr().out
    assert "purity" in printed and "detector" in printed
    assert Store(db).count_runs() == 300

    assert main(["--db", str(db), "clusters", "-v"]) == 0
    assert main(["--db", str(db), "runs", "--limit", "5"]) == 0
    assert main(["--db", str(db), "cost"]) == 0
    capsys.readouterr()

    run_id = Store(db).list_runs(limit=1)[0]["run_id"]
    assert main(["--db", str(db), "show", run_id]) == 0
    assert "agent.run" in capsys.readouterr().out

    out_html = tmp_path / "report.html"
    assert main(["--db", str(db), "report", "--out", str(out_html)]) == 0
    assert report_assets_are_local(out_html.read_text())

    out_otlp = tmp_path / "traces.json"
    assert main(["--db", str(db), "export", "--out", str(out_otlp), "--limit", "20"]) == 0
    body = json.loads(out_otlp.read_text())
    assert body["resourceSpans"][0]["scopeSpans"][0]["spans"]

    validation = tmp_path / "validation.json"
    assert main(["validate", str(corpus / "runs"),
                 "--labels", str(corpus / "labels" / "before.json"),
                 "--labels", str(corpus / "labels" / "after.json"),
                 "--k", "6", "--stability-reps", "2", "--json", str(validation)]) == 0
    numbers = json.loads(validation.read_text())
    assert 0.0 <= numbers["purity"] <= 1.0
    assert len(numbers["detectors"]) == 6

    assert main(["--db", str(db), "regress", str(corpus / "runs"), "--k", "6",
                 "--permutations", "500"]) == 0
    assert "BH alpha" in capsys.readouterr().out
    assert Store(db).aggregate("regression", "before->after") is not None

    assert main(["--db", str(db), "summarise", "c00", "--offline"]) == 0
    assert "distinguishing features" in capsys.readouterr().out


def test_cli_rejects_a_missing_corpus(tmp_path):
    with pytest.raises(SystemExit, match="no trajectories"):
        main(["--db", str(tmp_path / "x.db"), "analyse", str(tmp_path / "nothing")])


def test_cli_report_on_an_empty_store(tmp_path):
    db = tmp_path / "empty.db"
    Store(db).initialise(reset=True)
    with pytest.raises(SystemExit, match="no runs"):
        main(["--db", str(db), "report"])


def test_cli_show_rejects_an_unknown_run(tmp_path, store):
    with pytest.raises(SystemExit, match="no such run"):
        main(["--db", str(store.path), "show", "nope"])


def test_cli_summarise_rejects_an_unknown_cluster(store):
    with pytest.raises(SystemExit, match="no cluster"):
        main(["--db", str(store.path), "summarise", "c99", "--offline"])


def test_run_id_with_awkward_characters_is_routable(client, store):
    """Sandbox run ids contain colons, so the path has to survive encoding."""
    import urllib.parse
    run_id = store.list_runs(limit=1)[0]["run_id"]
    quoted = urllib.parse.quote(run_id, safe="")
    assert client.get(f"/api/runs/{quoted}").status_code == 200
    assert re.match(r"^[\w:.\-]+$", run_id)
