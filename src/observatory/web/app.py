"""Local FastAPI app over an analysed store. Read-only, localhost, no build step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..attribution import PriceTable, attribute
from ..spans import Status, build_trace
from ..store import Store
from ..summarise import deterministic_summary

HERE = Path(__file__).parent
STATIC = HERE / "static"


def create_app(store: Store) -> FastAPI:
    app = FastAPI(title="agent-observatory", docs_url=None, redoc_url=None,
                  openapi_url=None)
    prices = PriceTable()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((HERE / "templates" / "index.html").read_text())

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        path = (STATIC / name).resolve()
        if path.parent != STATIC.resolve() or not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path)

    @app.get("/api/overview")
    def overview() -> JSONResponse:
        return JSONResponse({
            "summary": store.aggregate("overview", "summary") or {},
            "validation": store.aggregate("validation", "taxonomy"),
            "detectors": store.aggregate("validation", "detectors"),
            "detector_counts": store.aggregate("detector", "counts") or {},
            "attribution": store.aggregate("attribution", "population") or {},
            "sweep": store.aggregate("cluster", "sweep") or [],
            "stability_sweep": store.aggregate("cluster", "stability_sweep") or [],
            "merge_tree": store.aggregate("cluster", "merge_tree") or [],
            "regressions": store.aggregates("regression"),
            "cohorts": store.cohorts(),
        })

    @app.get("/api/clusters")
    def clusters() -> JSONResponse:
        rows = store.clusters()
        for row in rows:
            row["summary"] = deterministic_summary(row)
        return JSONResponse(rows)

    @app.get("/api/runs")
    def runs(cluster: str | None = None, cohort: str | None = None, status: str | None = None,
             pathology: str | None = None, q: str | None = None,
             order: str = "cost", limit: int = Query(100, ge=1, le=500),
             offset: int = Query(0, ge=0)) -> JSONResponse:
        return JSONResponse(store.list_runs(cluster=cluster, cohort=cohort, status=status,
                                            pathology=pathology, query=q, order=order,
                                            limit=limit, offset=offset))

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> JSONResponse:
        row = store.get_run(run_id)
        if row is None:
            raise HTTPException(404, "no such run")
        events = store.get_events(run_id)
        trace = build_trace(events)
        costs = attribute(trace, prices)
        root = trace.root
        spans: list[dict[str, Any]] = []
        for depth, span in trace.walk():
            sc = costs.spans[span.span_id]
            spans.append({
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "depth": depth,
                "name": span.name,
                "kind": span.kind.value.removeprefix("SPAN_KIND_").lower(),
                "offset_ms": (span.start_ns - root.start_ns) / 1e6,
                "duration_ms": span.duration_ms,
                "self_ms": sc.self_ms,
                "error": span.status is Status.ERROR,
                "status_message": span.status_message,
                "tokens_in": sc.tokens_in,
                "tokens_out": sc.tokens_out,
                "cost_usd": sc.cost_usd,
                "incl_cost_usd": sc.incl_cost_usd,
                "incl_tokens": sc.incl_tokens,
                "attributes": {k: v for k, v in span.attributes.items() if v is not None},
                "events": [{"name": e.name, "at_ms": (e.time_ns - root.start_ns) / 1e6,
                            "attributes": e.attributes} for e in span.events],
            })
        row["spans"] = spans
        row["wall_ms"] = root.duration_ms
        row["by_tool"] = costs.by_tool
        row["by_model"] = costs.by_model
        row["trace_id"] = trace.trace_id
        row["events"] = [e.to_dict() for e in events]
        return JSONResponse(row)

    return app
