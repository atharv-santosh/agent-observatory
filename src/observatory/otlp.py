"""OTLP/JSON export, written to a file by default and posted only if you ask for it.

Requiring a collector to be up before you can look at recorded data would be silly, so the default
sink is a file holding exactly the body an OTLP/HTTP endpoint would receive. If you do have a
collector, `--endpoint` posts the same body with urllib; there is no OTel SDK dependency.
"""

from __future__ import annotations

import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .spans import Span, Trace

SCHEMA_URL = "https://opentelemetry.io/schemas/1.27.0"


def _anyvalue(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_anyvalue(v) for v in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": [
            {"key": str(k), "value": _anyvalue(v)} for k, v in value.items()]}}
    return {"stringValue": "" if value is None else str(value)}


def _attrs(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _anyvalue(v)} for k, v in mapping.items() if v is not None]


def span_to_otlp(span: Span) -> dict[str, Any]:
    body: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": span.kind.value,
        "startTimeUnixNano": str(span.start_ns),
        "endTimeUnixNano": str(span.end_ns),
        "attributes": _attrs(span.attributes),
        "status": {"code": span.status.value},
    }
    if span.parent_id:
        body["parentSpanId"] = span.parent_id
    if span.status_message:
        body["status"]["message"] = span.status_message
    if span.events:
        body["events"] = [
            {"name": e.name, "timeUnixNano": str(e.time_ns), "attributes": _attrs(e.attributes)}
            for e in span.events
        ]
    return body


def traces_to_otlp(traces: list[Trace], *, service_name: str = "agent-observatory",
                   scope_version: str = "0.1.0") -> dict[str, Any]:
    """Build one ExportTraceServiceRequest body covering every trace given."""
    return {
        "resourceSpans": [{
            "resource": {"attributes": _attrs({
                "service.name": service_name,
                "telemetry.sdk.name": "agent-observatory",
                "telemetry.sdk.language": "python",
            })},
            "schemaUrl": SCHEMA_URL,
            "scopeSpans": [{
                "scope": {"name": "observatory.spans", "version": scope_version},
                "spans": [span_to_otlp(s) for t in traces for s in t.spans],
                "schemaUrl": SCHEMA_URL,
            }],
        }]
    }


def write_otlp(traces: list[Trace], path: Path, *, gzip_out: bool | None = None) -> int:
    """Write the OTLP body to disk; returns bytes written. gzip inferred from a .gz suffix."""
    body = json.dumps(traces_to_otlp(traces), separators=(",", ":")).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_out is None:
        gzip_out = path.suffix == ".gz"
    if gzip_out:
        path.write_bytes(gzip.compress(body))
    else:
        path.write_bytes(body)
    return path.stat().st_size


def post_otlp(traces: list[Trace], endpoint: str, *, timeout: float = 10.0,
              headers: dict[str, str] | None = None) -> tuple[int, str]:
    """POST to an OTLP/HTTP traces endpoint. Optional by design; nothing in the tests calls it."""
    body = json.dumps(traces_to_otlp(traces)).encode()
    request = urllib.request.Request(  # noqa: S310 - endpoint is operator supplied
        endpoint, data=body, method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", "replace")[:2000]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:2000]
    except OSError as exc:
        return 0, f"{type(exc).__name__}: {exc}"
