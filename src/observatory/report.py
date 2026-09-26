"""A single HTML file with everything in it: no CDN, no fonts, no fetches, no build step.

Shares its stylesheet with the web app by inlining the same file, so the two cannot drift apart. The
cluster map is inline SVG computed here rather than drawn by a charting library, which is the only
way to keep one file with no external requests.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .store import Store

STATIC = Path(__file__).parent / "web" / "static"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec or ".3f")
    if isinstance(value, int):
        return format(value, spec or ",")
    return _esc(value)


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f'<th class="{"n" if i in numeric else ""}">{_esc(h)}</th>'
                   for i, h in enumerate(headers))
    body = []
    for row in rows:
        cells = "".join(f'<td class="{"n" if i in numeric else ""}">{cell}</td>'
                        for i, cell in enumerate(row))
        body.append(f"<tr>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _stats(pairs: list[tuple[str, str]]) -> str:
    items = "".join(f'<div class="stat"><div class="k">{_esc(k)}</div>'
                    f'<div class="v">{v}</div></div>' for k, v in pairs)
    return f'<div class="stats">{items}</div>'


def _merge_summary(merge_tree: list[list[Any]]) -> str:
    return " | ".join("c{:02d}+c{:02d} at {:.2f}".format(int(a), int(b), float(d))
                      for a, b, d in merge_tree[:6])


def _cluster_map_svg(clusters: list[dict[str, Any]], *, width: int = 620,
                     height: int = 320) -> str:
    """Inline SVG scatter of the cluster centroids, positioned by classical MDS."""
    if not clusters:
        return "<p class='muted'>no clusters</p>"
    xs = [c["x"] for c in clusters]
    ys = [c["y"] for c in clusters]
    pad = 46
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    biggest = max(c["size"] for c in clusters) or 1
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'style="max-width:{width}px" role="img" aria-label="cluster map">']
    for c in clusters:
        px = pad + (c["x"] - min(xs)) / span_x * (width - 2 * pad)
        py = height - pad - (c["y"] - min(ys)) / span_y * (height - 2 * pad)
        r = 5 + 20 * (c["size"] / biggest) ** 0.5
        quality = c.get("quality") or {}
        fill = "var(--bad)" if quality.get("incoherent") else "var(--accent)"
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.1f}" fill="{fill}" '
                     f'opacity="0.75"><title>{_esc(c["slug"])}: {_esc(c["name"])} '
                     f'(n={c["size"]})</title></circle>')
        parts.append(f'<text x="{px:.1f}" y="{py + r + 11:.1f}" text-anchor="middle" '
                     f'font-size="11" fill="var(--muted)" '
                     f'font-family="var(--mono)">{_esc(c["slug"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _sparkline(points: list[tuple[int, float]], *, width: int = 240, height: int = 46,
               label: str = "") -> str:
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1.0
    coords = []
    for x, y in points:
        px = (x - min(xs)) / ((max(xs) - min(xs)) or 1) * (width - 8) + 4
        py = height - 6 - (y - lo) / span * (height - 14)
        coords.append(f"{px:.1f},{py:.1f}")
    peak = max(points, key=lambda p: p[1])
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{_esc(label)}">'
            f'<polyline points="{" ".join(coords)}" fill="none" stroke="var(--accent)" '
            f'stroke-width="1.2"/>'
            f'<text x="4" y="10" font-size="10" fill="var(--muted)">{_esc(label)} '
            f'peak k={peak[0]} at {peak[1]:.3f}</text></svg>')


def build_report(store: Store) -> str:
    """Render the whole report as one HTML string."""
    overview = store.aggregate("overview", "summary") or {}
    clusters = store.clusters()
    validation = store.aggregate("validation", "taxonomy")
    detectors = store.aggregate("validation", "detectors")
    detector_counts = store.aggregate("detector", "counts") or {}
    population = store.aggregate("attribution", "population") or {}
    sweep = store.aggregate("cluster", "sweep") or []
    merge_tree = store.aggregate("cluster", "merge_tree") or []
    regressions = store.aggregates("regression")
    css = (STATIC / "styles.css").read_text()
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    blocks: list[str] = []

    blocks.append("<section>" + _stats([
        ("runs", _fmt(overview.get("n_runs"))),
        ("events", _fmt(overview.get("n_events"))),
        ("spans", _fmt(overview.get("n_spans"))),
        ("clusters", _fmt(overview.get("k"))),
        ("unclustered", f"{(overview.get('noise_fraction') or 0) * 100:.1f}%"),
        ("cost", f"${overview.get('total_cost_usd', 0):.4f}"),
        ("tokens", _fmt(overview.get("total_tokens"))),
        ("detector findings", _fmt(overview.get("n_findings"))),
    ]) + "</section>")

    # --- taxonomy ---------------------------------------------------------------------------
    rows = []
    for c in clusters:
        quality = c.get("quality") or {}
        share = (f"{quality['dominant_share'] * 100:.0f}% {_esc(quality['dominant_label'])}"
                 if quality else "-")
        flag = ' <span class="pill bad">incoherent</span>' if quality.get("incoherent") else ""
        features = ", ".join(_esc(n.split("/", 1)[-1]) for n, _ in c["top_features"][:4])
        rows.append([f'<span class="mono">{_esc(c["slug"])}</span>',
                     f'{_esc(c["name"])}{flag}',
                     _fmt(c["size"]), f'{c["cohesion"]:.2f}', share,
                     f'<span class="muted">{features}</span>'])
    blocks.append("<section><h2>Derived failure taxonomy</h2>"
                  "<p>Every cluster below was found without labels, and named from its own "
                  "highest-lift centroid features.</p>"
                  + _table(["cluster", "name", "runs", "cohesion", "dominant label", "centroid"],
                           rows, numeric={2, 3})
                  + "</section>")

    blocks.append('<section><h2>Cluster map</h2><p>Centroids projected to two dimensions by '
                  'classical MDS on cosine distance. Area is cluster size.</p>'
                  + _cluster_map_svg(clusters)
                  + (f'<p class="muted">Nearest neighbours by average linkage: '
                     f'{_esc(_merge_summary(merge_tree))}</p>' if merge_tree else "")
                  + "</section>")

    # --- validation -------------------------------------------------------------------------
    if validation:
        pairs = [
            ("purity", f"{validation['purity']:.3f}"),
            ("inverse purity", f"{validation['inverse_purity']:.3f}"),
            ("adjusted Rand", f"{validation['adjusted_rand']:.3f}"),
            ("NMI", f"{validation['nmi']:.3f}"),
            ("silhouette", f"{validation['silhouette']:.3f}"),
            ("labelled runs", _fmt(validation["n_labelled"])),
        ]
        if validation.get("stability_ari") is not None:
            pairs.append(("stability ARI", f"{validation['stability_ari']:.3f}"))
        coverage = _table(["planted mode", "largest cluster holds", "clusters it spans"],
                          [[_esc(m), f"{v * 100:.1f}%",
                            str(validation["mode_split"].get(m, 0))]
                           for m, v in sorted(validation["mode_coverage"].items(),
                                              key=lambda kv: -kv[1])], numeric={1, 2})
        incoherent = validation.get("incoherent") or []
        note = (f'<p class="bad">Incoherent clusters (dominant label under 50%): '
                f'{_esc(", ".join(incoherent))}.</p>' if incoherent
                else '<p class="good">No cluster fell below 50% dominant label share.</p>')
        blocks.append("<section><h2>Validation against planted ground truth</h2>"
                      "<p>The corpus generator planted these modes and wrote them to a sidecar "
                      "file that never enters the event stream, so no feature can read them.</p>"
                      + _stats(pairs) + note
                      + '<div class="split"><div><h3>Mode recovery</h3>' + coverage + "</div>"
                      + (f'<div><h3>k selection</h3>'
                         f'{_sparkline([(int(k), float(v)) for k, v in sweep], label="silhouette")}'
                         f'</div>' if sweep else "<div></div>")
                      + "</div></section>")

    # --- detectors --------------------------------------------------------------------------
    if detectors:
        rows = []
        for d in detectors:
            lo, hi = d["precision_ci"]
            cls = "bad" if d["precision"] < 0.8 else "good"
            rows.append([_esc(d["pathology"]), _fmt(d["fired"]),
                         f'<span class="{cls}">{d["precision"]:.3f}</span>',
                         f"[{lo:.2f}, {hi:.2f}]", f"{d['recall']:.3f}", f"{d['f1']:.3f}",
                         _fmt(d["false_positives"]), _fmt(d["missed"])])
        blocks.append("<section><h2>Detector precision</h2>"
                      "<p>Measured against the same labelled corpus. A detector without a "
                      "precision number is an alert nobody should trust.</p>"
                      + _table(["pathology", "fired", "precision", "95% CI", "recall", "F1",
                                "false pos", "missed"], rows,
                               numeric={1, 2, 3, 4, 5, 6, 7}) + "</section>")
    elif detector_counts:
        blocks.append("<section><h2>Detector findings</h2>"
                      + _table(["pathology", "runs"],
                               [[_esc(k), _fmt(v)] for k, v in
                                sorted(detector_counts.items(), key=lambda kv: -kv[1])],
                               numeric={1})
                      + '<p class="muted">No labels supplied, so no precision is claimed.</p>'
                      + "</section>")

    # --- attribution ------------------------------------------------------------------------
    for group, title, sort_key in (("by_tool", "Slowest tools", "total_ms"),
                                   ("by_step", "Costliest steps", "cost_usd"),
                                   ("by_model", "Cost by model", "cost_usd")):
        buckets = population.get(group) or {}
        if not buckets:
            continue
        ranked = sorted(buckets.items(), key=lambda kv: -kv[1].get(sort_key, 0))[:14]
        peak = max((b.get(sort_key, 0) for _, b in ranked), default=1) or 1
        rows = []
        for key, b in ranked:
            width = 100 * b.get(sort_key, 0) / peak
            bad = " bad" if b.get("error_rate", 0) > 0.2 else ""
            rows.append([f'<span class="mono">{_esc(key)}</span>', _fmt(int(b["calls"])),
                         f"${b['cost_usd']:.5f}", _fmt(int(b["tokens_in"] + b["tokens_out"])),
                         f"{b['mean_ms']:.1f}", f"{b['max_ms']:.1f}",
                         f"{b['error_rate'] * 100:.1f}%",
                         f'<div class="bar{bad}" style="width:{width:.1f}%"></div>'])
        blocks.append(f"<section><h2>{_esc(title)}</h2>"
                      + _table(["key", "calls", "cost", "tokens", "mean ms", "max ms",
                                "errors", ""], rows, numeric={1, 2, 3, 4, 5, 6}) + "</section>")

    # --- regression -------------------------------------------------------------------------
    for key, body in sorted(regressions.items()):
        rows = []
        for d in body["clusters"]:
            cls = "bad" if d["significant"] and d["diff"] > 0 else (
                "good" if d["significant"] else "muted")
            rows.append([f'<span class="mono">{_esc(d["key"])}</span>', _esc(d["name"]),
                         f"{d['rate_a'] * 100:.2f}%", f"{d['rate_b'] * 100:.2f}%",
                         f"{d['risk_ratio']:.2f}",
                         f"[{d['rr_ci'][0]:.2f}, {d['rr_ci'][1]:.2f}]",
                         f"{d['p_value']:.4f}", f"{d['p_adjusted']:.4f}",
                         f'<span class="{cls}">{_esc(d["direction"])}</span>'])
        grew = [d for d in body["clusters"] if d["significant"] and d["diff"] > 0]
        named = ", ".join("{} ({}) x{:.2f}".format(d["key"], d["name"], d["risk_ratio"])
                          for d in grew)
        verdict = (f'<p class="bad">Regressed: {_esc(named)}.</p>' if grew
                   else '<p class="good">No cluster grew significantly.</p>')
        blocks.append(f"<section><h2>Regression: {_esc(key)}</h2>"
                      f"<p>{_esc(body['label_a'])} n={_fmt(body['n_a'])} against "
                      f"{_esc(body['label_b'])} n={_fmt(body['n_b'])}. Permutation test on each "
                      f"cluster's prevalence, Benjamini-Hochberg at alpha "
                      f"{body['alpha']}.</p>{verdict}"
                      + _table(["cluster", "name", body["label_a"], body["label_b"], "RR",
                                "RR 95% CI", "p", "q", ""], rows,
                               numeric={2, 3, 4, 5, 6, 7}) + "</section>")

    statuses = overview.get("statuses") or {}
    if statuses:
        blocks.append("<section><h2>Termination</h2>"
                      + _table(["status", "runs"], [[_esc(k), _fmt(v)]
                                                    for k, v in statuses.items()], numeric={1})
                      + "</section>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>agent-observatory report</title>
<style>{css}</style></head>
<body>
<header><h1>agent-observatory</h1>
<span class="muted">failure taxonomy over {_fmt(overview.get('n_runs'))} runs</span>
<span class="muted" style="margin-left:auto">{_esc(generated)}</span></header>
<main>{''.join(blocks)}</main>
<footer>Self-contained: no external stylesheets, fonts, scripts or requests.
Reproduce with <code>obs analyse</code> then <code>obs report</code>.</footer>
</body></html>
"""


def write_report(store: Store, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(store))
    return out


def report_assets_are_local(text: str) -> bool:
    """True when the rendered report contains nothing that would hit the network."""
    lowered = text.lower()
    banned = ("http://", "https://", "//cdn", "<link", "@import", "src=", "fetch(",
              "xmlhttprequest")
    return not any(token in lowered for token in banned)
