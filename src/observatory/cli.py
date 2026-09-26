"""The obs command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attribution import PriceTable
from .corpus import BASELINE_WEIGHTS, load_labels, shifted_weights, write_corpus
from .otlp import post_otlp, write_otlp
from .pipeline import analyse, load_trajectories, persist
from .regression import compare_populations
from .report import write_report
from .spans import Status
from .store import Store
from .validate import evaluate_detectors

DEFAULT_DB = Path("obs_data/observatory.db")


def _load(paths: list[Path], prices: PriceTable) -> tuple[list, dict[str, list]]:
    trajectories = load_trajectories(paths)
    if not trajectories:
        raise SystemExit(f"no trajectories found under {', '.join(str(p) for p in paths)}")
    del prices
    return trajectories, {t[0].run_id: t for t in trajectories}


def _prices(args: argparse.Namespace) -> PriceTable:
    return PriceTable.from_file(Path(args.prices)) if args.prices else PriceTable()


def _truth(paths: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    modes: dict[str, str] = {}
    pathologies: dict[str, list[str]] = {}
    for path in paths:
        for run_id, label in load_labels(Path(path)).items():
            modes[run_id] = label.mode
            pathologies[run_id] = label.pathologies
    return modes, pathologies


def cmd_gen(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if args.with_regression:
        weights = shifted_weights(BASELINE_WEIGHTS, mode=args.shift_mode,
                                  multiplier=args.shift_multiplier)
        summaries = [
            write_corpus(out, n_runs=args.n, seed=args.seed, cohort="before",
                         shard_size=args.shard_size),
            write_corpus(out, n_runs=args.n, seed=args.seed + 1, cohort="after",
                         weights=weights, shard_size=args.shard_size),
        ]
    else:
        summaries = [write_corpus(out, n_runs=args.n, seed=args.seed, cohort=args.cohort,
                                  shard_size=args.shard_size)]
    for summary in summaries:
        print(f"{summary['cohort']:<10} runs={summary['n_runs']:<6} "
              f"events={summary['n_events']:<7} -> {summary['runs_dir']}")
        for mode, count in summary["mode_counts"].items():
            print(f"    {mode:<20} {count:>6}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    prices = _prices(args)
    trajectories, by_run = _load([Path(p) for p in args.paths], prices)
    modes, pathologies = _truth(args.labels or [])
    analysis = analyse(trajectories, prices=prices, k=args.k,
                       k_range=(args.k_min, args.k_max), seed=args.seed,
                       min_df=args.min_df, noise_z=args.noise_z,
                       truth=modes or None, stability_reps=args.stability_reps)
    store = Store(Path(args.db))
    persist(analysis, store, trajectories=by_run)

    overview = analysis.overview()
    print(f"runs {overview['n_runs']}  events {overview['n_events']}  "
          f"spans {overview['n_spans']}  clusters {overview['k']}  "
          f"noise {overview['noise_fraction']:.1%}")
    print(f"cost ${overview['total_cost_usd']:.4f}  tokens {overview['total_tokens']:,}  "
          f"features {overview['feature_columns']}")
    print()
    print(f"{'cluster':<8} {'n':>5}  {'coh':>5}  name")
    for c in analysis.clustering.clusters:
        print(f"{c.slug:<8} {c.size:>5}  {c.cohesion:>5.2f}  {c.name}")
    if analysis.clustering.noise:
        print(f"{'noise':<8} {len(analysis.clustering.noise):>5}         "
              f"runs no cluster explains")
    if analysis.taxonomy:
        print()
        print(analysis.taxonomy.summary())
        for q in analysis.taxonomy.clusters:
            flag = "  INCOHERENT" if q.incoherent else ""
            print(f"  {q.slug:<6} n={q.size:<5} {q.dominant_label:<20} "
                  f"share={q.dominant_share:.2f}{flag}")
        if analysis.taxonomy.incoherent:
            print(f"  incoherent clusters: {', '.join(analysis.taxonomy.incoherent)}")
    if pathologies:
        print()
        print(f"{'detector':<24} {'fired':>6} {'prec':>6} {'95% ci':>14} {'recall':>7}")
        scores = evaluate_detectors(list(analysis.facts.values()), pathologies)
        store.put_aggregate("validation", "detectors", [{
            "pathology": s.pathology, "fired": s.fired, "precision": s.precision,
            "recall": s.recall, "f1": s.f1, "precision_ci": list(s.precision_ci),
            "true_positives": s.true_positives, "false_positives": s.false_positives,
            "missed": s.missed, "examples_fp": s.examples_fp, "examples_fn": s.examples_fn,
        } for s in scores])
        for s in scores:
            ci = f"[{s.precision_ci[0]:.2f},{s.precision_ci[1]:.2f}]"
            print(f"{s.pathology:<24} {s.fired:>6} {s.precision:>6.3f} {ci:>14} {s.recall:>7.3f}")
    print()
    print(f"wrote {args.db}")
    return 0


def cmd_clusters(args: argparse.Namespace) -> int:
    store = Store(Path(args.db))
    for c in store.clusters():
        quality = c.get("quality") or {}
        flag = "  INCOHERENT" if quality.get("incoherent") else ""
        print(f"{c['slug']:<6} n={c['size']:<6} coh={c['cohesion']:.2f}  {c['name']}{flag}")
        if args.verbose:
            for name, weight in c["top_features"][:8]:
                print(f"         {weight:+.3f}  {name}")
            print(f"         examples: {', '.join(c['representatives'][:3])}")
            if quality:
                print(f"         dominant {quality['dominant_label']} "
                      f"{quality['dominant_share']:.0%}  mix={quality['label_mix']}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    store = Store(Path(args.db))
    rows = store.list_runs(cluster=args.cluster, cohort=args.cohort, status=args.status,
                           pathology=args.pathology, order=args.order, limit=args.limit)
    print(f"{'run_id':<28} {'status':<13} {'ev':>4} {'tools':>5} {'cost$':>9} {'ms':>9} cluster")
    for r in rows:
        print(f"{r['run_id'][:28]:<28} {(r['status'] or '')[:13]:<13} {r['n_events']:>4} "
              f"{r['n_tool_calls']:>5} {r['cost_usd']:>9.5f} {r['wall_ms']:>9.0f} "
              f"{r['cluster_slug'] or '-'} {','.join(r['pathologies'])}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from .spans import build_trace
    store = Store(Path(args.db))
    row = store.get_run(args.run_id)
    if row is None:
        raise SystemExit(f"no such run: {args.run_id}")
    events = store.get_events(args.run_id)
    trace = build_trace(events)
    attribution_rows = {h["name"]: h for h in row["hotspots"]}

    print(f"{row['run_id']}   status={row['status']}   cluster={row['cluster_slug']} "
          f"({row['cluster_name']})")
    print(f"events={row['n_events']} spans={len(trace.spans)} cost=${row['cost_usd']:.6f} "
          f"tokens={row['tokens_in']}+{row['tokens_out']} wall={row['wall_ms']:.0f}ms")
    if row["findings"]:
        print()
        for f in row["findings"]:
            print(f"  {f['pathology']:<24} strength={f['strength']:.2f}  "
                  f"{json.dumps(f['evidence'])[:110]}")
    print()
    for depth, span in trace.walk():
        mark = "!" if span.status is Status.ERROR else " "
        tok = ""
        tin = span.attributes.get("gen_ai.usage.input_tokens")
        if tin is not None:
            tok = f"  {tin}+{span.attributes.get('gen_ai.usage.output_tokens', 0)}tok"
        print(f"{mark} {'  ' * depth}{span.name[:60]:<60} {span.duration_ms:>9.1f}ms{tok}")
    del attribution_rows
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    store = Store(Path(args.db))
    population = store.aggregate("attribution", "population") or {}
    for group in ("by_tool", "by_model", "by_step"):
        buckets = population.get(group) or {}
        if not buckets:
            continue
        print(f"{group}")
        print(f"  {'key':<24} {'calls':>7} {'cost$':>10} {'tokens':>11} "
              f"{'mean ms':>9} {'max ms':>9} {'err':>6}")
        ranked = sorted(buckets.items(), key=lambda kv: -kv[1]["cost_usd"]
                        if group != "by_tool" else -kv[1]["total_ms"])
        for key, b in ranked[:args.limit]:
            print(f"  {key[:24]:<24} {b['calls']:>7.0f} {b['cost_usd']:>10.5f} "
                  f"{b['tokens_in'] + b['tokens_out']:>11,.0f} {b['mean_ms']:>9.1f} "
                  f"{b['max_ms']:>9.1f} {b['error_rate']:>6.1%}")
        print()
    return 0


def cmd_regress(args: argparse.Namespace) -> int:
    prices = _prices(args)
    trajectories, _ = _load([Path(p) for p in args.paths], prices)
    analysis = analyse(trajectories, prices=prices, k=args.k,
                       k_range=(args.k_min, args.k_max), seed=args.seed)
    group_a = analysis.cohort_members(args.before)
    group_b = analysis.cohort_members(args.after)
    if not group_a or not group_b:
        raise SystemExit(f"cohorts {args.before!r} n={len(group_a)} and {args.after!r} "
                         f"n={len(group_b)}; available: "
                         f"{sorted(set(analysis.cohorts.values()))}")
    report = compare_populations(analysis.clustering, group_a, group_b,
                                 all_facts=list(analysis.facts.values()), alpha=args.alpha,
                                 n_permutations=args.permutations, seed=args.seed,
                                 label_a=args.before, label_b=args.after)
    print(report.render())
    if args.db and Path(args.db).exists():
        Store(Path(args.db)).put_aggregate("regression", f"{args.before}->{args.after}", {
            "n_a": report.n_a, "n_b": report.n_b, "alpha": report.alpha,
            "label_a": report.label_a, "label_b": report.label_b,
            "clusters": [_delta_dict(d) for d in report.clusters],
            "pathologies": [_delta_dict(d) for d in report.pathologies],
        })
    return 0


def _delta_dict(d) -> dict:
    return {"key": d.key, "name": d.name, "rate_a": d.test.rate_a, "rate_b": d.test.rate_b,
            "k_a": d.test.k_a, "k_b": d.test.k_b, "risk_ratio": d.test.risk_ratio,
            "rr_ci": list(d.test.rr_ci), "diff": d.test.diff, "diff_ci": list(d.test.diff_ci),
            "p_value": d.test.p_value, "p_adjusted": d.p_adjusted,
            "significant": d.significant, "direction": d.direction}


def cmd_export(args: argparse.Namespace) -> int:
    from .spans import build_trace
    store = Store(Path(args.db))
    run_ids = ([args.run_id] if args.run_id
               else [r["run_id"] for r in store.list_runs(limit=args.limit, order="run_id")])
    traces = [build_trace(store.get_events(rid)) for rid in run_ids]
    if args.endpoint:
        status, body = post_otlp(traces, args.endpoint)
        print(f"POST {args.endpoint} -> {status} {body[:200]}")
        return 0 if 200 <= status < 300 else 1
    size = write_otlp(traces, Path(args.out))
    total = sum(len(t.spans) for t in traces)
    print(f"{len(traces)} traces, {total} spans -> {args.out} ({size:,} bytes)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = Store(Path(args.db))
    if store.count_runs() == 0:
        raise SystemExit(f"{args.db} holds no runs; run `obs analyse` first")
    path = write_report(store, Path(args.out))
    print(f"wrote {path} ({path.stat().st_size:,} bytes, no external requests)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('the web app needs the web extra: uv pip install -e ".[web]"') from None
    from .web.app import create_app
    store = Store(Path(args.db))
    if store.count_runs() == 0:
        raise SystemExit(f"{args.db} holds no runs; run `obs analyse` first")
    print(f"http://{args.host}:{args.port}  ({store.count_runs()} runs)")
    uvicorn.run(create_app(store), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    prices = _prices(args)
    trajectories, _ = _load([Path(p) for p in args.paths], prices)
    modes, pathologies = _truth(args.labels)
    analysis = analyse(trajectories, prices=prices, k=args.k,
                       k_range=(args.k_min, args.k_max), seed=args.seed,
                       truth=modes, stability_reps=args.stability_reps)
    taxonomy = analysis.taxonomy
    assert taxonomy is not None
    print(taxonomy.summary())
    print()
    print(f"{'cluster':<7} {'n':>5} {'share':>6} {'entropy':>8} {'coh':>5}  "
          f"{'dominant':<20} name")
    for q in taxonomy.clusters:
        flag = "  INCOHERENT" if q.incoherent else ""
        print(f"{q.slug:<7} {q.size:>5} {q.dominant_share:>6.2f} {q.entropy:>8.3f} "
              f"{q.cohesion:>5.2f}  {q.dominant_label:<20} {q.name[:44]}{flag}")
    print()
    print("planted mode recovery")
    for mode, cov in sorted(taxonomy.mode_coverage.items(), key=lambda kv: -kv[1]):
        print(f"  {mode:<22} largest cluster holds {cov:>6.1%} of it, "
              f"spread over {taxonomy.mode_split[mode]} cluster(s)")
    if taxonomy.incoherent:
        print()
        print(f"incoherent clusters (dominant label under 50%): "
              f"{', '.join(taxonomy.incoherent)}")
    print()
    print(f"{'detector':<24} {'fired':>6} {'tp':>5} {'fp':>5} {'fn':>5} {'prec':>6} "
          f"{'95% ci':>14} {'recall':>7} {'f1':>6}")
    for s in evaluate_detectors(list(analysis.facts.values()), pathologies):
        ci = f"[{s.precision_ci[0]:.2f},{s.precision_ci[1]:.2f}]"
        print(f"{s.pathology:<24} {s.fired:>6} {s.true_positives:>5} {s.false_positives:>5} "
              f"{s.missed:>5} {s.precision:>6.3f} {ci:>14} {s.recall:>7.3f} {s.f1:>6.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "purity": taxonomy.purity, "inverse_purity": taxonomy.inverse_purity,
            "adjusted_rand": taxonomy.adjusted_rand, "nmi": taxonomy.nmi, "k": taxonomy.k,
            "silhouette": taxonomy.silhouette, "noise_fraction": taxonomy.noise_fraction,
            "stability_ari": taxonomy.stability_ari,
            "mode_coverage": taxonomy.mode_coverage, "incoherent": taxonomy.incoherent,
            "detectors": [{"pathology": s.pathology, "precision": s.precision,
                           "recall": s.recall, "fired": s.fired,
                           "precision_ci": list(s.precision_ci)}
                          for s in evaluate_detectors(list(analysis.facts.values()),
                                                      pathologies)],
        }, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def cmd_summarise(args: argparse.Namespace) -> int:
    """Optional LLM narration of a cluster. Never required; nothing in the tests calls it."""
    from .summarise import summarise_cluster
    store = Store(Path(args.db))
    clusters = {c["slug"]: c for c in store.clusters()}
    target = clusters.get(args.cluster)
    if target is None:
        raise SystemExit(f"no cluster {args.cluster!r}; have {', '.join(clusters)}")
    text = summarise_cluster(store, target, model=args.model, offline=args.offline)
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obs", description="Observability and automatic failure taxonomy for agent runs.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="analysis store path")
    parser.add_argument("--prices", help="JSON token price table overriding the built-in one")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen", help="generate a synthetic corpus with planted failure modes")
    p.add_argument("out", help="output directory")
    p.add_argument("-n", type=int, default=2000, help="runs per cohort")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cohort", default="baseline")
    p.add_argument("--shard-size", type=int, default=250)
    p.add_argument("--with-regression", action="store_true",
                   help="write before/after cohorts with one mode made more common")
    p.add_argument("--shift-mode", default="retry_storm")
    p.add_argument("--shift-multiplier", type=float, default=2.5)
    p.set_defaults(func=cmd_gen)

    p = sub.add_parser("analyse", help="ingest, cluster, detect, and write the store")
    p.add_argument("paths", nargs="+", help="trajectory files or directories")
    p.add_argument("--labels", action="append", help="ground truth sidecar, repeatable")
    p.add_argument("--k", type=int, help="fix the cluster count instead of choosing it")
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument("--min-df", type=int, default=3)
    p.add_argument("--noise-z", type=float, default=3.5)
    p.add_argument("--stability-reps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("clusters", help="list the derived failure taxonomy")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_clusters)

    p = sub.add_parser("runs", help="list runs")
    p.add_argument("--cluster")
    p.add_argument("--cohort")
    p.add_argument("--status")
    p.add_argument("--pathology")
    p.add_argument("--order", default="cost",
                   choices=("cost", "duration", "events", "run_id", "errors"))
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="print one run's span tree")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("cost", help="cost and latency attribution across the population")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("regress", help="compare two cohorts and test which clusters grew")
    p.add_argument("paths", nargs="+")
    p.add_argument("--before", default="before")
    p.add_argument("--after", default="after")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--permutations", type=int, default=20000)
    p.add_argument("--k", type=int)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_regress)

    p = sub.add_parser("export", help="write OTLP/JSON, or post it to a collector")
    p.add_argument("--run-id")
    p.add_argument("--out", default="out/traces.otlp.json")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--endpoint", help="OTLP/HTTP traces endpoint; omit to write a file")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("report", help="write the self-contained HTML report")
    p.add_argument("--out", default="out/report.html")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("serve", help="run the local web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--log-level", default="warning")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("validate", help="score the taxonomy and detectors against ground truth")
    p.add_argument("paths", nargs="+")
    p.add_argument("--labels", action="append", required=True)
    p.add_argument("--k", type=int)
    p.add_argument("--k-min", type=int, default=5)
    p.add_argument("--k-max", type=int, default=16)
    p.add_argument("--stability-reps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write the numbers to this file")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("summarise", help="optional: narrate one cluster with a real model")
    p.add_argument("cluster")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--offline", action="store_true",
                   help="build the prompt and render the deterministic summary, call nothing")
    p.set_defaults(func=cmd_summarise)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
