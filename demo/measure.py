"""Produce every number in MEASUREMENTS.md. Offline, no credentials, deterministic."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observatory.cluster import cluster, silhouette, spherical_kmeans, subsample_stability
from observatory.corpus import Generator, shifted_weights
from observatory.derive import derive
from observatory.events import normalise
from observatory.features import Vectorizer, featurize
from observatory.pipeline import analyse
from observatory.regression import compare_populations
from observatory.validate import evaluate_clustering, evaluate_detectors

OUT = Path(__file__).parent / "measurements.json"


def prepared(seed: int, cohort: str, n: int, weights=None):
    streams, labels = Generator(seed=seed, cohort=cohort, weights=weights).generate(n)
    trajectories = [[normalise(e) for e in s] for s in streams]
    return trajectories, labels


def taxonomy_numbers(trajectories, labels, *, tag: str, k=None, stability_reps=8):
    t0 = time.perf_counter()
    analysis = analyse(trajectories, k=k, k_range=(5, 16), seed=0,
                       truth={lab.run_id: lab.mode for lab in labels},
                       stability_reps=stability_reps)
    elapsed = time.perf_counter() - t0
    report = analysis.taxonomy
    assert report is not None
    detectors = evaluate_detectors(list(analysis.facts.values()),
                                   {lab.run_id: lab.pathologies for lab in labels})
    print(f"\n=== {tag}: {len(trajectories)} runs, analysed in {elapsed:.1f}s")
    print(report.summary())
    print(f"coarse k {analysis.clustering.coarse_k} -> refined k {analysis.clustering.k}")
    print(f"{'cluster':<7} {'n':>5} {'share':>6} {'coh':>5}  {'dominant':<20} name")
    for q in report.clusters:
        flag = "  INCOHERENT" if q.incoherent else ""
        print(f"{q.slug:<7} {q.size:>5} {q.dominant_share:>6.2f} {q.cohesion:>5.2f}  "
              f"{q.dominant_label:<20} {q.name[:52]}{flag}")
    print("planted mode recovery:")
    for mode, cov in sorted(report.mode_coverage.items(), key=lambda kv: -kv[1]):
        print(f"  {mode:<22} {cov:>6.1%} in its largest cluster, "
              f"{report.mode_split[mode]} cluster(s)")
    print(f"{'detector':<24} {'fired':>6} {'tp':>5} {'fp':>4} {'fn':>4} {'prec':>6} "
          f"{'95% ci':>14} {'recall':>7} {'f1':>6}")
    for s in detectors:
        ci = f"[{s.precision_ci[0]:.2f},{s.precision_ci[1]:.2f}]"
        print(f"{s.pathology:<24} {s.fired:>6} {s.true_positives:>5} {s.false_positives:>4} "
              f"{s.missed:>4} {s.precision:>6.3f} {ci:>14} {s.recall:>7.3f} {s.f1:>6.3f}")
    return analysis, report, detectors, elapsed


def main() -> int:
    results: dict = {"env": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor() or platform.machine(),
    }}
    print(json.dumps(results["env"], indent=1))

    # 1. Headline: one 2,400-run cohort with the baseline mode mix.
    trajectories, labels = prepared(7, "baseline", 2400)
    analysis, report, detectors, elapsed = taxonomy_numbers(trajectories, labels,
                                                           tag="baseline cohort")
    results["headline"] = {
        "n_runs": len(trajectories), "seconds": elapsed,
        "overview": analysis.overview(),
        "coarse_k": analysis.clustering.coarse_k,
        "purity": report.purity, "inverse_purity": report.inverse_purity,
        "adjusted_rand": report.adjusted_rand, "nmi": report.nmi,
        "silhouette": report.silhouette, "noise_fraction": report.noise_fraction,
        "stability_ari": report.stability_ari,
        "mode_coverage": report.mode_coverage, "mode_split": report.mode_split,
        "incoherent": report.incoherent,
        "clusters": [{"slug": q.slug, "name": q.name, "size": q.size,
                      "dominant_label": q.dominant_label, "dominant_share": q.dominant_share,
                      "cohesion": q.cohesion, "entropy": q.entropy,
                      "label_mix": q.label_mix, "incoherent": q.incoherent}
                     for q in report.clusters],
        "detectors": [{"pathology": s.pathology, "fired": s.fired, "tp": s.true_positives,
                       "fp": s.false_positives, "fn": s.missed, "precision": s.precision,
                       "precision_ci": list(s.precision_ci), "recall": s.recall, "f1": s.f1,
                       "examples_fp": s.examples_fp}
                      for s in detectors],
    }

    # 2. Ablations: what each modelling decision is worth, on the same corpus.
    facts = [derive(t) for t in trajectories]
    raw = [featurize(f) for f in facts]
    ids = [f.run_id for f in facts]
    truth = {lab.run_id: lab.mode for lab in labels}
    print("\n=== ablations on the same 2,400 runs")
    ablations = {}
    for name, kwargs, vkwargs in [
        ("full", {}, {}),
        ("no refinement", {"refine": False}, {}),
        ("one-hot scalars instead of cumulative", {}, {"one_hot": True}),
        ("one block weight for all blocks", {}, {"flat_weights": True}),
        ("k by silhouette alone", {"select": "silhouette"}, {}),
        ("k by stability alone", {"select": "stability"}, {}),
    ]:
        v = Vectorizer(min_df=3)
        if vkwargs.get("flat_weights"):
            v.block_weights = dict.fromkeys(v.block_weights, 1.0)
        x = v.fit_transform(raw) if not vkwargs.get("one_hot") else _one_hot(v, raw)
        c = cluster(x, ids, v.column_names(), k_range=(5, 16), seed=0, stability_reps=8, **kwargs)
        r = evaluate_clustering(c, truth)
        ablations[name] = {"k": r.k, "purity": r.purity, "inverse_purity": r.inverse_purity,
                          "adjusted_rand": r.adjusted_rand, "nmi": r.nmi,
                          "incoherent": r.incoherent}
        print(f"  {name:<40} k={r.k:<3} purity={r.purity:.3f} inv={r.inverse_purity:.3f} "
              f"ARI={r.adjusted_rand:.3f} NMI={r.nmi:.3f} incoherent={len(r.incoherent)}")
    results["ablations"] = ablations

    # 3. k-selection criteria against the oracle, over four independent corpus seeds.
    print("\n=== k selection criteria, four corpus seeds of 800 runs each")
    rows_by_seed = {}
    for seed in (1, 2, 3, 4):
        tr, lab = prepared(seed, "baseline", 800)
        f = [derive(t) for t in tr]
        rw = [featurize(x) for x in f]
        i = [x.run_id for x in f]
        v = Vectorizer(min_df=3)
        x = v.fit_transform(rw)
        t = {la.run_id: la.mode for la in lab}
        rows = []
        for k in range(5, 14):
            lbl, _, _ = spherical_kmeans(x, k, n_init=4, seed=0)
            sil = silhouette(x, lbl, seed=0)
            stab = subsample_stability(x, k, reps=8, seed=0)
            r = evaluate_clustering(cluster(x, i, v.column_names(), k=k, seed=0), t)
            rows.append({"k": k, "silhouette": sil, "stability": stab,
                         "purity": r.purity, "adjusted_rand": r.adjusted_rand})
        rows_by_seed[seed] = rows
    criteria = {"silhouette": lambda r: r["silhouette"],
                "stability": lambda r: r["stability"],
                "silhouette x stability": lambda r: r["silhouette"] * r["stability"],
                "oracle (argmax ARI)": lambda r: r["adjusted_rand"]}
    results["k_selection"] = {"per_seed": rows_by_seed, "criteria": {}}
    for name, key in criteria.items():
        picks = [max(rows_by_seed[s], key=key) for s in rows_by_seed]
        aris = [p["adjusted_rand"] for p in picks]
        results["k_selection"]["criteria"][name] = {
            "ks": [p["k"] for p in picks], "mean_ari": float(np.mean(aris)),
            "min_ari": float(min(aris))}
        print(f"  {name:<24} k={[p['k'] for p in picks]} "
              f"mean ARI={np.mean(aris):.3f} min={min(aris):.3f}")

    # 4. Generalisation: the chosen configuration on seeds it was not tuned on.
    print("\n=== generalisation, 1,200 runs per seed")
    gen = []
    for seed in (51, 52, 53, 54, 55):
        tr, lab = prepared(seed, "baseline", 1200)
        a = analyse(tr, k_range=(5, 16), seed=0,
                    truth={la.run_id: la.mode for la in lab})
        r = a.taxonomy
        gen.append({"seed": seed, "k": r.k, "purity": r.purity,
                    "inverse_purity": r.inverse_purity, "adjusted_rand": r.adjusted_rand,
                    "nmi": r.nmi, "incoherent": r.incoherent})
        print(f"  seed {seed}: {r.summary()} incoherent={r.incoherent}")
    results["generalisation"] = {
        "runs": gen,
        "purity_mean": float(np.mean([g["purity"] for g in gen])),
        "purity_min": float(min(g["purity"] for g in gen)),
        "ari_mean": float(np.mean([g["adjusted_rand"] for g in gen])),
        "ari_min": float(min(g["adjusted_rand"] for g in gen)),
    }

    # 5. Regression view: a planted shift, and a null comparison.
    for tag, after_weights, after_seed in [
        ("planted 2.5x retry_storm", shifted_weights(mode="retry_storm", multiplier=2.5), 72),
        ("null (identical weights)", None, 73),
    ]:
        before, before_labels = prepared(71, "before", 2000)
        after, after_labels = prepared(after_seed, "after", 2000, weights=after_weights)
        pooled = before + after
        a = analyse(pooled, k_range=(5, 16), seed=0)
        t0 = time.perf_counter()
        rep = compare_populations(a.clustering, a.cohort_members("before"),
                                 a.cohort_members("after"),
                                 all_facts=list(a.facts.values()), n_permutations=20000, seed=0)
        secs = time.perf_counter() - t0
        print(f"\n=== regression: {tag}  ({secs:.1f}s for {len(rep.clusters)} "
              f"+ {len(rep.pathologies)} tests)")
        print(rep.render())
        planted = {}
        for lab in before_labels:
            planted.setdefault("before", {})
            planted["before"][lab.mode] = planted["before"].get(lab.mode, 0) + 1
        for lab in after_labels:
            planted.setdefault("after", {})
            planted["after"][lab.mode] = planted["after"].get(lab.mode, 0) + 1
        results.setdefault("regression", {})[tag] = {
            "n_a": rep.n_a, "n_b": rep.n_b, "seconds": secs,
            "planted_mode_counts": planted,
            "k": a.clustering.k,
            "clusters": [{"key": d.key, "name": d.name, "rate_a": d.test.rate_a,
                          "rate_b": d.test.rate_b, "risk_ratio": d.test.risk_ratio,
                          "rr_ci": list(d.test.rr_ci), "p": d.test.p_value,
                          "q": d.p_adjusted, "direction": d.direction} for d in rep.clusters],
            "pathologies": [{"key": d.name, "rate_a": d.test.rate_a, "rate_b": d.test.rate_b,
                             "risk_ratio": d.test.risk_ratio, "rr_ci": list(d.test.rr_ci),
                             "p": d.test.p_value, "q": d.p_adjusted,
                             "direction": d.direction} for d in rep.pathologies],
            "grew": [d.key for d in rep.grew],
            "shrank": [d.key for d in rep.shrank],
        }

    # 6. Throughput and artefact sizes.
    print("\n=== throughput")
    timings = {}
    for n in (300, 1200, 2400):
        tr, _ = prepared(61, "baseline", n)
        t0 = time.perf_counter()
        analyse(tr, k_range=(5, 12), seed=0)
        timings[n] = time.perf_counter() - t0
        events = sum(len(t) for t in tr)
        print(f"  {n:>5} runs / {events:>7} events: {timings[n]:>6.1f}s "
              f"({events / timings[n]:,.0f} events/s)")
    results["throughput"] = timings

    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {OUT}")
    return 0


def _one_hot(v: Vectorizer, raw) -> np.ndarray:
    """Re-run the vectoriser with one-hot bucket encoding, for the ablation."""
    import math

    from observatory.features import SCALAR_BLOCKS
    v.fit(raw)
    matrix = np.zeros((len(raw), len(v.columns)))
    for row, item in enumerate(raw):
        for key, count in item.counts.items():
            col = v.index.get(key)
            if col is not None:
                matrix[row, col] = 1.0 + math.log(count) if count > 1 else 1.0
        for key, on in item.flags.items():
            col = v.index.get(key)
            if on and col is not None:
                matrix[row, col] = 1.0
        for name, block in SCALAR_BLOCKS.items():
            bucket = v._bucket(name, item.scalars.get(name, 0.0))
            col = v.index.get((block, f"s:{name}:b{bucket}"))
            if col is not None:
                matrix[row, col] = 1.0
    return v._normalise(matrix)


if __name__ == "__main__":
    subprocess.run(["true"], check=False)
    raise SystemExit(main())
