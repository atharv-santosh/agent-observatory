"""Validating the taxonomy, which is the difference between analysis and decoration.

Purity alone is easy to game: split the corpus into two hundred clusters of ten and every cluster is
pure. So purity is reported alongside adjusted Rand and normalised mutual information, which punish
that, and alongside per-mode coverage, which catches the opposite failure of one planted mode being
shattered across five clusters. A cluster whose dominant label accounts for less than a threshold of
its members is reported as incoherent by name rather than quietly averaged away.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .cluster import Clustering, subsample_stability
from .derive import RunFacts
from .detectors import PATHOLOGIES, run_detectors
from .stats import wilson_interval


@dataclass(slots=True)
class ClusterQuality:
    slug: str
    name: str
    size: int
    dominant_label: str
    dominant_share: float
    purity_contribution: float
    entropy: float
    cohesion: float
    label_mix: dict[str, int] = field(default_factory=dict)

    @property
    def incoherent(self) -> bool:
        return self.dominant_share < 0.5


@dataclass(slots=True)
class TaxonomyReport:
    n_runs: int
    n_labelled: int
    k: int
    purity: float
    inverse_purity: float
    adjusted_rand: float
    nmi: float
    silhouette: float
    noise_fraction: float
    clusters: list[ClusterQuality] = field(default_factory=list)
    mode_coverage: dict[str, float] = field(default_factory=dict)
    mode_split: dict[str, int] = field(default_factory=dict)
    incoherent: list[str] = field(default_factory=list)
    stability_ari: float | None = None
    stability_reps: int = 0

    def summary(self) -> str:
        parts = [
            f"k={self.k}",
            f"purity={self.purity:.3f}",
            f"inv_purity={self.inverse_purity:.3f}",
            f"ARI={self.adjusted_rand:.3f}",
            f"NMI={self.nmi:.3f}",
            f"silhouette={self.silhouette:.3f}",
            f"noise={self.noise_fraction:.3f}",
        ]
        if self.stability_ari is not None:
            parts.append(f"stability_ARI={self.stability_ari:.3f}")
        return "  ".join(parts)


def _contingency(truth: list[str], pred: list[int]) -> dict[tuple[str, int], int]:
    table: dict[tuple[str, int], int] = {}
    for t, p in zip(truth, pred, strict=True):
        table[(t, p)] = table.get((t, p), 0) + 1
    return table


def purity(truth: list[str], pred: list[int]) -> float:
    """Fraction of runs whose cluster's majority label is their own label."""
    if not truth:
        return 0.0
    by_cluster: dict[int, Counter[str]] = {}
    for t, p in zip(truth, pred, strict=True):
        by_cluster.setdefault(p, Counter())[t] += 1
    return sum(c.most_common(1)[0][1] for c in by_cluster.values()) / len(truth)


def inverse_purity(truth: list[str], pred: list[int]) -> float:
    """The symmetric view: how concentrated each true label is in a single cluster."""
    if not truth:
        return 0.0
    by_label: dict[str, Counter[int]] = {}
    for t, p in zip(truth, pred, strict=True):
        by_label.setdefault(t, Counter())[p] += 1
    return sum(c.most_common(1)[0][1] for c in by_label.values()) / len(truth)


def adjusted_rand(truth: list[str], pred: list[int]) -> float:
    """Adjusted Rand index, corrected for chance agreement."""
    n = len(truth)
    if n < 2:
        return 0.0
    table = _contingency(truth, pred)
    row = Counter(truth)
    col = Counter(pred)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    index = sum(comb(v) for v in table.values())
    exp_row = sum(comb(v) for v in row.values())
    exp_col = sum(comb(v) for v in col.values())
    expected = exp_row * exp_col / comb(n)
    maximum = (exp_row + exp_col) / 2
    return (index - expected) / (maximum - expected) if maximum != expected else 0.0


def nmi(truth: list[str], pred: list[int]) -> float:
    """Normalised mutual information with arithmetic-mean normalisation."""
    n = len(truth)
    if n == 0:
        return 0.0
    table = _contingency(truth, pred)
    row = Counter(truth)
    col = Counter(pred)
    mutual = 0.0
    for (t, p), count in table.items():
        joint = count / n
        mutual += joint * math.log(joint / ((row[t] / n) * (col[p] / n)))
    h_truth = -sum((v / n) * math.log(v / n) for v in row.values())
    h_pred = -sum((v / n) * math.log(v / n) for v in col.values())
    denom = (h_truth + h_pred) / 2
    return mutual / denom if denom > 0 else 0.0


def evaluate_clustering(clustering: Clustering, truth: dict[str, str], *,
                        include_noise: bool = False,
                        incoherent_threshold: float = 0.5) -> TaxonomyReport:
    """Score a clustering against ground-truth labels, per cluster and overall."""
    pairs = [(rid, int(lab)) for rid, lab in zip(clustering.run_ids, clustering.labels,
                                                 strict=True)
             if rid in truth and (include_noise or lab >= 0)]
    labels_t = [truth[rid] for rid, _ in pairs]
    labels_p = [lab for _, lab in pairs]

    by_cid = {c.cid: c for c in clustering.clusters}
    quality: list[ClusterQuality] = []
    grouped: dict[int, list[str]] = {}
    for t, p in zip(labels_t, labels_p, strict=True):
        grouped.setdefault(p, []).append(t)
    total = len(labels_t) or 1
    for cid, members in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        counts = Counter(members)
        dominant, n_dominant = counts.most_common(1)[0]
        h = -sum((v / len(members)) * math.log(v / len(members)) for v in counts.values())
        c = by_cid.get(cid)
        quality.append(ClusterQuality(
            slug=c.slug if c else f"c{cid:02d}",
            name=c.name if c else "noise",
            size=len(members),
            dominant_label=dominant,
            dominant_share=n_dominant / len(members),
            purity_contribution=n_dominant / total,
            entropy=h,
            cohesion=c.cohesion if c else 0.0,
            label_mix=dict(counts.most_common()),
        ))

    coverage: dict[str, float] = {}
    split: dict[str, int] = {}
    for mode in sorted(set(labels_t)):
        cids = Counter(p for t, p in zip(labels_t, labels_p, strict=True) if t == mode)
        n_mode = sum(cids.values())
        coverage[mode] = cids.most_common(1)[0][1] / n_mode if n_mode else 0.0
        split[mode] = len([c for c, v in cids.items() if v >= max(2, 0.05 * n_mode)])

    return TaxonomyReport(
        n_runs=len(clustering.run_ids),
        n_labelled=len(labels_t),
        k=clustering.k,
        purity=purity(labels_t, labels_p),
        inverse_purity=inverse_purity(labels_t, labels_p),
        adjusted_rand=adjusted_rand(labels_t, labels_p),
        nmi=nmi(labels_t, labels_p),
        silhouette=clustering.silhouette,
        noise_fraction=clustering.noise_fraction,
        clusters=quality,
        mode_coverage=coverage,
        mode_split=split,
        incoherent=[q.slug for q in quality if q.dominant_share < incoherent_threshold],
    )


def stability(x: np.ndarray, run_ids: list[str], column_names: list[str], *, k: int,
              reps: int = 8, fraction: float = 0.8, seed: int = 0) -> float:
    """Mean adjusted Rand between clusterings of overlapping subsamples at the chosen k.

    Answers the question purity cannot: is the structure a property of the data or of this
    particular run of k-means. Shares its implementation with the k-selection criterion, because two
    definitions of stability would eventually disagree.
    """
    del run_ids, column_names
    return subsample_stability(x, k, reps=reps, fraction=fraction, seed=seed)


@dataclass(slots=True)
class DetectorScore:
    pathology: str
    fired: int
    true_positives: int
    false_positives: int
    missed: int
    precision: float
    recall: float
    precision_ci: tuple[float, float]
    examples_fp: list[str] = field(default_factory=list)
    examples_fn: list[str] = field(default_factory=list)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_detectors(all_facts: list[RunFacts], truth: dict[str, list[str]], *,
                       min_strength: float = 0.0) -> list[DetectorScore]:
    """Precision and recall per detector against per-run pathology labels."""
    fired: dict[str, set[str]] = {p: set() for p in PATHOLOGIES}
    for facts in all_facts:
        if facts.run_id not in truth:
            continue
        for finding in run_detectors(facts):
            if finding.strength >= min_strength:
                fired[finding.pathology].add(finding.run_id)

    scored = []
    for pathology in PATHOLOGIES:
        actual = {rid for rid, labels in truth.items() if pathology in labels}
        hits = fired[pathology] & actual
        fps = fired[pathology] - actual
        fns = actual - fired[pathology]
        n_fired = len(fired[pathology])
        scored.append(DetectorScore(
            pathology=pathology,
            fired=n_fired,
            true_positives=len(hits),
            false_positives=len(fps),
            missed=len(fns),
            precision=len(hits) / n_fired if n_fired else 0.0,
            recall=len(hits) / len(actual) if actual else 0.0,
            precision_ci=wilson_interval(len(hits), n_fired),
            examples_fp=sorted(fps)[:5],
            examples_fn=sorted(fns)[:5],
        ))
    return scored
