"""Comparing two populations of runs and saying which failure clusters actually grew.

The trap here is clustering each population separately and then trying to match cluster 3 in one to
cluster 7 in the other. Cluster identity is not portable, so the two populations are pooled and
clustered once, and only then split back apart. The clusters are the same objects in both halves by
construction, which is the only way the comparison means anything.

Then, because a fifteen-cluster comparison is fifteen hypothesis tests, the p-values go through
Benjamini-Hochberg. Without that, about one cluster will look like a regression every time you run
this on two populations that are identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cluster import Clustering
from .derive import RunFacts
from .detectors import PATHOLOGIES, run_detectors
from .stats import ProportionTest, benjamini_hochberg, compare_proportions


@dataclass(slots=True)
class ClusterDelta:
    key: str
    name: str
    test: ProportionTest
    p_adjusted: float
    significant: bool

    @property
    def direction(self) -> str:
        if not self.significant:
            return "flat"
        return "grew" if self.test.diff > 0 else "shrank"

    def line(self) -> str:
        t = self.test
        return (f"{self.key:<8} {self.name[:46]:<46} "
                f"{t.rate_a * 100:6.2f}% -> {t.rate_b * 100:6.2f}%  "
                f"RR {t.risk_ratio:5.2f} [{t.rr_ci[0]:.2f},{t.rr_ci[1]:.2f}]  "
                f"p={t.p_value:.4f} q={self.p_adjusted:.4f}  {self.direction}")


@dataclass(slots=True)
class RegressionReport:
    n_a: int
    n_b: int
    alpha: float
    clusters: list[ClusterDelta] = field(default_factory=list)
    pathologies: list[ClusterDelta] = field(default_factory=list)
    label_a: str = "before"
    label_b: str = "after"

    @property
    def grew(self) -> list[ClusterDelta]:
        return [d for d in self.clusters if d.significant and d.test.diff > 0]

    @property
    def shrank(self) -> list[ClusterDelta]:
        return [d for d in self.clusters if d.significant and d.test.diff < 0]

    def render(self) -> str:
        lines = [f"{self.label_a} n={self.n_a}   {self.label_b} n={self.n_b}   "
                 f"BH alpha={self.alpha}"]
        lines.append("")
        lines.append("clusters")
        for delta in self.clusters:
            lines.append("  " + delta.line())
        lines.append("")
        lines.append("pathologies")
        for delta in self.pathologies:
            lines.append("  " + delta.line())
        if self.grew:
            lines.append("")
            lines.append("regressions: " + ", ".join(
                f"{d.key} ({d.name[:40]}) x{d.test.risk_ratio:.2f}" for d in self.grew))
        else:
            lines.append("")
            lines.append("regressions: none at this alpha")
        return "\n".join(lines)


def _split(members: list[str], group_a: set[str]) -> tuple[int, int]:
    a = sum(1 for m in members if m in group_a)
    return a, len(members) - a


def compare_populations(clustering: Clustering, group_a: set[str], group_b: set[str], *,
                        all_facts: list[RunFacts] | None = None, alpha: float = 0.05,
                        n_permutations: int = 20000, seed: int = 0,
                        label_a: str = "before", label_b: str = "after") -> RegressionReport:
    """Test every cluster, and optionally every pathology, for a change in prevalence."""
    n_a, n_b = len(group_a), len(group_b)
    report = RegressionReport(n_a=n_a, n_b=n_b, alpha=alpha, label_a=label_a, label_b=label_b)

    raw: list[tuple[str, str, ProportionTest]] = []
    for c in clustering.clusters:
        k_a, k_b = _split(c.members, group_a)
        raw.append((c.slug, c.name, compare_proportions(
            k_a, n_a, k_b, n_b, n_permutations=n_permutations, seed=seed)))
    noise_a, noise_b = _split(clustering.noise, group_a)
    if clustering.noise:
        raw.append(("noise", "unclustered", compare_proportions(
            noise_a, n_a, noise_b, n_b, n_permutations=n_permutations, seed=seed)))

    adjusted, rejected = benjamini_hochberg([t.p_value for _, _, t in raw], alpha)
    report.clusters = [
        ClusterDelta(key=key, name=name, test=test, p_adjusted=q, significant=bool(flag))
        for (key, name, test), q, flag in zip(raw, adjusted, rejected, strict=True)
    ]
    report.clusters.sort(key=lambda d: (not d.significant, -d.test.diff))

    if all_facts:
        fired: dict[str, set[str]] = {p: set() for p in PATHOLOGIES}
        for facts in all_facts:
            for finding in run_detectors(facts):
                fired[finding.pathology].add(finding.run_id)
        praw: list[tuple[str, str, ProportionTest]] = []
        for pathology in PATHOLOGIES:
            members = sorted(fired[pathology])
            k_a, k_b = _split(members, group_a)
            praw.append((pathology[:8], pathology, compare_proportions(
                k_a, n_a, k_b, n_b, n_permutations=n_permutations, seed=seed)))
        padj, prej = benjamini_hochberg([t.p_value for _, _, t in praw], alpha)
        report.pathologies = [
            ClusterDelta(key=key, name=name, test=test, p_adjusted=q, significant=bool(flag))
            for (key, name, test), q, flag in zip(praw, padj, prej, strict=True)
        ]
        report.pathologies.sort(key=lambda d: (not d.significant, -d.test.diff))

    return report
