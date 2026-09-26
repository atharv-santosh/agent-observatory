"""Spherical k-means over trajectory vectors, with k chosen by reproducibility and names from centroids.

Written out rather than imported because the interesting parts are the parts a library would hide:
which distance, how k is chosen, what happens to points that belong to nothing, and how a cluster
earns a name. All four are decisions about the failure taxonomy, not implementation details.

The distance is cosine, which is why every vector arrives L2-normalised: an assignment is then a
single matrix multiply and an argmax, and a centroid is a normalised mean. Points too far from every
centroid become noise rather than being forced into the nearest cluster, because a taxonomy that
claims to explain a run it does not explain is worse than one that admits the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MIN_SIM = 1e-12


@dataclass(slots=True)
class Cluster:
    cid: int
    slug: str
    name: str
    members: list[str]
    centroid: np.ndarray
    top_features: list[tuple[str, float]] = field(default_factory=list)
    cohesion: float = 0.0
    representatives: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(slots=True)
class Clustering:
    """The taxonomy: a label per run, a named cluster per label, and the runs nothing explains."""

    run_ids: list[str]
    labels: np.ndarray
    clusters: list[Cluster]
    k: int
    silhouette: float
    inertia: float
    coarse_k: int = 0
    noise: list[str] = field(default_factory=list)
    sweep: list[tuple[int, float]] = field(default_factory=list)
    stability_sweep: list[tuple[int, float]] = field(default_factory=list)
    merge_tree: list[tuple[int, int, float]] = field(default_factory=list)

    def label_of(self) -> dict[str, int]:
        return dict(zip(self.run_ids, (int(v) for v in self.labels), strict=True))

    def cluster_of(self) -> dict[str, Cluster | None]:
        by_cid = {c.cid: c for c in self.clusters}
        return {rid: by_cid.get(int(lab)) for rid, lab in zip(self.run_ids, self.labels,
                                                              strict=True)}

    @property
    def noise_fraction(self) -> float:
        return len(self.noise) / max(len(self.run_ids), 1)


def _kmeanspp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding in cosine distance."""
    n = x.shape[0]
    first = int(rng.integers(n))
    centers = [first]
    closest = 1.0 - x @ x[first]
    np.maximum(closest, 0.0, out=closest)
    for _ in range(1, k):
        total = float(closest.sum())
        if total <= _MIN_SIM:
            pick = int(rng.integers(n))
        else:
            pick = int(rng.choice(n, p=closest / total))
        centers.append(pick)
        d = 1.0 - x @ x[pick]
        np.maximum(d, 0.0, out=d)
        closest = np.minimum(closest, d)
    return x[centers].copy()


def _normalise_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms < _MIN_SIM] = 1.0
    return m / norms


def spherical_kmeans(x: np.ndarray, k: int, *, n_init: int = 8, max_iter: int = 100,
                     seed: int = 0, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, centroids, inertia) for the best of n_init restarts."""
    if k < 1 or k > x.shape[0]:
        raise ValueError(f"k={k} out of range for {x.shape[0]} points")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for init in range(n_init):
        rng = np.random.default_rng(seed * 1000 + init)
        centroids = _kmeanspp(x, k, rng)
        labels = np.zeros(x.shape[0], dtype=np.int32)
        prev = np.inf
        for _ in range(max_iter):
            sims = x @ centroids.T
            labels = np.asarray(sims.argmax(axis=1), dtype=np.int32)
            inertia = float((1.0 - sims.max(axis=1)).sum())
            new = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new[c] = x[mask].sum(axis=0)
                else:
                    # An empty cluster is reseeded onto the worst-explained point, which is what
                    # keeps k honest: if the data cannot support k groups, the reseeds thrash and
                    # the silhouette for that k drops.
                    worst = int((1.0 - sims.max(axis=1)).argmax())
                    new[c] = x[worst]
            centroids = _normalise_rows(new)
            if prev - inertia <= tol:
                break
            prev = inertia
        sims = x @ centroids.T
        labels = np.asarray(sims.argmax(axis=1), dtype=np.int32)
        inertia = float((1.0 - sims.max(axis=1)).sum())
        if best is None or inertia < best[2]:
            best = (labels, centroids, inertia)
    assert best is not None
    return best


def silhouette(x: np.ndarray, labels: np.ndarray, *, max_points: int = 1500,
               seed: int = 0) -> float:
    """Mean cosine silhouette, subsampled deterministically above max_points."""
    n = x.shape[0]
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, max_points, replace=False)
        idx.sort()
        x, labels = x[idx], labels[idx]
        n = max_points
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 0.0
    dist = np.clip(1.0 - x @ x.T, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    scores = np.zeros(n)
    for i in range(n):
        own = labels == labels[i]
        n_own = int(own.sum())
        a = dist[i, own].sum() / (n_own - 1) if n_own > 1 else 0.0
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            mask = labels == c
            if mask.any():
                b = min(b, float(dist[i, mask].mean()))
        scores[i] = 0.0 if n_own <= 1 or not np.isfinite(b) else (b - a) / max(a, b)
    return float(scores.mean())


def _mark_noise(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                noise_z: float) -> np.ndarray:
    """Demote points whose distance to their own centroid is a robust outlier for that cluster."""
    labels = labels.copy()
    for c in range(centroids.shape[0]):
        mask = labels == c
        if int(mask.sum()) < 8:
            continue
        d = np.clip(1.0 - x[mask] @ centroids[c], 0.0, 2.0)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        if mad < 1e-9:
            continue
        cutoff = med + noise_z * 1.4826 * mad
        idx = np.flatnonzero(mask)
        labels[idx[d > cutoff]] = -1
    return labels


def _merge_tree(centroids: np.ndarray) -> list[tuple[int, int, float]]:
    """Average-linkage merges over the centroids, giving the cluster map a hierarchy.

    Run over k centroids rather than n points, so it stays trivially cheap and answers a different
    question: which named failure modes are near-neighbours of each other.
    """
    k = centroids.shape[0]
    if k < 2:
        return []
    dist = np.clip(1.0 - centroids @ centroids.T, 0.0, 2.0)
    np.fill_diagonal(dist, np.inf)
    groups = {i: [i] for i in range(k)}
    merges: list[tuple[int, int, float]] = []
    active = set(range(k))
    while len(active) > 1:
        best = (np.inf, -1, -1)
        for a in sorted(active):
            for b in sorted(active):
                if b <= a:
                    continue
                d = float(np.mean([dist[i, j] for i in groups[a] for j in groups[b]]))
                if d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        merges.append((a, b, best[0]))
        groups[a] = groups[a] + groups[b]
        active.discard(b)
    return merges


def subsample_stability(x: np.ndarray, k: int, *, reps: int = 3, fraction: float = 0.8,
                        seed: int = 0) -> float:
    """Mean agreement between clusterings of two overlapping subsamples at this k.

    Silhouette alone is a poor way to choose k here: it is a geometric compactness score, it is
    biased towards small k on cosine data, and across four corpus seeds it picked anything from 5 to
    7 for the same generator. Stability asks the question that actually matters. Two 80% subsamples
    share about two thirds of the corpus; if k groups are really there, the shared runs are grouped
    the same way in both, and if k is too large the surplus clusters are cut differently each time.
    """
    n = x.shape[0]
    size = max(k * 4, int(n * fraction))
    if size >= n:
        return 0.0
    rng = np.random.default_rng(seed + 9001)
    scores: list[float] = []
    for rep in range(reps):
        idx_a = rng.choice(n, size, replace=False)
        idx_b = rng.choice(n, size, replace=False)
        shared = np.intersect1d(idx_a, idx_b)
        if shared.size < k * 2:
            continue
        pos_a = {int(v): i for i, v in enumerate(idx_a)}
        pos_b = {int(v): i for i, v in enumerate(idx_b)}
        lab_a, _, _ = spherical_kmeans(x[idx_a], k, n_init=3, seed=seed * 31 + rep)
        lab_b, _, _ = spherical_kmeans(x[idx_b], k, n_init=3, seed=seed * 31 + rep + 500)
        left = [int(lab_a[pos_a[int(v)]]) for v in shared]
        right = [int(lab_b[pos_b[int(v)]]) for v in shared]
        scores.append(_pair_agreement(left, right))
    return float(np.mean(scores)) if scores else 0.0


def _pair_agreement(a: list[int], b: list[int]) -> float:
    """Adjusted Rand index between two labellings of the same points."""
    from collections import Counter
    n = len(a)
    if n < 2:
        return 0.0
    joint = Counter(zip(a, b, strict=True))
    row, col = Counter(a), Counter(b)
    comb = lambda v: v * (v - 1) / 2  # noqa: E731
    index = sum(comb(v) for v in joint.values())
    er, ec = sum(comb(v) for v in row.values()), sum(comb(v) for v in col.values())
    expected = er * ec / comb(n)
    maximum = (er + ec) / 2
    return (index - expected) / (maximum - expected) if maximum != expected else 0.0


def _refine(x: np.ndarray, idx: np.ndarray, depth: int, *, max_depth: int, min_size: int,
            min_gain: float, seed: int, out: list[np.ndarray]) -> None:
    """Split a cluster in two while the split is clean enough to be worth having.

    The flat partition and the taxonomy people want are not the same object. Left to itself the
    criterion picks a coarse k, because a large tight healthy population is geometrically the best
    thing in the data and silhouette rewards it, and the rare failure modes get absorbed. Splitting
    only where there is clear internal substructure resolves them without dragging the whole
    taxonomy finer: on a 3,600-run corpus this raises purity from 0.750 to 0.846 while inverse
    purity only falls from 0.984 to 0.953, where forcing flat k up to 9 to get similar purity drops
    inverse purity to 0.794.
    """
    if depth >= max_depth or idx.size < min_size * 2:
        out.append(idx)
        return
    sub = x[idx]
    labels, _, _ = spherical_kmeans(sub, 2, n_init=6, seed=seed + depth * 7 + 1)
    left, right = idx[labels == 0], idx[labels == 1]
    if left.size < min_size or right.size < min_size:
        out.append(idx)
        return
    if silhouette(sub, labels, max_points=900, seed=seed) < min_gain:
        out.append(idx)
        return
    _refine(x, left, depth + 1, max_depth=max_depth, min_size=min_size, min_gain=min_gain,
            seed=seed, out=out)
    _refine(x, right, depth + 1, max_depth=max_depth, min_size=min_size, min_gain=min_gain,
            seed=seed, out=out)


def cluster(x: np.ndarray, run_ids: list[str], column_names: list[str], *,
            k: int | None = None, k_range: tuple[int, int] = (4, 18), n_init: int = 8,
            seed: int = 0, noise_z: float = 3.5, top_features: int = 6,
            select: str = "product", stability_reps: int = 6,
            score_silhouette: bool = True, refine: bool = True, refine_max_depth: int = 3,
            refine_min_size: int = 30, refine_min_gain: float = 0.30) -> Clustering:
    """Fit the taxonomy: pick k, cluster, refine, mark noise, name every cluster."""
    if x.shape[0] != len(run_ids):
        raise ValueError("vector count does not match run id count")
    sweep: list[tuple[int, float]] = []
    if k is None:
        lo, hi = k_range
        hi = min(hi, max(lo, x.shape[0] // 4))
        rows: list[tuple[int, float, float]] = []
        for candidate in range(lo, hi + 1):
            labels, _, _ = spherical_kmeans(x, candidate, n_init=max(3, n_init // 2), seed=seed)
            sil = silhouette(x, labels, seed=seed)
            stab = (subsample_stability(x, candidate, reps=stability_reps, seed=seed)
                    if select != "silhouette" else 0.0)
            rows.append((candidate, sil, stab))
        # Silhouette times stability: a k has to produce compact clusters and produce the same ones
        # again on a different subsample. Measured against four corpus seeds whose ground truth is
        # known, this picks a k with mean ARI 0.812 where silhouette alone gets 0.795 and stability
        # alone 0.716 (the oracle choice is 0.852). Silhouette alone slides to k=5 and merges the
        # rare modes; stability alone rewards coarse taxonomies for the same reason.
        if select == "silhouette":
            k = max(rows, key=lambda r: r[1])[0]
        elif select == "stability":
            k = max(rows, key=lambda r: r[2])[0]
        else:
            k = max(rows, key=lambda r: r[1] * r[2])[0]
        sweep = [(c, s) for c, s, _ in rows]
        stability_sweep = [(c, st) for c, _, st in rows]
    else:
        stability_sweep = []

    labels, centroids, inertia = spherical_kmeans(x, k, n_init=n_init, seed=seed)
    coarse_k = k
    if refine:
        leaves: list[np.ndarray] = []
        for cid in range(k):
            _refine(x, np.flatnonzero(labels == cid), 0, max_depth=refine_max_depth,
                    min_size=refine_min_size, min_gain=refine_min_gain, seed=seed, out=leaves)
        leaves = [leaf for leaf in leaves if leaf.size]
        labels = np.full(x.shape[0], -1, dtype=np.int32)
        for i, leaf in enumerate(leaves):
            labels[leaf] = i
        k = len(leaves)
        centroids = _normalise_rows(np.vstack([x[leaf].sum(axis=0) for leaf in leaves]))
    labels = _mark_noise(x, labels, centroids, noise_z)
    score = (silhouette(x[labels >= 0], labels[labels >= 0], seed=seed)
             if score_silhouette and (labels >= 0).any() else 0.0)

    global_mean = x.mean(axis=0)
    clusters: list[Cluster] = []
    for cid in range(k):
        mask = labels == cid
        members = [run_ids[i] for i in np.flatnonzero(mask)]
        if not members:
            continue
        centroid = _normalise_rows(x[mask].sum(axis=0)[None, :])[0]
        lift = centroid - global_mean
        order = np.argsort(-lift)[:top_features]
        feats = [(column_names[i], float(lift[i])) for i in order if lift[i] > 1e-6]
        sims = x[mask] @ centroid
        idx = np.flatnonzero(mask)[np.argsort(-sims)]
        clusters.append(Cluster(
            cid=cid,
            slug=f"c{cid:02d}",
            name=name_cluster(feats),
            members=members,
            centroid=centroid,
            top_features=feats,
            cohesion=float(sims.mean()),
            representatives=[run_ids[i] for i in idx[:5]],
        ))

    clusters.sort(key=lambda c: -c.size)
    remap = {c.cid: i for i, c in enumerate(clusters)}
    relabelled = np.full_like(labels, -1)
    for old, new in remap.items():
        relabelled[labels == old] = new
    for i, c in enumerate(clusters):
        c.cid, c.slug = i, f"c{i:02d}"

    stacked = np.vstack([c.centroid for c in clusters]) if clusters else np.zeros((0, x.shape[1]))
    return Clustering(
        run_ids=run_ids,
        labels=relabelled,
        clusters=clusters,
        k=len(clusters),
        coarse_k=coarse_k,
        silhouette=score,
        inertia=inertia,
        noise=[run_ids[i] for i in np.flatnonzero(relabelled < 0)],
        sweep=sweep,
        stability_sweep=stability_sweep,
        merge_tree=_merge_tree(stacked),
    )


_BUCKET_WORDS = ("lowest", "low", "mid", "high", "highest")


def _readable(column: str) -> str | None:
    """Translate one feature column name into a fragment fit for a cluster name."""
    block, _, name = column.partition("/")
    if name.startswith("t3:"):
        return name[3:].replace(">", "->")
    if name.startswith("t2:"):
        return name[3:].replace(">", "->")
    if name.startswith("t:"):
        return name[2:]
    if name.startswith("tf:"):
        return f"{name[3:]}-fails"
    if name.startswith("e:"):
        words = name[2:].split()
        return "err(" + " ".join(words[:3]) + ")"
    if name.startswith("term:"):
        return f"ends-{name[5:]}"
    if name.startswith("gate:"):
        return f"gate-{name[5:]}"
    if name.startswith("fin:"):
        return f"finish-{name[4:]}"
    if name.startswith("d:"):
        return f"decides-{name[2:]}"
    if name.startswith("f:"):
        return name[2:].replace("_", "-")
    if name.startswith("s:"):
        body = name[2:]
        metric, _, bucket = body.rpartition(":b")
        try:
            b = int(bucket)
        except ValueError:
            return metric
        word = _BUCKET_WORDS[min(b, len(_BUCKET_WORDS) - 1)]
        if word == "mid":
            return None
        return f"{metric.replace('_', '-')}-{word}"
    del block
    return name


def name_cluster(top_features: list[tuple[str, float]], *, parts: int = 3) -> str:
    """Build a cluster name out of its own highest-lift centroid features."""
    seen: list[str] = []
    for column, _weight in top_features:
        frag = _readable(column)
        if frag and frag not in seen:
            seen.append(frag)
        if len(seen) >= parts:
            break
    return " / ".join(seen) if seen else "unnamed"


def project_2d(centroids_or_points: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Classical MDS on cosine distance, for the cluster map. Power iteration, no dependencies."""
    x = centroids_or_points
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    if n == 1:
        return np.zeros((1, 2))
    d2 = np.clip(1.0 - x @ x.T, 0.0, 2.0) ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    rng = np.random.default_rng(seed)
    coords = np.zeros((n, 2))
    for axis in range(2):
        v = rng.normal(size=n)
        v /= np.linalg.norm(v) or 1.0
        val = 0.0
        for _ in range(300):
            w = b @ v
            val = float(np.linalg.norm(w))
            if val < 1e-12:
                break
            w /= val
            if np.linalg.norm(w - v) < 1e-10:
                v = w
                break
            v = w
        coords[:, axis] = v * np.sqrt(max(val, 0.0))
        b = b - val * np.outer(v, v)
    return coords
