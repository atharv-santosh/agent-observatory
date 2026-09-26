"""Featurisation, clustering, naming, and recovery of the planted ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from observatory.cluster import (
    cluster,
    name_cluster,
    project_2d,
    silhouette,
    spherical_kmeans,
    subsample_stability,
)
from observatory.features import (
    DEFAULT_BLOCK_WEIGHTS,
    SCALAR_BLOCKS,
    Vectorizer,
    _alternation_span,
    _cycle_signature,
    _max_consecutive_repeat,
    featurize,
)
from observatory.validate import (
    adjusted_rand,
    evaluate_clustering,
    inverse_purity,
    nmi,
    purity,
)

# --- the loop signature, which is the load-bearing feature -----------------------------------

@pytest.mark.parametrize(("seq", "period", "repeats"), [
    (["a", "a", "a", "a"], 1, 4),
    (["a", "b", "a", "b", "a", "b"], 2, 3),
    (["a", "b", "c", "a", "b", "c"], 3, 2),
    (["a", "b", "c", "d"], 0, 0),
    ([], 0, 0),
])
def test_cycle_signature_finds_the_period_the_data_has(seq, period, repeats):
    """A retry loop is period 1 and an oscillation is period 2; the feature names neither."""
    got_period, got_repeats, _ = _cycle_signature(seq)
    assert (got_period, got_repeats) == (period, repeats)


def test_cycle_signature_prefers_more_repeats_then_the_shorter_period():
    period, repeats, _ = _cycle_signature(["x", "a", "a", "a", "a", "a", "y"])
    assert (period, repeats) == (1, 5)


def test_max_consecutive_repeat_and_alternation_span():
    assert _max_consecutive_repeat(["a", "a", "b", "a", "a", "a"]) == 3
    assert _max_consecutive_repeat([]) == 0
    assert _alternation_span(["a", "b", "a", "b", "a"]) == 5
    assert _alternation_span(["a", "b", "c", "d"]) == 0


# --- featurisation ---------------------------------------------------------------------------

def test_no_hand_authored_feature_encodes_a_pathology_name():
    """The feature schema must not smuggle in the taxonomy it is supposed to discover.

    Only the hand-authored names are checked. Data-derived columns (tool names, error shapes,
    decision values) legitimately carry whatever words the corpus contains, and one of this
    corpus's decision values happens to be "retry"; that is the agent's vocabulary, not the
    featuriser's.
    """
    from observatory.detectors import PATHOLOGIES
    raw = featurize(next(iter([])) if False else _minimal_facts())
    authored = list(SCALAR_BLOCKS) + [name for _, name in raw.flags] + list(DEFAULT_BLOCK_WEIGHTS)
    joined = " ".join(authored).lower()
    for pathology in PATHOLOGIES:
        assert pathology not in joined
    for word in ("retry", "oscillat", "hallucinat", "premature", "exhaust", "loop_detect"):
        assert word not in joined, f"an authored feature name mentions {word!r}"


def _minimal_facts():
    from observatory.derive import derive
    from observatory.events import normalise
    return derive([normalise({"run_id": "x", "seq": 0, "ts": 0.0, "type": "run_start",
                              "payload": {}})])


def test_ground_truth_never_reaches_the_features(small_corpus, small_facts):
    """The corpus writes labels to a sidecar; nothing in the event stream carries the mode."""
    modes = {label.mode for label in small_corpus[1]}
    for facts in small_facts:
        assert "ground_truth" not in facts.labels
        assert "mode" not in facts.labels
        assert not (modes & {str(v) for v in facts.labels.values()})


def test_every_block_has_columns_and_a_weight(small_facts):
    v = Vectorizer(min_df=2).fit([featurize(f) for f in small_facts])
    blocks = {b for b, _ in v.columns}
    assert blocks <= set(DEFAULT_BLOCK_WEIGHTS)
    assert set(SCALAR_BLOCKS.values()) <= blocks
    for block in blocks:
        assert DEFAULT_BLOCK_WEIGHTS[block] > 0


def test_vectors_are_unit_length_and_blocks_are_separately_normalised(small_facts):
    v = Vectorizer(min_df=2)
    x = v.fit_transform([featurize(f) for f in small_facts])
    assert x.shape == (len(small_facts), len(v.columns))
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 1.0)
    # No single block can exceed its own weight after normalisation.
    for block, weight in DEFAULT_BLOCK_WEIGHTS.items():
        cols = [i for i, (b, _) in enumerate(v.columns) if b == block]
        if cols:
            assert np.max(np.linalg.norm(x[:, cols], axis=1)) <= weight + 1e-9


def test_scalar_encoding_is_cumulative_so_buckets_stay_ordered(small_facts):
    """One-hot buckets make 'two repeats' as far from 'one' as from 'seven'. Cumulative does not."""
    v = Vectorizer(min_df=2, n_buckets=5).fit([featurize(f) for f in small_facts])
    raw = featurize(small_facts[0])
    edges = v.edges["cycle_repeats"]
    assert edges, "the fitted corpus should give this scalar at least one bucket edge"
    cols = [i for i, (_, n) in enumerate(v.columns) if n.startswith("s:cycle_repeats:")]

    def lit(value: float) -> int:
        clone = type(raw)(run_id="x", counts={}, scalars={**raw.scalars,
                                                         "cycle_repeats": value}, flags={})
        return int((v.transform([clone])[0][cols] > 0).sum())

    probes = [edges[0] - 1] + [e + 1e-9 for e in edges] + [max(edges) + 1e6]
    counts = [lit(p) for p in probes]
    assert counts == sorted(counts), counts
    assert counts[-1] > counts[0], "a larger scalar must light strictly more columns"
    # And the distance between adjacent buckets must be smaller than between distant ones.
    assert counts[1] - counts[0] <= counts[-1] - counts[0]


def test_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="before fit"):
        Vectorizer().transform([])


def test_fit_on_empty_corpus_is_an_error():
    with pytest.raises(ValueError, match="empty corpus"):
        Vectorizer().fit([])


# --- k-means ---------------------------------------------------------------------------------

def _blobs(seed: int = 0, per: int = 60, dim: int = 24, k: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    rows = []
    for c in centres:
        pts = c + rng.normal(scale=0.12, size=(per, dim))
        rows.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
    return np.vstack(rows)


def test_spherical_kmeans_recovers_well_separated_blobs():
    x = _blobs()
    labels, centroids, inertia = spherical_kmeans(x, 4, n_init=8, seed=0)
    assert centroids.shape == (4, x.shape[1])
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)
    truth = [f"b{i // 60}" for i in range(x.shape[0])]
    assert adjusted_rand(truth, list(labels)) > 0.95
    assert inertia >= 0


def test_spherical_kmeans_is_deterministic_for_a_seed():
    x = _blobs(seed=3)
    a, _, _ = spherical_kmeans(x, 4, seed=7)
    b, _, _ = spherical_kmeans(x, 4, seed=7)
    assert np.array_equal(a, b)


def test_spherical_kmeans_rejects_impossible_k():
    with pytest.raises(ValueError, match="out of range"):
        spherical_kmeans(_blobs(per=3, k=1), 99, seed=0)


def test_silhouette_is_high_for_separated_and_low_for_random():
    x = _blobs()
    good, _, _ = spherical_kmeans(x, 4, seed=0)
    rng = np.random.default_rng(1)
    bad = rng.integers(0, 4, size=x.shape[0])
    assert silhouette(x, good, seed=0) > 0.6
    assert silhouette(x, bad, seed=0) < 0.1
    assert silhouette(x, np.zeros(x.shape[0], dtype=int), seed=0) == 0.0


def test_stability_is_high_for_real_structure_and_low_for_noise():
    assert subsample_stability(_blobs(), 4, reps=4, seed=0) > 0.9
    rng = np.random.default_rng(5)
    noise = rng.normal(size=(240, 24))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    assert subsample_stability(noise, 8, reps=4, seed=0) < 0.35


def test_noise_marking_demotes_a_far_outlier():
    x = _blobs(per=40, k=2, dim=12)
    outlier = np.zeros((1, 12))
    outlier[0, 0] = 1.0
    x = np.vstack([x, outlier / np.linalg.norm(outlier)])
    ids = [f"r{i}" for i in range(x.shape[0])]
    result = cluster(x, ids, [f"c{i}" for i in range(12)], k=2, noise_z=2.5, refine=False)
    assert result.noise_fraction < 0.2


def test_noise_can_be_switched_off():
    x = _blobs()
    ids = [f"r{i}" for i in range(x.shape[0])]
    result = cluster(x, ids, [f"c{i}" for i in range(x.shape[1])], k=4, noise_z=1e9,
                     refine=False)
    assert result.noise == []
    assert (result.labels >= 0).all()


def test_project_2d_degenerate_inputs():
    assert project_2d(np.zeros((0, 5))).shape == (0, 2)
    assert project_2d(np.ones((1, 5))).shape == (1, 2)
    coords = project_2d(_blobs(per=1, k=5, dim=8))
    assert coords.shape == (5, 2) and np.isfinite(coords).all()


# --- naming ----------------------------------------------------------------------------------

@pytest.mark.parametrize(("column", "fragment"), [
    ("tool/t:search_docs", "search_docs"),
    ("tool/t2:read_file>write_file", "read_file->write_file"),
    ("tool/tf:charge", "charge-fails"),
    ("term/term:failed", "ends-failed"),
    ("term/f:no_run_end", "no-run-end"),
    ("tok/fin:MAX_TOKENS", "finish-MAX_TOKENS"),
    ("loop/s:cycle_repeats:b4", "cycle-repeats-highest"),
])
def test_cluster_names_are_readable_translations_of_centroid_columns(column, fragment):
    assert name_cluster([(column, 1.0)]) == fragment


def test_mid_buckets_are_skipped_because_average_is_not_a_finding():
    assert name_cluster([("loop/s:cycle_repeats:b2", 1.0)]) == "unnamed"


def test_cluster_name_joins_the_top_three_distinct_fragments():
    name = name_cluster([("term/term:failed", 3.0), ("tool/tf:charge", 2.0),
                         ("tool/t:charge", 1.5), ("tool/t:read_file", 1.0)])
    assert name == "ends-failed / charge-fails / charge"


def test_empty_feature_list_still_yields_a_name():
    assert name_cluster([]) == "unnamed"


# --- metrics ---------------------------------------------------------------------------------

def test_purity_and_ari_on_a_perfect_and_a_random_partition():
    truth = ["a"] * 10 + ["b"] * 10
    perfect = [0] * 10 + [1] * 10
    assert purity(truth, perfect) == 1.0
    assert inverse_purity(truth, perfect) == 1.0
    assert adjusted_rand(truth, perfect) == pytest.approx(1.0)
    assert nmi(truth, perfect) == pytest.approx(1.0)
    everything_one = [0] * 20
    assert purity(truth, everything_one) == 0.5
    assert inverse_purity(truth, everything_one) == 1.0
    assert adjusted_rand(truth, everything_one) == pytest.approx(0.0)


def test_purity_is_gameable_by_splitting_which_is_why_ari_is_reported_too():
    truth = ["a"] * 10 + ["b"] * 10
    singletons = list(range(20))
    assert purity(truth, singletons) == 1.0
    assert adjusted_rand(truth, singletons) < 0.05
    assert inverse_purity(truth, singletons) == pytest.approx(0.1)


def test_metrics_on_empty_input():
    assert purity([], []) == 0.0
    assert inverse_purity([], []) == 0.0
    assert adjusted_rand([], []) == 0.0
    assert nmi([], []) == 0.0


# --- recovery of the plant -------------------------------------------------------------------

def test_clustering_recovers_the_planted_modes(analysis, small_corpus):
    truth = {label.run_id: label.mode for label in small_corpus[1]}
    report = evaluate_clustering(analysis.clustering, truth)
    # Floors, not the headline. The numbers in MEASUREMENTS.md come from the full 4,000-run
    # corpus; this fixture is 800 runs so the rare modes have fewer members to cluster on.
    assert report.k >= 5
    assert report.purity >= 0.72, report.summary()
    assert report.inverse_purity >= 0.70, report.summary()
    assert report.adjusted_rand >= 0.45, report.summary()
    assert report.nmi >= 0.60, report.summary()
    assert report.noise_fraction < 0.20


def test_the_distinctive_modes_land_in_one_cluster_each(analysis, small_corpus):
    """These four have structure nothing else shares, so anything less is a real failure."""
    truth = {label.run_id: label.mode for label in small_corpus[1]}
    report = evaluate_clustering(analysis.clustering, truth)
    for mode in ("context_exhaustion", "tool_outage", "oscillation", "premature_stop"):
        assert report.mode_coverage[mode] >= 0.80, (
            f"{mode} coverage {report.mode_coverage[mode]:.2f}, "
            f"split over {report.mode_split[mode]} clusters")


def test_incoherent_clusters_are_named_rather_than_averaged_away(analysis, small_corpus):
    truth = {label.run_id: label.mode for label in small_corpus[1]}
    report = evaluate_clustering(analysis.clustering, truth)
    flagged = {q.slug for q in report.clusters if q.dominant_share < 0.5}
    assert set(report.incoherent) == flagged
    for q in report.clusters:
        assert q.incoherent == (q.dominant_share < 0.5)
        assert sum(q.label_mix.values()) == q.size


def test_every_cluster_is_named_from_its_own_centroid(analysis):
    for c in analysis.clustering.clusters:
        assert c.name and c.name != "unnamed"
        assert c.members
        assert 0.0 < c.cohesion <= 1.0
        assert c.representatives
        assert set(c.representatives) <= set(c.members)
        assert np.isclose(np.linalg.norm(c.centroid), 1.0)


def test_refinement_raises_purity_without_collapsing_inverse_purity(small_facts, small_corpus):
    truth = {label.run_id: label.mode for label in small_corpus[1]}
    raw = [featurize(f) for f in small_facts]
    ids = [f.run_id for f in small_facts]
    v = Vectorizer(min_df=3)
    x = v.fit_transform(raw)
    flat = evaluate_clustering(
        cluster(x, ids, v.column_names(), k_range=(5, 11), seed=0, refine=False), truth)
    refined = evaluate_clustering(
        cluster(x, ids, v.column_names(), k_range=(5, 11), seed=0, refine=True), truth)
    assert refined.k >= flat.k
    assert refined.purity >= flat.purity - 1e-9
    assert refined.inverse_purity >= flat.inverse_purity - 0.15


def test_labels_and_clusters_agree(analysis):
    clustering = analysis.clustering
    by_slug = {c.slug: c for c in clustering.clusters}
    for run_id, c in clustering.cluster_of().items():
        if c is None:
            assert run_id in clustering.noise
        else:
            assert run_id in by_slug[c.slug].members
    assert sum(c.size for c in clustering.clusters) + len(clustering.noise) == len(
        clustering.run_ids)


def test_merge_tree_covers_every_cluster_once(analysis):
    merges = analysis.clustering.merge_tree
    assert len(merges) == max(analysis.clustering.k - 1, 0)
    assert all(0.0 <= d <= 2.0 for _, _, d in merges)


def test_mismatched_vector_and_id_counts_is_an_error():
    with pytest.raises(ValueError, match="does not match"):
        cluster(_blobs(per=10, k=2, dim=6), ["only-one"], [f"c{i}" for i in range(6)], k=2)
