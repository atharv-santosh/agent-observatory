"""The hand-rolled statistics, and the regression view they exist for.

The false positive rate matters more here than the true positive rate. A regression view that finds
a regression between two populations drawn from the same distribution is worse than no view.
"""

from __future__ import annotations

import numpy as np
import pytest

from observatory.cluster import cluster
from observatory.corpus import BASELINE_WEIGHTS, Generator, shifted_weights
from observatory.derive import derive
from observatory.events import normalise
from observatory.features import Vectorizer, featurize
from observatory.pipeline import analyse
from observatory.regression import compare_populations
from observatory.stats import (
    benjamini_hochberg,
    bootstrap_ratio_ci,
    compare_proportions,
    permutation_proportion_test,
    wilson_interval,
)

# --- permutation test ------------------------------------------------------------------------

def test_permutation_test_detects_a_large_real_difference():
    p = permutation_proportion_test(10, 500, 120, 500, n_permutations=4000, seed=0)
    assert p < 0.001


def test_permutation_test_returns_a_uniform_p_under_the_null():
    """The property that makes it trustworthy: with no effect, p is uniform, so alpha is alpha."""
    rng = np.random.default_rng(0)
    rate, n = 0.12, 400
    ps = []
    for trial in range(120):
        k_a = int(rng.binomial(n, rate))
        k_b = int(rng.binomial(n, rate))
        ps.append(permutation_proportion_test(k_a, n, k_b, n, n_permutations=2000,
                                              seed=trial))
    below = sum(1 for p in ps if p < 0.05) / len(ps)
    assert below < 0.12, f"false positive rate {below:.3f} at alpha 0.05"
    assert 0.35 < float(np.mean(ps)) < 0.65


def test_permutation_test_degenerate_inputs():
    assert permutation_proportion_test(0, 0, 0, 0) == 1.0
    assert permutation_proportion_test(0, 50, 0, 50) == 1.0
    assert permutation_proportion_test(50, 50, 50, 50) == 1.0


def test_permutation_test_is_deterministic_for_a_seed():
    a = permutation_proportion_test(5, 200, 20, 200, n_permutations=3000, seed=4)
    b = permutation_proportion_test(5, 200, 20, 200, n_permutations=3000, seed=4)
    assert a == b


def test_one_sided_p_is_smaller_for_an_increase():
    kwargs = {"n_permutations": 4000, "seed": 1}
    two = permutation_proportion_test(20, 300, 40, 300, two_sided=True, **kwargs)
    one = permutation_proportion_test(20, 300, 40, 300, two_sided=False, **kwargs)
    assert one < two


# --- effect size -----------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    (rr_lo, rr_hi), (d_lo, d_hi) = bootstrap_ratio_ci(20, 400, 60, 400, n_boot=4000, seed=0)
    assert rr_lo < 3.0 < rr_hi
    assert d_lo < 0.10 < d_hi


def test_bootstrap_ci_is_finite_with_a_zero_baseline():
    (rr_lo, rr_hi), _ = bootstrap_ratio_ci(0, 300, 15, 300, n_boot=2000, seed=0)
    assert np.isfinite(rr_lo) and np.isfinite(rr_hi)


def test_compare_proportions_reports_rates_ratio_and_p():
    result = compare_proportions(10, 400, 50, 400, n_permutations=4000, seed=0)
    assert result.rate_a == pytest.approx(0.025)
    assert result.rate_b == pytest.approx(0.125)
    assert result.risk_ratio > 3.0
    assert result.diff == pytest.approx(0.1)
    assert result.p_value < 0.001


def test_wilson_interval_is_honest_about_small_samples():
    lo_small, hi_small = wilson_interval(3, 3)
    lo_big, hi_big = wilson_interval(300, 300)
    assert lo_small < 0.5 < lo_big, "three out of three is not 300 out of 300 worth of evidence"
    assert hi_small == pytest.approx(1.0, abs=1e-3)
    assert hi_big == pytest.approx(1.0, abs=1e-5)
    assert wilson_interval(0, 0) == (0.0, 1.0)


# --- multiple comparisons --------------------------------------------------------------------

def test_benjamini_hochberg_is_monotone_and_never_shrinks_a_p_value():
    ps = [0.001, 0.008, 0.02, 0.3, 0.7, 0.9]
    adjusted, rejected = benjamini_hochberg(ps, 0.05)
    assert all(a >= p - 1e-12 for a, p in zip(adjusted, ps, strict=True))
    assert adjusted == sorted(adjusted)
    assert rejected[0] and not rejected[-1]


def test_benjamini_hochberg_preserves_input_order():
    adjusted, rejected = benjamini_hochberg([0.9, 0.001, 0.4], 0.05)
    assert rejected == [False, True, False]
    assert adjusted[1] < adjusted[2] < adjusted[0]


def test_benjamini_hochberg_on_empty_input():
    assert benjamini_hochberg([], 0.05) == ([], [])


def test_benjamini_hochberg_controls_the_false_discovery_rate():
    """Fifteen null tests at a time: uncorrected flags something, corrected mostly does not."""
    rng = np.random.default_rng(3)
    uncorrected = corrected = 0
    for _ in range(150):
        ps = list(rng.uniform(size=15))
        uncorrected += any(p < 0.05 for p in ps)
        corrected += any(benjamini_hochberg(ps, 0.05)[1])
    assert uncorrected > 75, "uncorrected testing should flag something most rounds"
    assert corrected < 15, f"BH flagged {corrected}/150 rounds of pure noise"


# --- the regression view ---------------------------------------------------------------------

def _analyse(streams):
    return analyse([[normalise(e) for e in s] for s in streams], k_range=(5, 10), seed=0)


def test_regression_view_finds_a_planted_shift(cohort_pair):
    """One mode made 2.5 times more common should show up, with a ratio in the right ballpark."""
    before_streams, before_labels = Generator(seed=301, cohort="before").generate(700)
    after_streams, _ = Generator(seed=302, cohort="after",
                                 weights=shifted_weights(mode="retry_storm",
                                                          multiplier=3.0)).generate(700)
    analysis = _analyse(before_streams + after_streams)
    report = compare_populations(analysis.clustering, analysis.cohort_members("before"),
                                 analysis.cohort_members("after"),
                                 all_facts=list(analysis.facts.values()),
                                 n_permutations=6000, seed=0)
    assert report.n_a == 700 and report.n_b == 700
    assert report.grew, report.render()
    # The retry_loop detector is the direct read on the planted change.
    retry = next(d for d in report.pathologies if d.name == "retry_loop")
    assert retry.significant and retry.test.diff > 0
    assert 1.3 < retry.test.risk_ratio < 3.0
    assert retry.test.rr_ci[0] > 1.0
    del before_labels, cohort_pair


def test_regression_view_finds_nothing_between_identical_populations(cohort_pair):
    """The number that matters. Two draws from the same weights must produce no regression."""
    trajectories, _, group_a, group_b = cohort_pair
    analysis = analyse(trajectories, k_range=(5, 10), seed=0)
    report = compare_populations(analysis.clustering, group_a, group_b,
                                 all_facts=list(analysis.facts.values()),
                                 n_permutations=6000, seed=0)
    assert report.grew == [], report.render()
    assert report.shrank == [], report.render()
    assert not any(d.significant for d in report.pathologies), report.render()


def test_clusters_are_fit_on_the_pooled_population_so_they_mean_the_same_thing(cohort_pair):
    trajectories, _, group_a, group_b = cohort_pair
    analysis = analyse(trajectories, k_range=(5, 10), seed=0)
    for c in analysis.clustering.clusters:
        assert set(c.members) <= (group_a | group_b)
    covered = set().union(*(set(c.members) for c in analysis.clustering.clusters))
    assert covered | set(analysis.clustering.noise) == group_a | group_b


def test_regression_report_renders_and_every_row_is_accounted_for(cohort_pair):
    trajectories, _, group_a, group_b = cohort_pair
    analysis = analyse(trajectories, k_range=(5, 10), seed=0)
    report = compare_populations(analysis.clustering, group_a, group_b,
                                 all_facts=list(analysis.facts.values()),
                                 n_permutations=2000, seed=0)
    text = report.render()
    assert "clusters" in text and "pathologies" in text and "regressions" in text
    for d in report.clusters:
        assert d.test.n_a == len(group_a) and d.test.n_b == len(group_b)
        assert d.p_adjusted >= d.test.p_value - 1e-12
        assert d.direction in {"grew", "shrank", "flat"}
    keys = [d.key for d in report.clusters]
    assert len(keys) == len(set(keys))


def test_unclustered_runs_get_their_own_regression_row(cohort_pair):
    trajectories, _, group_a, group_b = cohort_pair
    analysis = analyse(trajectories, k_range=(5, 10), seed=0)
    report = compare_populations(analysis.clustering, group_a, group_b, n_permutations=1000,
                                 seed=0)
    if analysis.clustering.noise:
        assert any(d.key == "noise" for d in report.clusters)


def test_shifted_weights_renormalises_and_rejects_unknown_modes():
    weights = shifted_weights(BASELINE_WEIGHTS, mode="oscillation", multiplier=3.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["oscillation"] > BASELINE_WEIGHTS["oscillation"]
    assert weights["healthy"] < BASELINE_WEIGHTS["healthy"]
    with pytest.raises(KeyError):
        shifted_weights(mode="not_a_mode")


def test_cluster_prevalence_split_matches_the_raw_counts(cohort_pair):
    trajectories, _, group_a, group_b = cohort_pair
    facts = [derive(t) for t in trajectories]
    raw = [featurize(f) for f in facts]
    ids = [f.run_id for f in facts]
    v = Vectorizer(min_df=3)
    clustering = cluster(v.fit_transform(raw), ids, v.column_names(), k=6, seed=0)
    report = compare_populations(clustering, group_a, group_b, n_permutations=500, seed=0)
    by_key = {c.slug: c for c in clustering.clusters}
    for d in report.clusters:
        if d.key == "noise":
            continue
        members = set(by_key[d.key].members)
        assert d.test.k_a == len(members & group_a)
        assert d.test.k_b == len(members & group_b)
