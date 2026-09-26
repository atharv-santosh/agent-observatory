"""Shared fixtures. Everything here runs offline with no credentials."""

from __future__ import annotations

from pathlib import Path

import pytest

from observatory.corpus import Generator, RunLabel
from observatory.derive import derive
from observatory.events import Event, normalise
from observatory.pipeline import analyse

FIXTURES = Path(__file__).parent / "fixtures" / "sibling"


@pytest.fixture(scope="session")
def small_corpus() -> tuple[list[list[Event]], list[RunLabel]]:
    """800 runs: big enough for clustering to mean something, small enough for a fast suite."""
    streams, labels = Generator(seed=101, cohort="baseline").generate(800)
    return [[normalise(e) for e in s] for s in streams], labels


@pytest.fixture(scope="session")
def small_facts(small_corpus):
    return [derive(t) for t in small_corpus[0]]


@pytest.fixture(scope="session")
def analysis(small_corpus):
    trajectories, labels = small_corpus
    return analyse(trajectories, k_range=(5, 12), seed=0,
                   truth={label.run_id: label.mode for label in labels})


@pytest.fixture(scope="session")
def cohort_pair() -> tuple[list[list[Event]], dict[str, list[str]], set[str], set[str]]:
    """Two cohorts drawn from identical weights, for the regression false-positive check."""
    from observatory.corpus import BASELINE_WEIGHTS
    a_streams, a_labels = Generator(seed=201, cohort="before",
                                    weights=BASELINE_WEIGHTS).generate(300)
    b_streams, b_labels = Generator(seed=202, cohort="after",
                                    weights=BASELINE_WEIGHTS).generate(300)
    trajectories = [[normalise(e) for e in s] for s in a_streams + b_streams]
    truth = {label.run_id: label.pathologies for label in a_labels + b_labels}
    return (trajectories, truth, {label.run_id for label in a_labels},
            {label.run_id for label in b_labels})
