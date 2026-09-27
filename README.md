# agent-observatory

Observability for populations of agent runs, with a failure taxonomy derived from the data rather
than guessed in advance.

When an agent fails in production you do not have one trajectory, you have thousands, and per-run
tracing tells you what happened once. This reads recorded trajectories, builds span trees with cost
attributed per subtree, clusters the population into a failure taxonomy without hand labels, and then
measures whether that taxonomy is real.

That last part is the point. Clustering trajectories produces groups that look plausible and cannot
be checked, which is how unvalidated clustering stays unvalidated. So the repo generates 2,400
synthetic runs with ten failure modes planted on purpose, writes what it planted to a sidecar file
that never enters the event stream, and scores the clusterer against it.

## The measured numbers

Cluster purity against the planted ground truth is 0.887 at k=11, with inverse purity 0.868,
adjusted Rand 0.821 and 3.6% of runs left unclustered. No cluster fell below 50% dominant label
share. Four of the ten planted modes were recovered exactly; `arg_hallucination` is the weak point at
55.4% spread over three clusters, and `slow_tool` is not recovered at all because a run whose only
distinguishing feature is one slow tool call is structurally a healthy run.

Detector precision across the six named pathologies runs from 0.854 (oscillation, 27 false
positives in 185 firings) to 1.000 (silent truncation, premature termination, hallucinated tool
arguments, context exhaustion), with retry loops at 0.985.

The regression view, given a cohort where one failure mode was made 2.5 times more likely, found
the right cluster: the one that is 75% `retry_storm` grew from 13.80% to 24.45% of the population,
risk ratio 1.77 with a 95% interval of 1.55 to 2.04, q=0.0005 after Benjamini-Hochberg across ten
clusters. Given two cohorts drawn from identical weights it found nothing, from 20 tests, smallest
adjusted p-value 0.96.

MEASUREMENTS.md has all of it, including the ablations that show two of the modelling decisions are
worth nothing on the final corpus.

## How it works

Flat event streams become span trees with OpenTelemetry `gen_ai` attributes and ids derived from the
run id, so re-running the tool over the same corpus produces an identical trace. Every span carries
both an exclusive and an inclusive cost, because a step span spends no tokens of its own and the
inherited number is the one that answers which step to cut. OTLP export writes the body a collector
would receive to a file; posting it is optional.

The clustering is spherical k-means on cosine distance, written out rather than imported, because the
interesting parts are the ones a library hides: k comes from silhouette times subsample stability,
runs too far from every centroid go unclustered rather than being forced somewhere, and each cluster
is named from its own highest-lift centroid features. No feature encodes a pathology name. A retry
loop and an oscillation are both a periodic tool sequence, separated only by the period the data
happens to have.

The two-population comparison uses a permutation test on each cluster's prevalence, a bootstrap
interval on the risk ratio, and Benjamini-Hochberg across clusters, because asking the question of
fifteen clusters at once otherwise hands you a regression by luck.

Fixtures from three sibling projects are vendored under `tests/fixtures/sibling`, including an
adapter for the one that writes scenario results instead of an event stream.

## Running it

```
uv pip install -e ".[dev,web]"
obs gen corpus -n 2000 --with-regression
obs analyse corpus/runs --labels corpus/labels/before.json --labels corpus/labels/after.json
obs regress corpus/runs
obs serve
```

Then open http://127.0.0.1:8731 to browse runs, drill into a span tree, and read the cluster map with
representative trajectories per cluster. `obs report` writes one self-contained HTML file with no
external requests. The rest of the CLI is `clusters`, `runs`, `show`, `cost`, `export`, `validate`
and `summarise`.

Everything runs offline with no credentials, since the input is recorded trajectories. The only model
call is `obs summarise`, which narrates a cluster and falls back to a deterministic summary when none
is configured. `pytest` is 208 tests and touches no network.

NOTES.md is the build log, including the detector that ran at 0.418 precision until it was measured
and the bug that reported every completion as zero output tokens.
