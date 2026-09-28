# agent-observatory

Observability across populations of agent runs, with the failure taxonomy derived
from the data instead of guessed up front.

When an agent fails in production you have thousands of trajectories, not one, and
per run tracing only tells you what happened once. This reads recorded trajectories,
builds span trees with cost attributed per subtree, clusters the population into a
failure taxonomy with no hand labels, then measures whether that taxonomy is real.

That last part matters. Clustering produces groups that look plausible and cannot be
checked, which is how unvalidated clustering stays that way. So it generates 2,400
synthetic runs with ten failure modes planted on purpose, writes what it planted to a
sidecar the event stream never sees, and scores the clusterer against it.

## Numbers

```
cluster purity vs planted truth   0.887 at k=11
inverse purity 0.868, adjusted Rand 0.821, 3.6% unclustered
detector precision                0.854 to 1.000
```

Four of ten planted modes recovered exactly. `arg_hallucination` is the weak one at
55.4% across three clusters. `slow_tool` is not recovered at all, because a run whose
only odd feature is one slow call is structurally a healthy run. The cost view catches
that instead, 94s max against a 945ms mean.

Oscillation is the worst detector at 0.854, 27 false positives in 185 firings. Context
exhaustion sat at 0.418 until measurement forced an absolute size floor instead of a
growth ratio.

Given a cohort with one mode made 2.5x more likely, the regression view found the right
cluster growing 13.80% to 24.45%, risk ratio 1.77 [1.55, 2.04], q=0.0005 after
Benjamini-Hochberg. On two cohorts from identical weights it flagged nothing across 20
tests.

## Running it

```
uv pip install -e ".[web]"
obs gen corpus -n 2400 --seed 7
obs analyse corpus/runs --labels corpus/labels/baseline.json
obs serve
```

No credentials needed, the input is recorded trajectories. It reads the event schema
the sibling projects write, so their journals and cassettes go straight in.
