# Measurements

Every number below came out of running the code in this repo. Nothing is estimated. Reproduce the
whole file with `python demo/measure.py`, which writes `demo/measurements.json` alongside the
console output quoted here.

**Environment**: macOS 26.4.1 (arm64), Python 3.11.9, numpy 2.4.6. Measured 2026-09-26.

## Cluster purity against planted ground truth

The corpus generator plants ten failure modes and writes what it planted to a sidecar file that
never enters the event stream, so no feature can read it. The clusterer sees only the trajectories.
2,400 runs, 52,809 events, 48,090 spans, 1,012 feature columns.

```
$ obs gen corpus -n 2400 --seed 7
$ obs analyse corpus/runs --labels corpus/labels/baseline.json --stability-reps 8

k=11  purity=0.887  inv_purity=0.868  ARI=0.821  NMI=0.845  silhouette=0.278  noise=0.036
      stability_ARI=0.715
```

| metric | value |
| --- | --- |
| purity | **0.887** |
| inverse purity | 0.868 |
| adjusted Rand index | 0.821 |
| normalised mutual information | 0.845 |
| mean cosine silhouette | 0.278 |
| subsample stability (ARI, 8 reps) | 0.715 |
| runs left unclustered | 3.6% |
| clusters below 50% dominant label | **0 of 11** |

Purity on its own is gameable: split into 2,400 singleton clusters and it is 1.000. That is why
inverse purity and ARI are reported beside it, and why the k sweep below shows purity rising and
inverse purity falling together as k grows. A purity figure without its k means nothing.

### The eleven clusters

Names are generated from each cluster's own highest-lift centroid features. The dominant label
column is the hand-planted mode, used only for scoring, never for fitting.

| cluster | n | cohesion | dominant label | share | name |
| --- | --- | --- | --- | --- | --- |
| c00 | 925 | 0.93 | healthy | 0.92 | max-prompt-tokens-high / prompt-growth-ratio-high / ends-completed |
| c01 | 258 | 0.92 | retry_storm | 0.75 | finish-STOP / ends-failed / plan-completion-highest |
| c02 | 176 | 0.90 | arg_hallucination | 0.64 | ends-failed / finish-STOP / fail-ratio-low |
| c03 | 163 | 0.95 | tool_outage | 1.00 | ends-compensated / decides-compensation_end |
| c04 | 156 | 0.94 | oscillation | 1.00 | ends-exhausted / finish-STOP / duration-s-highest |
| c05 | 134 | 0.96 | premature_stop | 1.00 | ends-completed / terminal-tool-missing |
| c06 | 132 | 0.93 | silent_truncation | 0.80 | finish-MAX_TOKENS / cap-finish-ratio-low / output-capped |
| c07 | 124 | 0.92 | gate_abandoned | 0.77 | gate-unresolved / gate-wait / ends-unterminated |
| c08 | 122 | 0.96 | context_exhaustion | 1.00 | ends-failed / max-prompt-tokens-highest |
| c09 | 84 | 0.93 | silent_truncation | 0.89 | finish-MAX_TOKENS / output-capped |
| c10 | 40 | 1.00 | premature_stop | 1.00 | terminal-tool-missing / ends-completed / no-tool-calls |

No cluster is incoherent by the 50% threshold. The weakest is c02 at 0.64, which mixes
`arg_hallucination` with `retry_storm`: both end in a failed run after repeated failing tool calls,
and once the tool names differ there is little structural difference left. c02 is the cluster to
distrust.

### Which planted modes the taxonomy actually recovered

| planted mode | largest cluster holds | clusters it spans |
| --- | --- | --- |
| context_exhaustion | 100.0% | 1 |
| gate_abandoned | 100.0% | 1 |
| oscillation | 100.0% | 1 |
| tool_outage | 100.0% | 1 |
| healthy | 96.4% | 1 |
| slow_tool | 95.9% | 1 |
| premature_stop | 76.1% | 2 |
| retry_storm | 75.5% | 2 |
| silent_truncation | 58.6% | 2 |
| arg_hallucination | 55.4% | 3 |

Four of the ten modes are recovered exactly. Three things this table is hiding, said plainly:

`slow_tool`'s 95.9% is not a success. Its runs land together because they land in c00, the healthy
cluster. A mode whose only distinguishing feature is one tool call taking 30 seconds instead of 300
milliseconds is structurally a healthy run, and this taxonomy does not see it. The cost view does,
as a per-tool maximum of 94 seconds against a 945 millisecond mean.

`silent_truncation` gets two clusters instead of one, c06 and c09, split by whether the truncation
marker landed in a model output or a tool result. Two coherent clusters for one mode is a mild
failure, and inverse purity charges for it.

`arg_hallucination` is the genuine weak point at 55.4% over three clusters. Its detector runs at
1.000 precision because a tool schema is decisive evidence that no structural feature has access to.

## Detector precision against the labelled sample

Same 2,400 runs. Labels are per-run sets, not one per run, because the detectors are not mutually
exclusive: an oscillating run that gives up without calling its terminal tool really has terminated
prematurely too, and the generator records both.

| pathology | fired | true pos | false pos | missed | precision | 95% CI | recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| retry_loop | 458 | 451 | 7 | 0 | 0.985 | 0.97 to 0.99 | 1.000 | 0.992 |
| oscillation | 185 | 158 | 27 | 0 | **0.854** | 0.80 to 0.90 | 1.000 | 0.921 |
| silent_truncation | 221 | 221 | 0 | 10 | 1.000 | 0.98 to 1.00 | 0.957 | 0.978 |
| premature_termination | 325 | 325 | 0 | 28 | 1.000 | 0.99 to 1.00 | 0.921 | 0.959 |
| hallucinated_tool_args | 220 | 220 | 0 | 0 | 1.000 | 0.98 to 1.00 | 1.000 | 1.000 |
| context_exhaustion | 120 | 120 | 0 | 9 | 1.000 | 0.97 to 1.00 | 0.930 | 0.964 |

Oscillation is the weakest at 0.854, with 27 false positives out of 185 firings. All of them are
runs where the decision channel happened to alternate between two routing values for long enough to
clear the coverage floor without the run being stuck. It is the one detector here whose alerts would
need a human to confirm.

Two of these numbers used to be much worse, and the fixes are in NOTES.md: context exhaustion ran at
0.418 until it required an absolute size floor rather than a growth ratio, and oscillation at 0.767
until it required the alternation to cover half the channel.

## The regression view

Two populations of 2,000 runs each. Clusters are fitted once on the pooled 4,000 so the same cluster
means the same thing in both halves, then split back apart. Every cluster gets a permutation test on
its prevalence (20,000 relabellings), a bootstrap CI on the risk ratio, and a
Benjamini-Hochberg correction across all clusters tested together.

### A planted change: retry_storm made 2.5 times more likely

The generator put 190 `retry_storm` runs in the before cohort and 431 in the after cohort.

```
$ obs regress corpus/runs --before before --after after

clusters
  c01  finish-STOP / ends-failed / terminal-tool-missing  13.80% -> 24.45%  RR 1.77 [1.55,2.04]  p=0.0000 q=0.0005  grew
  c00  max-prompt-tokens-high / prompt-growth-ratio-high  49.05% -> 44.05%  RR 0.90 [0.84,0.96]  p=0.0019 q=0.0097  shrank
  c06  ends-failed / fail-ratio-low / finish-STOP          2.50% ->  3.40%  RR 1.36 [0.94,1.96]  p=0.1156 q=0.1682  flat
  ... six more, all flat

pathologies
  retry_loop              16.80% -> 27.20%  RR 1.62 [1.43,1.83]  p=0.0000 q=0.0003  grew
  oscillation              9.65% ->  6.90%  RR 0.72 [0.58,0.89]  p=0.0023 q=0.0070  shrank
  premature_termination   14.10% -> 10.95%  RR 0.78 [0.66,0.92]  p=0.0039 q=0.0079  shrank

regressions: c01 (finish-STOP / ends-failed / terminal-too) x1.77
```

It found the planted change and named the right cluster: c01, the one whose members are 75%
`retry_storm`, grew from 13.80% to 24.45% of the population, risk ratio 1.77 with a 95% interval of
1.55 to 2.04 that excludes 1, q=0.0005. The `retry_loop` detector reads the same change directly at
risk ratio 1.62. Nine other clusters were tested and none was flagged.

The one report to read carefully is `shrank`. c00 and the oscillation and premature-termination
pathologies did fall, and that is arithmetic rather than a finding: prevalences sum to one, so making
one mode more common makes every other mode rarer as a share. The view reports the direction it
measured and does not pretend to know which way the causation runs.

### The null: two cohorts drawn from identical weights

The number that decides whether the view is usable at all.

```
$ obs regress corpus-null/runs --before before --after after

clusters: 13 clusters plus unclustered, smallest q = 0.9619, all flat
pathologies: 6 tested, smallest q = 0.2679, all flat

regressions: none at this alpha
```

Nothing flagged, from 14 cluster tests and 6 pathology tests on 4,000 runs. The smallest adjusted
p-value across the clusters was 0.96. Uncorrected, one cluster came in at p=0.0446; that is exactly
the false positive Benjamini-Hochberg exists to absorb, and it absorbed it.

Supporting evidence from the unit tests rather than the corpus: over 120 null comparisons the
permutation test's p-values are uniform (mean 0.35 to 0.65) and fall below 0.05 less than 12% of the
time, and over 150 rounds of 15 simultaneous null tests, uncorrected testing flags something in more
than half the rounds while BH flags something in fewer than 15.

## What each modelling decision is worth

Ablations on the same 2,400 runs, one change at a time.

| configuration | k | purity | inverse purity | ARI | NMI |
| --- | --- | --- | --- | --- | --- |
| full | 11 | 0.887 | 0.868 | 0.821 | 0.845 |
| no divisive refinement | 10 | 0.888 | 0.886 | 0.829 | 0.854 |
| one-hot scalar buckets instead of cumulative | 10 | 0.905 | 0.904 | 0.836 | 0.856 |
| one block weight for every block | 8 | 0.805 | 0.762 | **0.549** | 0.776 |
| k by silhouette alone | 11 | 0.887 | 0.868 | 0.821 | 0.845 |
| k by subsample stability alone | 12 | 0.894 | 0.773 | 0.598 | 0.826 |

Only one of these decisions is clearly carrying its weight on this corpus. Per-block normalisation
with tuned weights is worth 0.27 ARI: without it the tool n-gram block, which has most of the
columns, decides every distance. Choosing k by stability alone is worth avoiding, at 0.598.

The other three are honest nulls or worse on this corpus, and NOTES.md explains each:

Divisive refinement is very slightly negative here (0.829 to 0.821 ARI) because the selection
criterion already chose k=10. On a pooled two-cohort corpus where it chose k=5 the same pass took
purity from 0.750 to 0.846. It is kept for that case.

Cumulative bucket encoding is marginally *worse* than one-hot in the final configuration. It was
worth 0.14 ARI when it was introduced, on the block layout then in use, but the block split and
weight retune that followed solved the same problem. Kept as the correct encoding of an ordered
variable, not because this table supports it.

The silhouette-times-stability criterion picks the same k as silhouette alone here.

## Choosing k without labels

Four independent corpus seeds of 800 runs each. Each criterion picks a k, and the ARI it would have
got is looked up afterwards. The oracle row is the best k by ARI, which no unsupervised criterion can
see.

| criterion | k chosen per seed | mean ARI | worst ARI |
| --- | --- | --- | --- |
| silhouette | 6, 5, 5, 5 | 0.768 | 0.711 |
| subsample stability | 7, 7, 9, 7 | 0.598 | 0.567 |
| silhouette times stability (default) | 6, 5, 5, 5 | 0.768 | 0.711 |
| oracle (argmax ARI) | 5, 6, 5, 5 | 0.785 | 0.711 |

The default criterion lands within 0.017 ARI of an oracle that can see the labels. It also agrees
with silhouette alone on all four seeds at eight stability repetitions, so the stability term is not
earning its keep here; it is retained because the case it guards against, a compact partition that
does not reproduce on a different subsample, produces a taxonomy nobody could trust and would
otherwise be invisible.

## Generalisation to unseen corpora

The feature blocks and weights were chosen on one 600-run corpus (seed 1). These five seeds had no
part in choosing anything, at 1,200 runs each.

| seed | k | purity | inverse purity | ARI | NMI | incoherent clusters |
| --- | --- | --- | --- | --- | --- | --- |
| 51 | 9 | 0.907 | 0.935 | 0.872 | 0.891 | none |
| 52 | 9 | 0.897 | 0.947 | 0.837 | 0.887 | none |
| 53 | 10 | 0.900 | 0.748 | 0.581 | 0.817 | none |
| 54 | 10 | 0.931 | 0.799 | 0.628 | 0.848 | none |
| 55 | 7 | 0.837 | 0.979 | 0.756 | 0.852 | none |

Purity 0.837 to 0.931, mean 0.894. ARI 0.581 to 0.872, mean 0.735. The ARI spread is wider than the
purity spread and it comes from k: the seeds where the criterion picks 10 pay for the extra clusters
in inverse purity. No seed produced an incoherent cluster.

## Cost and latency attribution

From the same 2,400-run corpus. Tool spans carry no tokens of their own, which is correct and is why
their cost column is zero: a tool's cost is latency, and the model call that decided to invoke it
carries the tokens.

```
$ obs cost

by_tool
  key                     calls    cost$      tokens   mean ms    max ms    err
  submit_report            2312  0.00000           0     463.6     900.0   4.4%
  run_query                 742  0.00000           0    1287.2   92649.8  29.8%
  create_ticket             719  0.00000           0    1159.3   89208.1  28.2%
  send_email                773  0.00000           0    1017.4   84989.5  33.8%
```

Population total: $14.658 over 26,570,444 tokens. The mean tool call is around one second and the
worst is 94 seconds, which is the planted `slow_tool` mode surfacing here instead of in the taxonomy.

Termination status across the corpus: 1,359 completed, 609 failed, 176 compensated, 158 exhausted,
98 never terminated at all.

## Throughput

Ingest, span construction, cost attribution, featurisation, k sweep with stability, clustering and
all six detectors, single process, single core.

| runs | events | seconds | events/s |
| --- | --- | --- | --- |
| 300 | 6,611 | 0.4 | 15,284 |
| 1,200 | 26,405 | 2.6 | 10,340 |
| 2,400 | 52,743 | 9.7 | 5,457 |

Scaling is superlinear because the k sweep computes a silhouette per candidate k, and silhouette is
quadratic in the number of points (capped by a deterministic 1,500-point subsample). Fixing k with
`--k` removes the sweep entirely.

The regression view is separate and cheap: 16 permutation tests at 20,000 relabellings each took
0.2 seconds, because a two-proportion permutation null is a hypergeometric draw rather than an
actual array shuffle.

## Artefacts

| artefact | size |
| --- | --- |
| SQLite store, 2,400 runs with compressed events | 7.3 MB |
| self-contained HTML report | 27.3 KB |
| OTLP/JSON export, 300 traces / 5,987 spans | 3.3 MB |

The store holds every run's raw events gzipped, so any span tree can be rebuilt on demand rather
than 2,400 of them being pre-rendered for the handful anyone opens.

## Test suite

```
$ python -m pytest -q
208 passed in 17.24s
```

208 tests, no credentials, no network, no model calls. The suite includes the checks the rest of this
file depends on: that no generated run's event stream contains its own label, that no hand-authored
feature name mentions a pathology, that the permutation test is uniform under the null, and that the
regression view finds nothing between two cohorts drawn from identical weights.
