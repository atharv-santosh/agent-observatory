# Build notes

A real log of building this, including the things that were wrong until measurement said so.
The point of the repo is that the clustering is validated, so the honest record of what the
validation rejected is part of the work.

## Why the corpus came first

The first thing built was not the ingest or the clustering, it was the generator. Clustering a pile
of real trajectories produces clusters that look plausible and cannot be checked, and that is
exactly the failure mode this repo is supposed to avoid. So the order was: plant ten failure modes,
write what was planted to a sidecar file, then build a clusterer that never sees it and measure how
much it recovers.

Two rules came out of that and shaped everything after:

The ground truth is never written into the event stream. There is a test that serialises every
generated run and searches it for its own label, because if that regresses then every purity number
here becomes a lie and nothing would visibly break.

The modes deliberately overlap. They share a tool vocabulary, their step and token budgets overlap,
and about eight percent of runs carry an incidental second pathology recorded truthfully in the
labels. A corpus where every mode is obvious would prove nothing.

## The feature design constraint

No feature may encode a taxonomy. An `is_retry_loop` column would be a hand label smuggled into the
input, and a cluster built on it could only rediscover the guess. So the features are structural:
tool n-grams, how periodic the tool sequence is at each period from one to four, recurring error
text shapes, the token curve, how the run stopped.

A retry loop and an oscillation both come out of one measurement, the longest periodic run in the
sequence: period one and period two of the same thing. The detectors get to name them, the features
do not. There is a test asserting no authored feature name contains any pathology name.

## Cumulative bucket encoding, and why the final ablation does not support it

Scalars like "how many repeats in the longest cycle" get bucketed by corpus quantiles. The first
version encoded the bucket one-hot, and the clustering was poor in a way that took a while to
understand: the healthy population kept fragmenting into four, five, six clusters while the rare
failure modes got absorbed into it.

One-hot destroys the ordering. Buckets of a scalar are ordered, so "two repeats" should be near
"one repeat" and far from "seven", but one-hot makes all three equidistant. Incidental variation
inside the healthy population, a tool that happened to repeat twice, therefore read as a categorical
difference and split the cluster. Cumulative encoding fixes that by construction: bucket b lights
columns 0 through b, so adjacent buckets differ in one column and distant ones differ in several.

Switching to it was a large improvement at the time. On a 600-run corpus, with the block layout then
in use and nothing else changed:

| bucket encoding | k | purity | inverse purity | ARI | NMI | healthy coverage |
| --- | --- | --- | --- | --- | --- | --- |
| one-hot, 5 buckets | 11 | 0.864 | 0.651 | 0.457 | 0.753 | 0.43 |
| cumulative, 6 buckets | 7 | 0.827 | 0.860 | 0.597 | 0.795 | 0.74 |

Purity went down and everything else went up, which is the point: one-hot was buying purity by
shattering the majority class, and inverse purity and ARI are the metrics that notice. The healthy
population went from four clusters to one.

Then, at the end, the ablation was re-run properly on the final 2,400-run corpus with the final
block layout and the block weights that were retuned after this change. One-hot comes out
marginally *better*: ARI 0.836 against 0.821, purity 0.905 against 0.887.

So the honest conclusion is that this was not an independent win. Splitting the `size` and `time`
blocks and retuning the weights, both done afterwards, solved the same fragmentation problem, and
by the end the encoding was no longer load bearing. The three changes are confounded and the final
ablation cannot separate them. Cumulative encoding is kept because it is the correct encoding of an
ordered variable and because the current weights were tuned against it, not because the ablation
justifies it on its own. Anyone reading the measurement table and concluding the encoding does
nothing on this corpus is reading it right.

## Splitting the feature blocks

Features go in named blocks, each L2-normalised separately before being scaled by a block weight,
because the n-gram block has by far the most columns and otherwise quietly decides every distance.

The first block layout had one `shape` block holding step counts, token totals, durations and tool
latency together. That forced one weight onto two different kinds of signal: step and token counts
are mostly nuisance variation that fragments the healthy population, while a 90-second tool call is
a real finding. Splitting them into `size` (weight 0.35) and `time` (weight 1.10) was worth more
than any amount of tuning the single combined weight.

`plan_completion` also moved out of the size block into `term`, where it belongs: it is a statement
about how the run ended, not about how big it was.

A smaller one in the same spirit: decision variable names carry a step ordinal, so `route_3=auto`
and `route_7=auto` were different features. That turns a step index into high-cardinality noise, and
it alone was splitting the healthy population. Stripping the trailing ordinal in `derive` fixed it.

## Choosing k, and then not choosing it

Silhouette alone is a bad way to pick k here. It is a geometric compactness score, it is biased
towards small k on cosine data, and across four corpus seeds it picked anything from 5 to 7 for the
same generator. Subsample stability (cluster two overlapping 80% subsamples, measure the adjusted
Rand between them on the shared runs) is a better question but prefers coarse taxonomies for the
same reason.

The default is the product of the two, on the reasoning that a k should have to make compact
clusters and make the same ones again on a different subsample. The table is in MEASUREMENTS.md.

Two corrections to that, both from re-measuring at the end with eight stability repetitions instead
of four. First, the product and silhouette alone pick the same k on all four seeds, both
reaching mean ARI 0.768 against an oracle 0.785. The gap that justified the product (0.812 against
0.795) was measured at four repetitions and does not survive a better estimate of stability.
Stability alone remains clearly worse, at 0.598. The product is kept because the case it guards
against, a compact partition that does not reproduce on a different subsample, is the one that
would silently produce a taxonomy nobody could trust; but on this corpus it is not doing visible
work, and MEASUREMENTS.md says so.

Second, the divisive refinement pass. On a pooled two-cohort corpus where the criterion chose k=5,
refinement took purity from 0.750 to 0.846 while inverse purity fell only from 0.984 to 0.953,
against 0.794 for forcing flat k to 9 instead. That is a real win and it is why the pass exists. But
on the single-cohort corpus the headline numbers come from, where the criterion already chooses
k=10, refinement adds one cluster and changes nothing: purity 0.888 to 0.887, ARI 0.829 to 0.821, so
very slightly worse. It is on by default because the failure it prevents is much worse than the
cluster it sometimes adds for nothing, not because it helps everywhere.

## Two detectors that were wrong until they were measured

**Context exhaustion** first fired on prompt growth ratio alone, and ran at 0.418 precision. The
reason is obvious in hindsight: every multi-turn agent's prompt grows several-fold because history
accumulates, so "the prompt grew four times over" describes almost every run that took more than a
few turns. Exhaustion is about approaching a limit, not about growing. Adding an absolute floor
against the declared context window took it to 1.000. When a run does not declare its window the
detector assumes a conservative one and records in the finding that it assumed, because refusing to
fire silently would be worse.

**Oscillation** had no coverage floor, so four elements of ABAB anywhere in a run counted. Four
alternating calls turn up by chance often enough that the decision channel alone produced about a
quarter of the firings. Requiring the alternation to cover at least half the channel, and three
cycles rather than two on the noisier decision channel, took precision from 0.767 to 0.854. It is
still the weakest detector and MEASUREMENTS.md says so.

**Premature termination** looked like it had a precision problem, at 0.583, and it did not. It was
firing on oscillating runs that gave up without calling their terminal tool, which is a premature
termination by any reasonable definition. The bug was in the corpus labels, not the detector: they
recorded one pathology per run when the detectors are not mutually exclusive. Labelling those runs
with both took precision to 1.000 without touching the detector. The same applied to
`arg_hallucination` runs that made the same rejected call three or more times, which really is also
a retry loop.

That is worth stating plainly, because it is the kind of thing that makes a precision number
meaningless: if the labels are wrong in a way that correlates with the detector, the measurement
tells you nothing.

## Bugs found by running it rather than reading it

* **Every completion reported zero output tokens.** `usage_tokens` walks a list of candidate key
  pairs and matched if *either* key was present, so a Gemini-shaped payload with
  `prompt_tokens`/`candidates_tokens` matched the OpenAI pair `prompt_tokens`/`completion_tokens` on
  its input key alone and returned the output count as zero. Every cost figure in the repo was low.
  Caught by printing one run's span tree and noticing `2458+0tok`. Both keys are now required before
  a pair matches, with the either-key behaviour kept as a fallback for genuinely partial payloads.

* **Five sixths of the sandbox corpus vanished.** agent-sandbox stores the same scenario ids under
  one directory per model configuration, so keying run ids on the scenario id alone collapsed six
  configurations of `direct_injection-01` into one run. The pipeline deduplicates by run id, so
  there was no error, just a corpus that was 233 runs instead of 568. Caught by a test asserting
  the fixture tree yields more than 300 trajectories.

* **`obs gen --with-regression` wrote a stale third cohort.** The baseline cohort was generated and
  its summary then overwritten by the before cohort's, but the files had already been written. The
  next `obs analyse` over the corpus directory silently picked up 3,600 runs across three cohorts
  instead of 2,400 across two, which is how the first regression measurement came out wrong.

* **The incidental pathology injection used step index 900.** Harmless but silly, and it showed up
  in the span tree as `agent.step[900]` sitting after `agent.step[4]`. Now it continues from the
  run's own last step.

## Things left deliberately imperfect

`slow_tool` gets no cluster of its own. 95.9% of its runs land in a single cluster, which reads well
until you notice the cluster is the healthy one: the mode's only distinguishing feature is one tool
call taking 30 seconds instead of 300 milliseconds, and structurally the run is a healthy run.
Weighting the `time` block high enough to separate it costs more elsewhere than it gains. A
latency-only regression is a job for the cost view, which surfaces it directly as a per-tool maximum,
and not for a taxonomy built on trajectory structure.

`arg_hallucination` is the worst-recovered mode, at 55.4% in its largest cluster and spread over
three. It is genuinely hard: a run that made an invalid call and failed looks a lot like a run that
made a valid call that failed, once the tool names differ. Its detector runs at 1.000 precision
because the tool schema is decisive evidence that no structural feature has access to, which is a
fair illustration of why both halves of this repo exist.

`silent_truncation` gets two clusters rather than one (58.6% in the larger), split by whether the
truncation marker landed in a model output or a tool result. Two coherent clusters for one planted
mode is a less serious failure than one incoherent cluster for two modes, and the split is legible
in the cluster names, but inverse purity counts it against the taxonomy and should.

The `phantom_value` sub-signal in the hallucinated-argument detector (an identifier-shaped argument
that never appeared anywhere earlier in the run) is reported at lower strength than the schema
violations, because it is a heuristic and the others are not. It contributes no firings on this
corpus; it is there for real trajectories where an agent invents a customer id.
