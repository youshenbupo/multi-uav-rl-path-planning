# AAMAS 2027 statistical decision for the current evidence

**Decision date:** 2026-08-02

**Decision:** keep the existing post-actor-isolation package explicitly
descriptive. Do not add p-values, confidence intervals, significance decisions,
rankings, or superiority language to the current five-seed data.

## Why this is the defensible choice

The descriptive aggregation rule was fixed before the three summary artifacts
were generated, but no confirmatory endpoints, directional hypotheses, effect
thresholds, multiplicity rule, or inferential procedure were fixed before the
values were inspected. Selecting those elements now would be retrospective.
The complete values are also mixed across scale, scenario, and metric, which
makes a post-hoc endpoint choice especially vulnerable to selective reporting.

There are only five independently trained seeds per arm. Episode records are
nested within a trained policy and are not independent training replicates. An
exact paired sign-flip test over five nonzero seed differences has only 32 sign
assignments; a conventional two-sided test cannot attain a p-value below
`2/32 = 0.0625`. A one-sided test or mid-p convention selected after seeing the
directions would not repair the design. Parametric intervals with four degrees
of freedom would be unstable and would not overcome the retrospective endpoint
problem.

Accordingly, the current paper may report only:

- the 20-episode mean within each seed-scenario cell;
- the across-seed arithmetic mean and sample standard deviation over five
  trained-seed cell means;
- same-seed paired descriptive deltas, always full minus independently trained
  no-uncertainty; and
- complete task and CBF diagnostic sections with all six scenarios retained.

These quantities characterize the observed experiment package. They do not
establish that an effect differs from zero, that one method is superior, that a
component caused a change, or that the system is safe or generalizes robustly.

## Conditions for any future confirmatory extension

Any new inferential study must be frozen and committed before new seed results
are viewed. It must specify:

1. a primary comparison and whether it targets the integrated uncertainty
   pathway or separately trained graph-only and CBF-only components;
2. one primary scenario or a prespecified scenario-combination rule;
3. one primary task endpoint, its direction, and a practically meaningful
   effect threshold;
4. the trained seed as the independent unit and the exact pairing rule;
5. a sample-size/power or precision justification that does not treat episodes
   as independent policies;
6. the estimator, interval/test, assumptions, missing/failed-seed rule, and
   multiplicity policy;
7. a fixed stopping rule and new output roots; and
8. a commitment to retain unfavorable, null, invalid, and aborted outcomes.

Additional seeds collected under such a plan must not silently merge with the
current five-seed package as if the entire analysis had been prospective. The
manuscript must distinguish the original descriptive cohort from any later
confirmatory cohort.

## Manuscript wording

Use language such as "the five-seed descriptive mean was" and "the direction
varied across scenarios." Do not use "significant," "outperformed," "safer,"
"robust," or "generalized" for the current results. Sample standard deviation
describes observed between-seed spread; it is not a confidence interval.
