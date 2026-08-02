# AAMAS 2027 paper configuration record

**Prepared:** 2026-08-02

**Status:** awaiting explicit user confirmation under the academic-paper Phase
0 gate. This record does not authorize external submission.

## Recommended configuration

| Field | Recommended value | Evidence or constraint |
| --- | --- | --- |
| Paper type | Empirical multi-agent systems conference paper | The repository contains implemented MARL methods, controlled simulation experiments and retained evidence. |
| Discipline | Multi-agent reinforcement learning, safe autonomy and multi-UAV coordination | This keeps the paper on the fixed communication-to-uncertainty-to-graph-to-CBF mainline. |
| Target venue | AAMAS 2027 | Exact 2027 template, page limit and policy must be verified from the official call when available. No submission action is authorized. |
| Working title | *Communication-Uncertainty-Aware Predictive Graph Coordination with an Adaptive CBF Execution Shield for Multi-UAV Systems* | Avoids unsupported “safe”, “guaranteed”, “robust”, “calibrated”, “real-time” and “first” wording. |
| Main language | English | Appropriate for the target venue. A Chinese abstract may be retained as an internal companion, not presumed to be part of the submission. |
| Canonical drafting format | Markdown in the repository | Keeps claims and artifact links auditable before venue formatting. LaTeX conversion must wait for the official AAMAS 2027 template. |
| Citation format | Venue-native BibTeX style, exact style pending official 2027 instructions | Do not guess a style or convert citations before the venue source is available. Every bibliographic identity and DOI must be verified. |
| Main-text target | 5,500--6,500 English words before venue-template compression | A working budget only. The official 2027 page limit overrides it after verification. |
| Abstract | 180--230 English words plus an internal Chinese companion | The abstract must state five seeds, six scenarios, descriptive-only mixed results and limitations without superiority or safety language. |
| Evidence level | Descriptive simulation study | The already inspected five-seed cohort has no prospective inferential design. |
| Primary comparison claim | Integrated uncertainty pathway only | The 5/8-UAV ablation jointly removes graph-risk and CBF-margin uncertainty. |
| Component ablation decision | Do not run graph-only and CBF-only arms unless the user chooses a component-level causal claim | Recommended because the defensible paper can report the integrated mechanism and current evidence is mixed. |
| External action | Prohibited | No upload, submission, author-account action or external correspondence without new explicit authorization. |

## Fixed paper question

How does an auditable integrated mechanism that predicts only delivered stale
neighbour packets, propagates a deterministic age-dependent uncertainty bound
through predictive interaction-graph risk, and adapts an online CBF execution
margin behave under delayed/lost communication and dynamic obstacles?

This is a characterization question. The current manuscript must not rewrite it
as a superiority, formal-safety, calibrated-uncertainty or robust-generalization
hypothesis.

## Claim contract

The draft may claim implementation and descriptive evaluation of:

1. an actor that sees own truth and delivered neighbour packets only;
2. packet-age state prediction and a deterministic uncertainty proxy;
3. propagation of that proxy into graph risk and the online CPU-OSQP CBF
   margin;
4. matching five-seed, six-scenario evidence at 3 UAV and independently trained
   integrated-pathway comparisons at 5/8 UAV; and
5. complete raw episode, invalid-root and CBF replay provenance.

It may not claim:

- statistical significance or superiority;
- a safety guarantee or decentralized CBF;
- calibrated uncertainty;
- component-level graph or CBF causality;
- real-time capability or robust generalization;
- exhaustive novelty or a “first” combination; or
- external submission readiness based only on the current scaffold.

## Planned structure and word budget

| Section | Working words | Required job |
| --- | ---: | --- |
| Abstract | 180--230 | State problem, integrated mechanism, exact evaluation scale, mixed descriptive result and limitations. |
| 1. Introduction | 650--800 | Motivate coupled communication and execution uncertainty; state bounded contributions. |
| 2. Related work | 700--850 | Use only the source-tiered DACOM, DHCG, graph-CBF and UAV evidence ledger; no field-wide exclusion. |
| 3. Problem formulation | 650--800 | Separate actor, critic and execution-filter information boundaries. |
| 4. Method | 1,200--1,450 | Define packet age/prediction, uncertainty proxy, graph construction, MAPPO and CBF/fallback. |
| 5. Experimental protocol | 850--1,000 | Record frozen seeds, transitions, scenarios, metrics, exclusions, CUDA/CPU split and descriptive aggregation. |
| 6. Results | 900--1,100 | Show all six scenarios, mixed task outcomes and separate CBF diagnostics without ranking. |
| 7. Limitations and broader validity | 550--700 | Cover five seeds, joint ablation, deterministic proxy, centralized CBF, solver timing and simulation scope. |
| 8. Reproducibility and declarations | 350--500 | Link raw roots, hashes and code; include mandatory declarations without fabrication. |

## Existing inputs accepted for drafting

- method boundary: `docs/aamas2027_method_and_readiness.md`;
- paper scaffold: `docs/aamas2027_manuscript_package.md`;
- statistical boundary: `docs/aamas2027_statistical_decision.md`;
- eligible-root and aggregation plan: `docs/aamas2027_analysis_plan.md`;
- independently validated values:
  `docs/post_actor_isolation_descriptive_validation.md`;
- portable result surface and hashes:
  `docs/aamas2027_pdf_report_validation.md`;
- primary-source evidence tiers: `docs/related_work_matrix.md`;
- reproducibility commands: `docs/reproducibility.md`; and
- current exclusions and runtime limitations: `docs/known_issues.md`.

## Fields that cannot be inferred or fabricated

The user must supply or explicitly approve the following before finalization:

1. author names, order, affiliations and contact details;
2. CRediT author contributions;
3. funding sources or a confirmed no-funding statement;
4. conflicts of interest or a confirmed no-conflict statement;
5. intended public data/code availability and any repository URL;
6. final AI-assistance disclosure wording;
7. whether the recommended integrated-pathway-only claim is accepted; and
8. any stricter target length or formatting preference.

The simulation-only study currently indicates no human-subject, animal or
personal-data involvement, but the final ethics declaration must still be
reviewed by the user.

## Confirmation requested

Recommended approval is:

> Approve this configuration; draft an English AAMAS 2027 internal manuscript
> in Markdown at 5,500--6,500 words, retain descriptive integrated-pathway
> claims only, do not run component-only ablations, keep author/funding/COI/data
> availability fields visibly unresolved, and never submit externally.

Any requested change to the claim level, new experiments, target length or
declaration fields must be stated before Phase 1/2 drafting continues.
