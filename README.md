# Communication-uncertainty-aware multi-UAV coordination

This repository is an auditable research implementation for a prospective
AAMAS 2027 paper. Its single paper-facing chain is:

> communication delay, loss, and staleness -> neighbour-state prediction
> uncertainty -> predictive interaction graph -> coordinated action ->
> uncertainty-adaptive CBF margin -> execution around dynamic obstacles.

The repository is not an autonomous submission system. No artifact here
authorizes an external paper submission.

## Evidence boundary

- An actor receives current truth only for itself. Neighbour information comes
  only from delivered packets that have not exceeded the configured staleness
  limit.
- Neural training and inference use CUDA. The online OSQP control-barrier
  filter runs on CPU and uses centralized simulator geometry at execution.
- The current uncertainty quantity is a deterministic age-dependent robustness
  bound, not a calibrated probabilistic standard deviation.
- The retained post-isolation results are five-seed descriptive evidence. They
  do not establish statistical superiority, a formal safety guarantee,
  real-time performance, or robust generalization.
- Invalid, interrupted, and pre-isolation artifacts are preserved for audit but
  excluded from paper-facing summaries.

## Current evidence package

The eligible post-actor-isolation package contains independently trained
checkpoints for seeds `20260719`--`20260723`, six scenarios, 20 evaluation
episodes per seed-scenario cell, retained episode JSONL, and all-source CBF
fallback replay provenance.

- 3 UAV: MLP MAPPO, self-loop GraphMAPPO, predictive graph without graph
  uncertainty, and uncertainty-aware predictive graph.
- 5 and 8 UAV: uncertainty-aware predictive graph and an independently trained
  joint no-uncertainty pathway ablation.

The 5/8-UAV ablation disables uncertainty in both graph risk and the CBF margin;
it does not isolate those two mechanisms separately. At 3 UAV, all four policy
arms retain the same uncertainty-adaptive CBF, so the predictive-graph arm is
only a graph-uncertainty comparison.

## Start here

- Research question and claim boundary: `docs/aamas2027_research_brief.md`
- Exact method and readiness assessment:
  `docs/aamas2027_method_and_readiness.md`
- Training protocol: `docs/training.md`
- Evaluation and aggregation protocol: `docs/evaluation.md`
- Reproducibility and audit contract: `docs/reproducibility.md`
- Descriptive-result validation:
  `docs/post_actor_isolation_descriptive_validation.md`
- Statistical decision for the current five-seed evidence:
  `docs/aamas2027_statistical_decision.md`
- Portable descriptive PDF and validation ledger:
  `docs/aamas2027_pdf_report_validation.md`
- Primary-source literature ledger: `docs/related_work_matrix.md`
- Manuscript assembly package: `docs/aamas2027_manuscript_package.md`
- Current status and handoff: `docs/progress.md` and
  `docs/next_conversation_handoff.md`

## Environment

Use the dedicated environment and checked-in lock specification:

```powershell
conda env create -f environment.yml
conda activate multiuav_rl
python -m pip install -e .
```

On the audited Windows host, the dedicated interpreter is
`D:\anaconda3\envs\multiuav_rl\python.exe`. See `docs/reproducibility.md` for
the exact validation commands and artifact rules.
