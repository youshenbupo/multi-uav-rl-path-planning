# Reproducibility and audit contract

## Scope

This document reproduces the post-actor-isolation experiment protocol. It does
not make old, interrupted, or otherwise excluded roots eligible, and it does
not authorize external submission. Always consult
`docs/aamas2027_analysis_plan.md` for the authoritative eligibility registry.

## Software environment

The audited host uses Windows PowerShell and
`D:\anaconda3\envs\multiuav_rl\python.exe`. Create the environment from
`environment.yml`; it pins Python 3.11.15, PyTorch 2.13.0+cu130, NumPy 2.4.4,
SciPy 1.17.1, Gymnasium 1.3.0, PettingZoo 1.26.1, CVXPY 1.9.2, OSQP 1.1.3,
Ruff 0.15.21, mypy 2.3.0, and the remaining direct dependencies.

```powershell
conda env create -f environment.yml
conda activate multiuav_rl
python -m pip install -e .
```

Verify hardware/runtime identity before starting a run:

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

Neural training and checkpoint evaluation use `--device cuda`. OSQP remains a
CPU solver. Do not move OSQP to GPU or change its protocol.

## Frozen experimental identity

- Seeds: `20260719`, `20260720`, `20260721`, `20260722`, `20260723`.
- Requested budget: 100,000 transitions per independently trained seed.
- Evaluation: six named scenarios, 20 indexed episodes per checkpoint and
  scenario.
- Core configs: `configs/rl/dynamic_mappo_baseline.yaml`,
  `configs/rl/dynamic_graph_baseline.yaml`,
  `configs/rl/dynamic_graph_5uav.yaml`,
  `configs/rl/dynamic_graph_5uav_predictive_no_uncertainty_ablation.yaml`,
  `configs/rl/dynamic_graph_8uav.yaml`, and
  `configs/rl/dynamic_graph_8uav_predictive_no_uncertainty_ablation.yaml`.
- Safety protocol: slack penalty `100`, maximum iterations `20,000`, OSQP
  absolute/relative tolerance `1e-4`, and accepted solve time at most `0.1 s`.

The checked-in core configs, not the generic safety example file, define the
paper-facing CBF settings.

## Non-overwrite rule

Every launch must use a new, descriptive output directory. Before training,
verify that no Python/GPU training process remains and that the target directory
does not exist. Never delete, rename over, resume into, or regenerate an old
root in place. If a run fails, preserve its partial files and add an
`ABORTED.json` sidecar containing the reason and observed state. A retry uses a
new suffix such as `retry1`.

The workspace may contain user-owned changes. Inspect `git status` before and
after work; do not stage `.gitignore` or unrelated files. Update
`docs/progress.md`, `docs/known_issues.md`, and
`docs/next_conversation_handoff.md` before committing a completed phase.

## Artifact contract

An eligible training root must contain a natural final checkpoint, resolved
configuration/identity metadata, training summary, and CBF telemetry. An
eligible evaluation cell must contain exactly 20 JSONL episode records with
one seed, one checkpoint identity, one scenario, and episode indices 0--19.
The six cells for a seed must use its final checkpoint.

Every CBF emergency source is replayed with the exact source protocol. Preserve
the source JSON/JSONL, replay JSONL, and event context. A zero-event source is
still represented by a diagnostic record proving that zero events were found.
Replay is diagnostic provenance and is not substituted for task evaluation.

## Quality checks

Run these with the dedicated interpreter from the repository root:

```powershell
& 'D:\anaconda3\envs\multiuav_rl\python.exe' -m ruff check .
& 'D:\anaconda3\envs\multiuav_rl\python.exe' -m mypy --explicit-package-bases multiuav scripts
& 'D:\anaconda3\envs\multiuav_rl\python.exe' -m pytest -q
git diff --check
git status --short
```

The current full-suite exception is an unrelated legacy MATLAB inventory audit:
the test expects 81 `.m` files under the sibling path
`D:\yolo\multiuav\legacy_hgalo\HGALO_恢复源码`, but that path is absent on the
audited host and the test observes zero files. An earlier host snapshot recorded
83 files, so the dependency is both external and drift-prone. Do not create or
edit that inventory as part of the RL paper package without a separate
ownership audit.

## Statistical reproduction

The raw grain is `(arm, trained_seed, scenario, episode)`. Average 20 episodes
within each seed-scenario cell first. Treat the resulting five trained-seed
means as the only independent repetition units and report their arithmetic mean
and sample standard deviation. Same-number seed pairing is a blocking device,
not additional replication. Do not run episode-level inference.

The current validated descriptive artifacts and hashes are listed in
`docs/post_actor_isolation_descriptive_validation.md`. That report also lists
all eligible input roots and explicit exclusions. Any future inferential
cohort requires a separately specified and reviewed plan before its output is
generated. `docs/aamas2027_statistical_decision.md` prohibits retrospective
inference on the current already viewed five-seed cohort.

## Portable PDF reproduction

`environment.yml` pins `reportlab`, `pdfplumber`, and `pypdf` for the PDF
surface. Generate only into a new directory:

```powershell
$revision = git rev-parse HEAD
$generated = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
& 'D:\anaconda3\envs\multiuav_rl\python.exe' scripts\build_aamas_descriptive_pdf.py `
  --three-uav-summary outputs\paired_summaries\post_actor_isolation_3uav_four_arm_descriptive_20260801.json `
  --five-uav-summary outputs\paired_summaries\post_actor_isolation_5uav_full_vs_no_uncertainty_descriptive_20260801.json `
  --eight-uav-summary outputs\paired_summaries\post_actor_isolation_8uav_full_vs_no_uncertainty_descriptive_20260801.json `
  --output-dir output\pdf\<new-unique-root> `
  --generated-at $generated `
  --git-revision $revision
```

The command refuses an existing output directory. Reopen the PDF with `pypdf`
and `pdfplumber`, run `pdfinfo`, render every page with `pdftoppm`, and inspect
the complete rendered page set before eligibility. The current eligible file,
hashes, extracted-table checks, visual audit, and preserved failed attempts are
recorded in `docs/aamas2027_pdf_report_validation.md`.

## Traceability checklist

Before using a number in a manuscript:

1. Trace it to an eligible raw episode JSONL root.
2. Confirm checkpoint, seed, scenario, episode count, and graph mode.
3. Confirm the root is absent from every `ABORTED` and historical-exclusion
   list.
4. Recompute the seed-cell mean and across-seed statistic.
5. Keep task metrics separate from CBF diagnostic metrics.
6. Link any fallback statement to source and replay JSONL.
7. Record the Git commit, config hash, output hash, and generation command.
8. Preserve mixed and unfavorable cells; do not select endpoints post hoc.
