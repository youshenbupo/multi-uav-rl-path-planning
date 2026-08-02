# ruff: noqa: E501
"""Build a bounded, source-backed HTML-report artifact from validated summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCENARIOS = (
    "nominal",
    "delay_only",
    "loss_only",
    "dynamic_only",
    "combined",
    "ood_communication_obstacle",
)
SCENARIO_LABELS = {
    "nominal": "Nominal",
    "delay_only": "Delay only",
    "loss_only": "Loss only",
    "dynamic_only": "Dynamic only",
    "combined": "Combined",
    "ood_communication_obstacle": "OOD comm. + obstacle",
}
SCENARIO_SHORT_LABELS = {
    "nominal": "Nom",
    "delay_only": "Delay",
    "loss_only": "Loss",
    "dynamic_only": "Dyn",
    "combined": "Comb",
    "ood_communication_obstacle": "OOD",
}
TASK_METRICS = (
    "success",
    "collision",
    "terrain_violation",
    "threat_violation",
    "episode_return",
    "path_length",
    "mission_time",
    "minimum_separation",
    "temporal_conflict_count",
    "energy_proxy",
    "decision_latency",
)
CBF_METRICS = (
    "cbf_emergency_count",
    "cbf_emergency_fallback_rate",
    "cbf_intervention_rate",
    "cbf_mean_correction",
    "cbf_mean_solve_time_seconds",
)
ARM_LABELS = {
    "mlp_mappo": "MLP MAPPO",
    "raw_graph_mappo": "Self-loop GraphMAPPO",
    "predictive_graph_no_uncertainty": "Predictive graph",
    "uncertainty_predictive_graph": "Uncertainty-aware graph",
}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _scenario_map(payload: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    summaries = payload.get("scenario_summaries")
    if not isinstance(summaries, list):
        raise ValueError(f"{label} lacks scenario_summaries.")
    mapped = {str(item["scenario"]): item for item in summaries}
    if set(mapped) != set(SCENARIOS):
        raise ValueError(f"{label} must contain exactly the six canonical scenarios.")
    for scenario, item in mapped.items():
        if item.get("seed_count") != 5 or item.get("episodes_per_seed") != 20:
            raise ValueError(f"{label}/{scenario} must use five seeds and 20 episodes per seed.")
    return mapped


def _require_metrics(
    scenario: dict[str, Any], section: str, expected: tuple[str, ...], label: str
) -> dict[str, Any]:
    metrics = scenario.get(section)
    if not isinstance(metrics, dict) or set(metrics) != set(expected):
        raise ValueError(f"{label}/{section} does not match the frozen metric list.")
    return metrics


def _three_uav_rows(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mapped = _scenario_map(payload, "3-UAV summary")
    task_rows: list[dict[str, Any]] = []
    cbf_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    expected_arms = tuple(ARM_LABELS)
    for scenario in SCENARIOS:
        summary = mapped[scenario]
        for section, metric_names, destination in (
            ("task_metrics", TASK_METRICS, task_rows),
            ("cbf_diagnostic_metrics", CBF_METRICS, cbf_rows),
        ):
            metrics = _require_metrics(summary, section, metric_names, f"3-UAV/{scenario}")
            for metric in metric_names:
                arm_values = metrics[metric]
                if tuple(arm_values) != expected_arms:
                    raise ValueError(
                        f"3-UAV/{scenario}/{metric} has unexpected arm order or names."
                    )
                for arm in expected_arms:
                    value = arm_values[arm]
                    row = {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "metric": metric,
                        "arm": ARM_LABELS[arm],
                        "seed_mean": float(value["seed_mean"]),
                        "seed_sample_std": float(value["seed_sample_std"]),
                        "trained_seed_count": 5,
                        "episodes_per_seed": 20,
                    }
                    destination.append(row)
                    if metric == "success":
                        success_rows.append(dict(row))
    return {"task": task_rows, "cbf": cbf_rows, "success": success_rows}


def _paired_rows(payload: dict[str, Any], scale: str) -> dict[str, list[dict[str, Any]]]:
    mapped = _scenario_map(payload, f"{scale} summary")
    task_rows: list[dict[str, Any]] = []
    cbf_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        summary = mapped[scenario]
        for section, metric_names, destination in (
            ("task_metrics", TASK_METRICS, task_rows),
            ("cbf_diagnostic_metrics", CBF_METRICS, cbf_rows),
        ):
            metrics = _require_metrics(summary, section, metric_names, f"{scale}/{scenario}")
            for metric in metric_names:
                value = metrics[metric]
                row = {
                    "scale": scale,
                    "scenario": scenario,
                    "scale_scenario": f"{scale} | {scenario}",
                    "scale_scenario_label": f"{scale} | {SCENARIO_LABELS[scenario]}",
                    "scale_scenario_short": f"{scale.removesuffix(' UAV')} | {SCENARIO_SHORT_LABELS[scenario]}",
                    "metric": metric,
                    "no_uncertainty_seed_mean": float(value["reference_seed_mean"]),
                    "full_seed_mean": float(value["treatment_seed_mean"]),
                    "paired_delta_full_minus_no_uncertainty": float(
                        value["paired_delta_treatment_minus_reference_mean"]
                    ),
                    "paired_delta_sample_std": float(value["paired_delta_sample_std"]),
                    "trained_seed_count": 5,
                    "episodes_per_seed": 20,
                }
                destination.append(row)
                if metric == "success":
                    success_rows.append(dict(row))
    return {"task": task_rows, "cbf": cbf_rows, "success": success_rows}


def _source(
    source_id: str, label: str, path: str, dataset: str, definitions: list[str]
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path.replace("\\", "/"),
        "query": {
            "engine": "artifact-snapshot",
            "language": "sql",
            "sql": f"SELECT * FROM {dataset}",
            "description": "Read all reviewed rows from the named bounded snapshot dataset.",
            "tables_used": [f"snapshot.{dataset}"],
            "filters": [
                "All six canonical scenarios retained",
                "Seeds 20260719-20260723 retained",
                "20 episodes averaged within each seed-scenario cell",
                "No invalid or pre-actor-isolation roots included",
            ],
            "metric_definitions": definitions,
        },
    }


def _success_table(table_id: str, dataset: str, source_id: str, paired: bool) -> dict[str, Any]:
    if paired:
        title = "5/8-UAV success evidence"
        columns = [
            {"field": "scale_scenario_label", "label": "Cell"},
            {"field": "no_uncertainty_seed_mean", "label": "No-unc."},
            {"field": "full_seed_mean", "label": "Full"},
            {
                "field": "paired_delta_full_minus_no_uncertainty",
                "label": "Delta",
                "movement": True,
            },
            {"field": "paired_delta_sample_std", "label": "Delta SD"},
        ]
        default_sort = {"field": "scale_scenario_label", "direction": "asc"}
    else:
        title = "3-UAV success evidence"
        columns = [
            {"field": "scenario_label", "label": "Scenario"},
            {"field": "arm", "label": "Arm"},
            {"field": "seed_mean", "label": "Mean"},
            {"field": "seed_sample_std", "label": "SD"},
        ]
        default_sort = {"field": "scenario_label", "direction": "asc"}
    return {
        "id": table_id,
        "title": title,
        "subtitle": "Five trained seeds; each seed-scenario value averages 20 episodes.",
        "showDescription": True,
        "dataset": dataset,
        "density": "compact",
        "sourceId": source_id,
        "defaultSort": default_sort,
        "columns": columns,
    }


def build_artifact(
    three_uav_path: Path,
    five_uav_path: Path,
    eight_uav_path: Path,
    provenance_path: Path,
    generated_at: str,
    git_revision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    three = _three_uav_rows(_load(three_uav_path))
    five = _paired_rows(_load(five_uav_path), "5 UAV")
    eight = _paired_rows(_load(eight_uav_path), "8 UAV")
    scale_task = five["task"] + eight["task"]
    scale_cbf = five["cbf"] + eight["cbf"]
    scale_success = five["success"] + eight["success"]

    negative_success = [
        row for row in scale_success if row["paired_delta_full_minus_no_uncertainty"] < 0
    ]
    negative_text = ", ".join(
        f"{row['scale']} {row['scenario']} ({row['paired_delta_full_minus_no_uncertainty']:+.2f})"
        for row in negative_success
    )
    source_paths = {
        "three_uav": three_uav_path.as_posix(),
        "five_uav": five_uav_path.as_posix(),
        "eight_uav": eight_uav_path.as_posix(),
    }
    definitions = [
        "Seed mean: arithmetic mean over the five trained-seed cell means.",
        "Seed sample SD: sample standard deviation over five trained-seed cell means.",
        "Paired delta: full uncertainty-aware method minus independently trained no-uncertainty for the same seed, summarized over five paired seed differences.",
    ]
    sources = [
        _source(
            "source_three_uav_success",
            "Normalized 3-UAV success rows",
            source_paths["three_uav"],
            "three_uav_success",
            definitions,
        ),
        _source(
            "source_three_uav_task",
            "Normalized 3-UAV task rows",
            source_paths["three_uav"],
            "three_uav_task",
            definitions,
        ),
        _source(
            "source_three_uav_cbf",
            "Normalized 3-UAV CBF diagnostic rows",
            source_paths["three_uav"],
            "three_uav_cbf",
            definitions,
        ),
        _source(
            "source_scale_success",
            "Normalized 5/8-UAV paired success rows",
            provenance_path.as_posix(),
            "scale_success",
            definitions,
        ),
        _source(
            "source_scale_task",
            "Normalized 5/8-UAV paired task rows",
            provenance_path.as_posix(),
            "scale_task",
            definitions,
        ),
        _source(
            "source_scale_cbf",
            "Normalized 5/8-UAV paired CBF diagnostic rows",
            provenance_path.as_posix(),
            "scale_cbf",
            definitions,
        ),
    ]
    snapshot_datasets: dict[str, list[dict[str, Any]]] = {
        "three_uav_success": three["success"],
        "three_uav_task": three["task"],
        "three_uav_cbf": three["cbf"],
        "scale_success": scale_success,
        "scale_task": scale_task,
        "scale_cbf": scale_cbf,
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Post-isolation multi-UAV descriptive evidence",
            "description": "Technical report of the complete five-seed, six-scenario descriptive package.",
            "generatedAt": generated_at,
            "cards": [],
            "charts": [
                {
                    "id": "chart_three_uav_success",
                    "title": "3-UAV success by method and scenario",
                    "subtitle": "Across-seed mean; five trained seeds and 20 episodes per seed-scenario cell.",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "three_uav_success",
                    "sourceId": "source_three_uav_success",
                    "encodings": {
                        "x": {
                            "field": "scenario_label",
                            "type": "ordinal",
                            "label": "Scenario",
                        },
                        "y": {
                            "field": "seed_mean",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Success",
                        },
                        "color": {"field": "arm", "type": "nominal", "label": "Method"},
                        "tooltip": [
                            {"field": "arm", "type": "nominal"},
                            {"field": "seed_mean", "type": "quantitative", "format": "percent"},
                            {"field": "seed_sample_std", "type": "quantitative"},
                            {"field": "trained_seed_count", "type": "quantitative"},
                        ],
                    },
                    "yAxisTitle": "Success fraction",
                    "valueFormat": "percent",
                    "maxRows": 24,
                },
                {
                    "id": "chart_scale_full_success",
                    "title": "Full-method success by scale and scenario",
                    "subtitle": "5 and 8 UAV; all six scenarios; five trained seeds per cell.",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "scale_success",
                    "sourceId": "source_scale_success",
                    "encodings": {
                        "x": {
                            "field": "scale_scenario_short",
                            "type": "ordinal",
                            "label": "Scale and scenario",
                        },
                        "y": {
                            "field": "full_seed_mean",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Full-method success",
                        },
                        "tooltip": [
                            {
                                "field": "full_seed_mean",
                                "type": "quantitative",
                                "format": "percent",
                            },
                            {"field": "trained_seed_count", "type": "quantitative"},
                            {"field": "episodes_per_seed", "type": "quantitative"},
                        ],
                    },
                    "yAxisTitle": "Success fraction",
                    "valueFormat": "percent",
                    "maxRows": 12,
                },
            ],
            "tables": [
                _success_table(
                    "table_three_success",
                    "three_uav_success",
                    "source_three_uav_success",
                    False,
                ),
                _success_table(
                    "table_scale_success",
                    "scale_success",
                    "source_scale_success",
                    True,
                ),
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Post-isolation multi-UAV descriptive evidence",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## The evidence is complete but mixed\n\n"
                        "Every displayed arm has five independently trained seeds and all six prespecified scenarios. "
                        "The current package supports descriptive characterization only: directions differ by scale, "
                        "scenario, and metric, while collision outcomes often sit at the same zero floor. It does not "
                        "support statistical superiority, a safety guarantee, component-level causality, or robust generalization."
                    ),
                },
                {
                    "id": "three_uav_finding",
                    "type": "markdown",
                    "body": (
                        "## The 3-UAV arms do not form one uniform ordering\n\n"
                        "The grouped bars retain every scenario and show across-seed success means for all four independently "
                        "trained arms. Read them as an audit of variation, not a ranking: the 3-UAV arms share the adaptive CBF, "
                        "and the self-loop graph is the historical raw-graph label rather than a neighbour-state graph."
                    ),
                },
                {"id": "three_uav_chart", "type": "chart", "chartId": "chart_three_uav_success"},
                {
                    "id": "scale_finding",
                    "type": "markdown",
                    "body": (
                        "## The joint uncertainty pathway is not uniformly favorable at scale\n\n"
                        "The bars show only the full method's across-seed success mean across all 5/8-UAV cells; they provide "
                        "scale-scenario context and are not a paired-effect plot. The exact paired table later retains both arm "
                        f"means and shows negative full-minus-no-uncertainty deltas in {negative_text}. Zero and positive cells "
                        "remain alongside them. The joint graph-plus-CBF ablation cannot assign an effect to either component."
                    ),
                },
                {
                    "id": "scale_full_chart",
                    "type": "chart",
                    "chartId": "chart_scale_full_success",
                },
                {
                    "id": "definitions",
                    "type": "markdown",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The raw grain is arm, trained seed, scenario, and episode. Twenty episode records are averaged within "
                        "each seed-scenario cell; the five trained-seed cell means are the only repetition units. Reported SD is "
                        "the sample SD across those five units. The paired direction is always full minus independently trained "
                        "no-uncertainty. Task outcomes and CBF diagnostics are kept separate."
                    ),
                },
                {
                    "id": "three_success_note",
                    "type": "markdown",
                    "body": "## Exact 3-UAV success evidence\n\nThe compact table retains all six scenarios and four arms for exact lookup.",
                },
                {
                    "id": "three_success_table",
                    "type": "table",
                    "tableId": "table_three_success",
                },
                {
                    "id": "scale_success_note",
                    "type": "markdown",
                    "body": "## Exact 5/8-UAV success evidence\n\nThe table retains both arm means, full-minus-no-uncertainty delta, and paired delta SD for every scale-scenario cell.",
                },
                {
                    "id": "scale_success_table",
                    "type": "table",
                    "tableId": "table_scale_success",
                },
                {
                    "id": "complete_metric_evidence",
                    "type": "markdown",
                    "body": (
                        "## The full metric matrix remains in the artifact\n\n"
                        "The bounded snapshot retains all 11 task metrics and five CBF diagnostic metrics in separate "
                        "3-UAV and 5/8-UAV datasets. They remain available through artifact and source detail without "
                        "forcing unreadable broad tables into the report surface. Solver timing and replay counts are "
                        "diagnostic and timing-sensitive; they must not replace the raw fallback ledger."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Validation and provenance\n\n"
                        "The three input summaries were generated only after a frozen descriptive plan and independently "
                        "recomputed from 4,800 raw episode records within numerical tolerance 1e-12. This report reshapes those "
                        "validated values without recomputing them. The provenance sidecar records input hashes, Git revision, "
                        "row counts, chart contracts, and the claim boundary."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations remain decisive\n\n"
                        "There are only five trained seeds and no inferential procedure. The 5/8-UAV comparison jointly changes "
                        "graph risk and CBF margin. The uncertainty bound is deterministic rather than calibrated. The CBF uses "
                        "centralized simulator geometry, slack, discrete integration, and emergency fallback. Current results "
                        "therefore do not prove safety, real-time capability, causality, or generalization."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Next steps\n\n"
                        "1. Audit any manuscript table or figure against this report, the sidecar, and eligible raw roots.\n"
                        "2. Decide prospectively whether the existing paper remains descriptive or collects new evidence under a frozen inferential plan.\n"
                        "3. Train graph-only and CBF-only uncertainty ablations only if component-level causal language is retained.\n"
                        "4. Expand primary literature coverage and complete the final manuscript-to-artifact audit before readiness review."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "Would additional independently trained seeds materially narrow the effect estimates? Do graph-risk and "
                        "CBF-margin uncertainty contribute differently across communication conditions? Are collision floors caused "
                        "by the execution shield, task geometry, or insufficient evaluation stress? These questions remain open."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshot_datasets,
            "accessIssues": [],
        },
        "sources": [],
        "package_info": {
            "git_revision": git_revision,
            "claim_boundary": "descriptive_only",
            "external_submission_authorized": False,
        },
    }
    provenance = {
        "generated_at": generated_at,
        "git_revision": git_revision,
        "inputs": {
            name: {"path": path, "sha256": _sha256(Path(path))}
            for name, path in source_paths.items()
        },
        "independent_unit": "trained_seed",
        "trained_seed_count": 5,
        "episodes_per_seed_scenario": 20,
        "scenarios": list(SCENARIOS),
        "task_metrics": list(TASK_METRICS),
        "cbf_diagnostic_metrics": list(CBF_METRICS),
        "paired_delta_direction": "full_minus_independently_trained_no_uncertainty",
        "dataset_row_counts": {key: len(value) for key, value in snapshot_datasets.items()},
        "chart_contracts": [
            {
                "id": "chart_three_uav_success",
                "question": "How do descriptive 3-UAV success means vary across all methods and scenarios?",
                "family": "comparison",
                "type": "grouped bar",
                "palette_policy": "relaxed multi-category with direct method legend",
                "claim": "variation only; no ranking or superiority",
            },
            {
                "id": "chart_scale_full_success",
                "question": "How does the full method's descriptive success vary across scale and scenario?",
                "family": "comparison",
                "type": "single-series vertical bar",
                "palette_policy": "single-root",
                "claim": "full-method variation only; paired comparison remains in the adjacent table",
            },
        ],
        "omitted_visuals": [
            {
                "question": "What are the full-minus-no-uncertainty success deltas at 5 and 8 UAVs?",
                "replacement": "Exact compact table with both arm means, paired delta, and paired delta SD",
                "reason": "The packaged horizontal form failed desktop overflow QA and the vertical signed form forced the enhanced reader into fallback; no unverified chart is shipped.",
            }
        ],
        "validation_report": "docs/post_actor_isolation_descriptive_validation.md",
        "eligibility_registry": "docs/aamas2027_analysis_plan.md",
        "claim_boundary": {
            "allowed": [
                "five-seed descriptive means",
                "sample standard deviations",
                "paired descriptive deltas",
            ],
            "prohibited": [
                "significance",
                "superiority",
                "formal safety",
                "component causality",
                "robust generalization",
                "external submission",
            ],
        },
    }
    return artifact, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-uav-summary", type=Path, required=True)
    parser.add_argument("--five-uav-summary", type=Path, required=True)
    parser.add_argument("--eight-uav-summary", type=Path, required=True)
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--git-revision", required=True)
    args = parser.parse_args()
    if args.output_artifact.exists() or args.output_provenance.exists():
        raise FileExistsError("Report outputs must be new; refusing to overwrite an existing file.")
    artifact, provenance = build_artifact(
        args.three_uav_summary,
        args.five_uav_summary,
        args.eight_uav_summary,
        args.output_provenance,
        args.generated_at,
        args.git_revision,
    )
    args.output_artifact.parent.mkdir(parents=True, exist_ok=False)
    args.output_artifact.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.output_provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
