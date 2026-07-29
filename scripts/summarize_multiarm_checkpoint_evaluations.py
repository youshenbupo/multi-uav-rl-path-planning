"""Create a descriptive, seed-level summary for matched checkpoint-evaluation arms.

Each seed-scenario cell comprises twenty evaluation episodes. This tool aggregates
within that cell and treats trained seeds, never individual episode records, as
the descriptive units. It intentionally reports no p-values or method claims.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_paired_checkpoint_evaluations import (
    EPISODES_PER_CELL,
    METRIC_FIELDS,
    _cell_mean,
    _load_cells,
    _sample_std,
)


def _load_arm_cells(
    root: Path, cell_prefix: str | None
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Load all cells or only cells whose immediate directory has a prefix."""
    if cell_prefix is None:
        return _load_cells(root)

    cells: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for jsonl_path in sorted(root.rglob("*.jsonl")):
        if not jsonl_path.parent.parent.name.startswith(cell_prefix):
            continue
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != EPISODES_PER_CELL:
            raise ValueError(
                f"{jsonl_path} must contain {EPISODES_PER_CELL} nonblank records, "
                f"got {len(records)}"
            )
        first = records[0]
        if not isinstance(first.get("seed"), int) or not isinstance(first.get("scenario"), str):
            raise ValueError(f"{jsonl_path} lacks integer seed or string scenario")
        key = (first["seed"], first["scenario"])
        if key in cells:
            raise ValueError(f"Duplicate seed-scenario cell {key} under {root}")
        episodes = sorted(record.get("episode") for record in records)
        if episodes != list(range(EPISODES_PER_CELL)):
            raise ValueError(
                f"{jsonl_path} must contain episode indices {list(range(EPISODES_PER_CELL))}"
            )
        if any(
            record.get("seed") != key[0] or record.get("scenario") != key[1]
            for record in records
        ):
            raise ValueError(f"{jsonl_path} mixes seed or scenario identities")
        cells[key] = records
    return cells


def summarize_multiarm_roots(
    arm_roots: dict[str, Path], *, arm_prefixes: dict[str, str] | None = None
) -> dict[str, Any]:
    """Summarize matched arms without treating episodes as independent replicates."""
    if len(arm_roots) < 2:
        raise ValueError("At least two named arms are required")

    arm_prefixes = arm_prefixes or {}
    unexpected_prefixes = set(arm_prefixes) - set(arm_roots)
    if unexpected_prefixes:
        raise ValueError(f"Prefixes given for unknown arms: {sorted(unexpected_prefixes)}")
    cells_by_arm = {
        name: _load_arm_cells(root, arm_prefixes.get(name)) for name, root in arm_roots.items()
    }
    reference_name = next(iter(cells_by_arm))
    reference_keys = set(cells_by_arm[reference_name])
    if not reference_keys:
        raise ValueError("No JSONL cells found")
    if any(set(cells) != reference_keys for cells in cells_by_arm.values()):
        raise ValueError("Arm cell identities differ")

    scenario_summaries: list[dict[str, Any]] = []
    for scenario in sorted({scenario for _, scenario in reference_keys}):
        keys = sorted(key for key in reference_keys if key[1] == scenario)
        task_metrics: dict[str, dict[str, dict[str, float]]] = {}
        cbf_diagnostic_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for metric in METRIC_FIELDS:
            arm_metrics: dict[str, dict[str, float]] = {}
            try:
                for arm_name, cells in cells_by_arm.items():
                    values = [_cell_mean(cells[key], metric) for key in keys]
                    arm_metrics[arm_name] = {
                        "seed_mean": statistics.fmean(values),
                        "seed_sample_std": _sample_std(values),
                    }
            except ValueError:
                continue
            if metric.startswith("cbf_"):
                cbf_diagnostic_metrics[metric] = arm_metrics
            else:
                task_metrics[metric] = arm_metrics
        if not task_metrics and not cbf_diagnostic_metrics:
            raise ValueError(f"No shared numeric metrics for scenario {scenario!r}")
        scenario_summaries.append(
            {
                "scenario": scenario,
                "seed_count": len(keys),
                "episodes_per_seed": EPISODES_PER_CELL,
                "seed_ids": [key[0] for key in keys],
                "task_metrics": task_metrics,
                "cbf_diagnostic_metrics": cbf_diagnostic_metrics,
            }
        )

    return {
        "arm_roots": {name: str(root) for name, root in arm_roots.items()},
        "arm_cell_prefixes": arm_prefixes,
        "independent_unit": "trained_seed",
        "episode_records_are_not_independent_training_replicates": True,
        "scenario_summaries": scenario_summaries,
        "claim_boundary": (
            "descriptive seed-cell summary only; no significance, method-effect, "
            "safety, or causal claim"
        ),
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    """Parse one ``NAME=ROOT`` CLI argument."""
    name, separator, raw_root = value.partition("=")
    if not separator or not name or not raw_root:
        raise argparse.ArgumentTypeError("--arm must use NAME=ROOT")
    return name, Path(raw_root)


def _parse_prefix(value: str) -> tuple[str, str]:
    """Parse one ``NAME=PREFIX`` CLI argument."""
    name, separator, prefix = value.partition("=")
    if not separator or not name or not prefix:
        raise argparse.ArgumentTypeError("--cell-prefix must use NAME=PREFIX")
    return name, prefix


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for the provenance-preserving multi-arm summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        type=_parse_arm,
        action="append",
        required=True,
        metavar="NAME=ROOT",
        help="named checkpoint-evaluation root; repeat once per arm",
    )
    parser.add_argument(
        "--cell-prefix",
        type=_parse_prefix,
        action="append",
        default=[],
        metavar="NAME=PREFIX",
        help="optional cell-directory prefix for an arm sharing an evaluation root",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    """Validate named roots and serialize their descriptive summary."""
    args = build_parser().parse_args()
    arm_roots = dict(args.arm)
    if len(arm_roots) != len(args.arm):
        raise ValueError("Arm names must be unique")
    arm_prefixes = dict(args.cell_prefix)
    if len(arm_prefixes) != len(args.cell_prefix):
        raise ValueError("Arm prefix names must be unique")
    summary = summarize_multiarm_roots(arm_roots, arm_prefixes=arm_prefixes)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
