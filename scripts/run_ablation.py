"""Run Phase-14 ablations on the same single mainline interface."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    """Parse the common experiment contract before ablation dispatch."""
    from multiuav.experiments.ablation import load_ablation_manifest
    from multiuav.experiments.cli import build_mainline_parser, spec_from_args
    from multiuav.experiments.runner import (
        create_experiment_output,
        evaluate_hierarchical_checkpoint,
    )

    parser = build_mainline_parser("ablation")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/ablation_manifest.yaml",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=4)
    args = parser.parse_args()
    manifest = load_ablation_manifest(args.manifest)
    root = args.output_root / args.experiment_name
    root.mkdir(parents=True, exist_ok=True)
    resolution_entries: list[dict[str, object]] = []
    resolution: dict[str, object] = {"entries": resolution_entries, "manifest": str(args.manifest)}
    base_spec = spec_from_args(args, method="ablation_manifest")
    for entry in manifest.entries:
        item: dict[str, object] = {
            "name": entry.name,
            "disabled": list(entry.disabled),
            "availability": entry.availability,
            "reason": entry.reason,
        }
        if entry.availability == "available":
            arm_spec = replace(
                base_spec,
                name=entry.name,
                method=entry.name,
                checkpoint=entry.checkpoint,
                use_cbf=base_spec.use_cbf and "cbf" not in entry.disabled,
            )
            output = create_experiment_output(arm_spec, root)
            results = evaluate_hierarchical_checkpoint(
                arm_spec,
                output,
                config_path=args.config,
                episodes_per_seed=args.episodes_per_seed,
            )
            item["output"] = str(output.root)
            item["seeds"] = [result.seed for result in results]
        resolution_entries.append(item)
    resolution_path = root / "ablation_resolution.json"
    resolution_path.write_text(json.dumps(resolution, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(root), "entries": resolution_entries}))


if __name__ == "__main__":
    main()
