"""Validate HDF5 expert episodes, including evaluator cost replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Validate a dataset and return a failing process status when it is invalid."""
    from multiuav.data.validation import validate_expert_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data/expert/matlab_experts.h5",
    )
    args = parser.parse_args()
    report = validate_expert_dataset(args.dataset)
    print(f"Episodes: {report.episode_count}")
    if report.is_valid:
        print("Validation: passed")
        return
    print("Validation: failed")
    for issue in report.issues:
        print(f"- {issue}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
