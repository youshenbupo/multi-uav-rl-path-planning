"""Inspect MATLAB regression exports without interpreting their semantics."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "data" / "regression" / "matlab"


def describe_array(name: str, value: Any) -> None:
    """Print shape, dtype, range, and non-finite counts for one value."""
    array = np.asarray(value)
    print(f"  Field: {name}")
    print(f"    Shape: {array.shape}")
    print(f"    Dtype: {array.dtype}")
    if np.issubdtype(array.dtype, np.number):
        finite_values = array[np.isfinite(array)]
        value_range = (
            "empty"
            if finite_values.size == 0
            else f"[{finite_values.min()}, {finite_values.max()}]"
        )
        print(f"    Range: {value_range}")
        print(f"    NaN count: {np.isnan(array).sum()}")
        print(f"    Inf count: {np.isinf(array).sum()}")


def nested_values(name: str, value: Any) -> Iterator[tuple[str, Any]]:
    """Yield leaf arrays from MATLAB structs and cell arrays with stable names."""
    if hasattr(value, "_fieldnames"):
        for field_name in value._fieldnames:
            yield from nested_values(f"{name}.{field_name}", getattr(value, field_name))
        return

    array = np.asarray(value)
    if array.dtype == object:
        for index, item in np.ndenumerate(array):
            yield from nested_values(f"{name}{index}", item)
        return

    if array.dtype.names is not None:
        for field_name in array.dtype.names:
            yield from nested_values(f"{name}.{field_name}", array[field_name])
        return

    yield name, value


def matlab_fields(mat_path: Path) -> Iterator[tuple[str, Any]]:
    """Yield top-level fields for MATLAB v7 files or v7.3 HDF5 files."""
    try:
        loaded = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    except NotImplementedError:
        with h5py.File(mat_path, "r") as handle:
            for name, dataset in handle.items():
                yield name, dataset[()]
        return

    for name, value in loaded.items():
        if not name.startswith("__"):
            yield name, value


def inspect_file(mat_path: Path) -> None:
    """Print a self-contained summary of one MATLAB reference file."""
    print(f"File: {mat_path.name}")
    fields = list(matlab_fields(mat_path))
    print("Fields: " + (", ".join(name for name, _ in fields) or "none"))
    if not fields:
        print("  Shape: n/a")
        print("  Dtype: n/a")
    for name, value in fields:
        for nested_name, nested_value in nested_values(name, value):
            describe_array(nested_name, nested_value)


def main() -> None:
    """Inspect all MATLAB reference files under the configured data directory."""
    mat_files = sorted(REFERENCE_ROOT.glob("*.mat"))
    print("MATLAB reference files:")
    if not mat_files:
        print("Fields: none")
        print("Shape: n/a")
        print("Dtype: n/a")
        print("No MATLAB exports found. Run tools/matlab/export_reference_cases.m first.")
        return

    for mat_path in mat_files:
        inspect_file(mat_path)

    metadata_path = REFERENCE_ROOT / "metadata.json"
    print(f"Metadata: {'present' if metadata_path.is_file() else 'missing'}")


if __name__ == "__main__":
    main()
