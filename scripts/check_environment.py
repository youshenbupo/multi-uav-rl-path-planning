"""Print the runtime capabilities required by the staged research project."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def package_version(distribution_name: str) -> str:
    """Return a distribution version or a stable unavailable marker."""
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "unavailable"


def torch_capabilities() -> tuple[str, str, str]:
    """Return the PyTorch version, CUDA availability, and GPU name."""
    try:
        import torch
    except ImportError:
        return "unavailable", "False", "unavailable"

    cuda_available = torch.cuda.is_available()
    gpu_name = "none"
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except RuntimeError:
            gpu_name = "unavailable"
    return torch.__version__, str(cuda_available), gpu_name


def main() -> None:
    """Print a concise, machine-readable human environment summary."""
    import platform

    torch_version, cuda_available, gpu_name = torch_capabilities()
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {package_version('numpy')}")
    print(f"PyTorch: {torch_version}")
    print(f"CUDA available: {cuda_available}")
    print(f"GPU: {gpu_name}")
    print(f"Gymnasium: {package_version('gymnasium')}")
    print(f"CVXPY: {package_version('cvxpy')}")
    print(f"OSQP: {package_version('osqp')}")


if __name__ == "__main__":
    main()
