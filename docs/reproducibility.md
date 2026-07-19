# Phase-14 reproducibility

Use the exact conda environment and record every requested seed. The shared
output directory retains `config.yaml`, environment metadata, raw JSONL,
summary CSV/JSON, figures, checkpoints, TensorBoard, and logs. Summaries use
mean, sample standard deviation, and 95% Student-t intervals; failed episodes
and outliers are retained.

The supported scale matrix is 3, 5, 8, 12, and 16 UAVs in
`configs/experiments/scale_profiles.yaml`. CPU is valid for deterministic
debugging; `--device cuda` requires a CUDA-visible PyTorch device. OSQP CBF
solves remain CPU-side and its emergency telemetry must be reported.
