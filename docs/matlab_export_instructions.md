# MATLAB Reference Export Instructions

Run the deterministic fixture exporter from PowerShell:

```powershell
matlab -batch "cd('D:/yolo/multiuav/reinforcement_learning'); addpath('tools/matlab'); export_reference_cases"
```

The script reads `legacy_hgalo/HGALO_恢复源码` and writes only to
`data/regression/matlab/`. It produces `reference_cases.mat` and
`metadata.json`.

Inspect the exported data from the configured Python environment:

```powershell
conda activate multiuav_rl
python scripts/inspect_matlab_reference.py
```

The restored baseline has no standalone CA-HGALO entrypoint. The exporter marks
algorithm-level CA-HGALO output as unavailable and never substitutes HGALO
output for it.
