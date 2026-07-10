# Reproducibility Notes

## Final Code Branch

This public release contains the final organized implementation used for the reported FedHGF experiments.

An alternative graph-construction variant was tested during development, but it is not used in this public release because it degraded HAI performance.

## Main Results

The locked single-seed commands are provided in:

- `scripts/run_best_configs.bat`
- `scripts/run_best_configs.sh`

These commands correspond to the main table configuration used for the final FedHGF results.

## Three-Seed Runs

For multi-seed FedHGF runs, use `--seeds 42,123,2024`.

Example:

```bash
python -u run_fair_comparison.py --dataset hai --method fedhgf --seeds 42,123,2024 --device cuda --w-fusion 0.40,0.30,0.30 --threshold-mode f1_rate_guard --lambda-c 0.005 --lambda-v 0.4 --batch-size 32 --lr 3e-4 --flow-lr 3e-4
```

## Baselines

Baseline implementations are provided under `baseline/`. The baseline runner uses the same dataset registry and client partitions as FedHGF.

For BATADAL, use `batadal_small` and `--label-mode last` to match the FedHGF protocol used in the final table.

## Ablation

The ablation runner uses the locked best configuration for each dataset and then applies one module-level intervention at a time:

```bash
python -u run_ablation.py --datasets wadi hai batadal_small --seeds 42,123,2024 --device cuda
```

## Dataset Availability

The repository does not redistribute SWaT, WADI, HAI, or BATADAL. Users should obtain the datasets from their official sources and place them under the expected local `Data/` layout.
