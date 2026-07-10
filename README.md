# FedHGF

Official implementation for the heterogeneous federated multivariate time-series anomaly detection benchmark used in our FedHGF experiments.

FedHGF studies anomaly detection when each federated client observes a different subset of variables. The implementation includes the proposed FedHGF model, topology-guided dataset partitions, baseline runners, ablation scripts, and reproducibility notes.

## Repository Structure

```text
FedHGF/
├── run_fair_comparison.py      # Main FedHGF benchmark runner
├── run_ablation.py             # FedHGF ablation study runner
├── dataset_registry_fair.py    # Dataset protocol registry
├── fedgad_full.py              # FedHGF training and scoring logic
├── modules.py                  # Encoder, graph encoder, and flow modules
├── metrics.py                  # Evaluation helpers
├── data_loader_*.py            # Dataset-specific loaders
├── baseline/                   # Unified baseline implementations and runner
├── docs/                       # Dataset split and reproducibility notes
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The experiments were developed with PyTorch and CUDA. CPU execution is possible for small smoke tests but is not recommended for the full benchmark.

## Data Preparation

Place the raw/preprocessed datasets under the project-level `Data/` directory expected by the dataset loaders. The four main datasets are:

- SWaT
- WADI
- HAI
- BATADAL

The code does not redistribute these datasets. 




## Dataset Protocol

The benchmark uses topology-guided heterogeneous clients rather than random feature splits. Each client is split chronologically into normal training, calibration, and test segments. Sliding windows are generated independently inside each segment to avoid temporal leakage.

See [docs/data_split.md](docs/data_split.md) for the client definitions, anchors, and split statistics.

FedHGF experiments.
- GPU, PyTorch, and CUDA versions can cause small numerical differences. Reported multi-seed runs are recommended when comparing methods.

