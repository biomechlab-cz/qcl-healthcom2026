# Hardware-Validated Multimodal Quantum Classifier for Biosignal Stress Recognition

Experiment code for the IEEE Healthcom 2026 paper. Scripts to reproduce all results from raw WESAD data.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

For RTX 50-series GPUs (Blackwell), install PyTorch with CUDA 12.8:
```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

## Data

Download WESAD from https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ and unpack so that `data/raw/WESAD/data/S{n}/S{n}.pkl` exists for each subject (15 subjects; S1 and S12 missing in the official release).

## Reproducing the experiments

Run in order from the repo root:

```bash
# 1. Preprocessing (raw ECG+EDA, no per-subject scaling)
python scripts/preprocess_ecg_eda.py --datasets WESAD --signal-sets ecg_eda --tasks binary

# 2. Feature extraction (27-feature HRV+EDA vector)
python scripts/extract_ecg_eda_features.py \
  "data/processed/WESAD_binary_ecg_eda_30s_50ovl.pkl" \
  --output-dir data/features

# 3. Classical baselines (Table 1)
python scripts/run_baselines_v1.py --datasets WESAD

# 4. Deep baselines (Table 1)
python scripts/run_deep_baselines.py \
  --datasets WESAD --models cnn1d transformer \
  --random-states 0 1 2 3 4 --epochs 25 \
  --output-dir results/v1/deep --summary-output results/v1/deep_baselines.csv

# 5. Sanity baselines (Table 2)
python scripts/run_quantum_sanity.py \
  --datasets WESAD --n-qubits 8 --n-tokens 8 --entanglement ring \
  --heads logreg rbf_svm --random-states 0 1 2 3 4 \
  --output results/quantum_sanity.csv

# 6. Quantum Fusion map sweep (Table 1, quantum rows)
python scripts/run_fusion_sweep.py \
  --datasets WESAD \
  --qubits-per-channel 2 3 4 --fusion-reps 1 2 3 \
  --entanglement minimal full --mixer-angle 0.0 0.4 \
  --heads ridge logreg rbf_svm --random-states 0 1 2 3 4 \
  --output results/v1/quantum_fusion_sweep.csv \
  --cost-output results/v1/quantum_fusion_sweep_costs.csv

# 7. Matched-input classical control
python scripts/run_matched_feature_control.py

# 8. Hardware (requires IBM Quantum Premium account)
python scripts/run_hardware_pilot.py --dataset WESAD --fold-idx 0 \
  --backend ibm_boston --hardware --shots 8192 --resilience-level 2 \
  --planned-qpu-min 8

# 9. Generate results figure
python scripts/make_results_figure.py
```

## Scripts

| Script | Purpose |
|---|---|
| `preprocess_ecg_eda.py` | Windowed ECG+EDA preprocessing (deferred scaling) |
| `extract_ecg_eda_features.py` | 27-feature HRV+EDA extraction |
| `quantum_models.py` | Circuit definitions (Fusion map, QRC) |
| `run_baselines_v1.py` | Classical baselines (RF, SVM, XGBoost, LR) |
| `run_deep_baselines.py` | 1D-CNN + Transformer on raw windows |
| `run_fusion_sweep.py` | Fusion feature-map simulator sweep |
| `run_quantum_sanity.py` | Sanity baselines (random features, ESN, QRC) |
| `run_matched_feature_control.py` | Classical control on the 6 encoded features |
| `run_hardware_pilot.py` | IBM hardware execution (EstimatorV2) |
| `make_results_figure.py` | Hardware ablation figure |
| `audit_*.py` | Data quality checks |

## Cross-validation

All experiments: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, `groups=subject_id`.

## Citation

```bibtex
@inproceedings{sokol2026hwqml,
  author    = {Sokol, Marek and Hejda, Jan and Volf, Petr and Tich\'{y}, Ale\v{s} and Kut\'{i}lek, Patrik},
  title     = {Hardware-Validated Multimodal Quantum Classifier for Biosignal Stress Recognition},
  booktitle = {Proc. IEEE Healthcom},
  year      = {2026}
}
```

## License

MIT
