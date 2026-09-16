# Benchmark results

These results were measured on 2026-09-16 on an NVIDIA GeForce RTX 5090. Each
model was evaluated separately for each of the nine subjects using stratified
five-fold cross-validation with seed 42. The unit of generalization is a new
trial from the same subject and training session; these numbers do not measure
cross-session or cross-subject transfer.

| Model | Mean accuracy | SD across subjects | Best subject | Worst subject |
|---|---:|---:|---:|---:|
| CSP + shrinkage LDA | 71.54% | 17.66% | S3: 86.96% | S5: 45.37% |
| CSP + linear SVM | **77.21%** | 17.83% | S3: 88.01% | S5: 46.39% |
| EEGNet (100 epochs) | 62.39% | 19.04% | S9: 81.42% | S2: 38.82% |

The chance level is 25%. CSP-SVM was the strongest mean baseline in this run.
EEGNet did not outperform the classical pipelines, which is plausible for only
288 trials per subject without augmentation or additional calibration data.
Subject-level values are preserved in `benchmark_results.csv`; reporting only
the aggregate mean would hide substantial between-subject variability.
PyTorch deterministic algorithms were enabled; two consecutive A01 reruns
produced identical fold summaries.

Commands:

```bash
python -m bci_2a.experiment --download --data-root data \
  --output results/classical --models csp-lda csp-svm --folds 5

CUDA_VISIBLE_DEVICES=2 python -m bci_2a.experiment --data-root data \
  --output results/eegnet --models eegnet --folds 5 --eegnet-epochs 100
```
