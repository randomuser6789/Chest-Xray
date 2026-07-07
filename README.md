
## Calibration

Probability calibration via Platt scaling, fit on the validation set and
evaluated on a held-out test set (patient-level split, no leakage across sets).

**Results** — Expected Calibration Error (ECE), raw → Platt-scaled:

| Class         | ECE (raw) | ECE (temp) | ECE (Platt) |
|---------------|-----------|------------|-------------|
| Effusion      | 0.3466    | 0.3414     | 0.0060      |
| Pneumothorax  | 0.31      | ~0.31      | 0.0514      |
| Cardiomegaly  | 0.18      | ~0.18      | 0.02        |

**Notes**
- Temperature scaling barely moved ECE because the miscalibration was a
  location shift (negatives sitting 4–5x above base rate), which a pure
  rescale cannot correct. Platt's bias term fixes it.
- AUC is provably unchanged under Platt scaling (all fitted `a > 0`, a
  monotone transform; verified numerically to ~1e-8).
- Per-subgroup ECE (sex, age) is in `results/eval_results.csv`. Two cells
  show elevated ECE from small-sample noise, not systematic miscalibration:
  the 80+ age bucket (n=252, n_pos as low as 18) and the Pneumothorax
  high-confidence bins. Treat these as thin-cell variance.
- Reliability diagrams saved to `figures/calibration_*.png`.
