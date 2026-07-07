# Chest X-Ray Multi-Label Classification with Calibration Analysis

A DenseNet-121 classifier for three thoracic findings on NIH ChestX-ray14, with a focus on
calibration and failure analysis rather than headline AUC alone. The central finding:
class-weighted training produces strong ranking but severely miscalibrated probabilities, and
the standard fix (temperature scaling) cannot correct it — while Platt scaling can, at no cost
to AUC.

## Findings at a glance

- **Discrimination**: Test AUC of 0.84 (Pneumothorax), 0.82 (Effusion), 0.89 (Cardiomegaly) on
  the official patient-disjoint test split.
- **Calibration**: Training with `pos_weight` inflated predicted probabilities to 4–5× the base
  rate. Temperature scaling failed to correct this (ECE unchanged or worse); per-class Platt
  scaling reduced ECE by ~10× (e.g. Effusion 0.35 → 0.006) with AUC provably unchanged.
- **Failure taxonomy**: An evidence-based (non-diagnostic) analysis of the model's 90
  highest-confidence errors, using patient history and label co-occurrence rather than visual
  reads. A concrete label-noise case is documented where the model tracks a stable signal
  across a patient's visits while the NIH label intermittently drops the finding.

## Dataset

NIH ChestX-ray14: 112,120 frontal chest X-rays, 30,805 patients, pre-resized to 224×224.
Target findings: Pneumothorax, Effusion, Cardiomegaly (prevalence ~5%, ~12%, ~2.5% overall —
heavily imbalanced).

The official patient-level split (`train_val_list` / `test_list`) is used and verified
patient-disjoint, so no patient appears in both training and test. A 90/10 patient-disjoint
validation split is carved from train_val for checkpoint selection and calibration fitting.

Known data limitations addressed in this project:
- Labels are NLP-extracted from radiology reports, not radiologist-verified — a documented
  source of label noise, central to the failure analysis.
- 16 records carry impossible ages (>100, up to 414) from a known metadata overflow bug; these
  are filtered.
- 27 records encode sub-year ages in months/days; these are converted to fractional years
  rather than dropped.

## Method

- **Model**: DenseNet-121 pretrained on ImageNet, final classifier replaced with a 3-way
  multi-label head, full fine-tuning. Single-channel X-rays expanded to 3 channels for the
  pretrained weights.
- **Training**: `BCEWithLogitsLoss` with per-class `pos_weight` (negatives/positives) to handle
  imbalance. AdamW, LR 1e-4, 8 epochs, best checkpoint by validation macro-AUC (epoch 7, val
  macro-AUC 0.887). Apple Silicon MPS.
- **Calibration**: Post-hoc per-class temperature scaling and Platt scaling, both fit on
  validation logits only and applied to test.
- **Evaluation**: Per-class test AUC, ECE (10-bin), reliability diagrams, and subgroup AUC + ECE
  by sex and age bucket, with a minimum-positive-count guard (n_pos ≥ 10) so thin subgroups
  report NA rather than noise.

## Results

### Discrimination (test set, n = 25,592)

| Finding      | Test AUC |
|--------------|----------|
| Pneumothorax | 0.840    |
| Effusion     | 0.823    |
| Cardiomegaly | 0.885    |

Performance is stable across sex and across the well-populated age buckets. The 80+ bucket
(n=252) is underpowered and its numbers are reported but flagged as noisy rather than
interpreted.

### Calibration: the main finding

Weighted-loss training left the model badly miscalibrated — predicted probabilities for true
negatives averaged 4–5× the base rate (e.g. Cardiomegaly negatives averaged ~20% predicted
probability against a ~4% base rate).

| Finding      | ECE raw | ECE temperature | ECE Platt |
|--------------|---------|------------------|-----------|
| Pneumothorax | 0.311   | 0.322            | 0.051     |
| Effusion     | 0.347   | 0.341            | 0.006     |
| Cardiomegaly | 0.178   | 0.196            | 0.018     |

**Why temperature scaling fails and Platt scaling works**: the miscalibration here is a
systematic upward shift in probabilities, not uniform overconfidence. Temperature scaling has
only a scale parameter — it shrinks a sigmoid symmetrically toward 0.5 and cannot translate the
distribution downward. Platt scaling adds an intercept (`sigmoid(a·logit + b)`); the fitted
intercepts were strongly negative (b ≈ −2.3 to −3.6), which is exactly the shift the
distribution needed. AUC is unchanged under both (fitted a > 0 keeps the transform monotonic —
verified to ~1e-8).

Reliability diagrams (raw / temperature / Platt) for each class are in `figures/`.

Per-subgroup ECE (sex, age) is in `results/eval_results.csv`. A couple of cells show elevated
ECE from small-sample noise rather than systematic miscalibration — most visibly the 80+ age
bucket (n=252, n_pos as low as 18) — treat these as thin-cell variance, not a subgroup effect.

### Failure taxonomy

The model's 90 highest-confidence errors (15 false positives + 15 false negatives per class,
ranked on calibrated probabilities) were categorized using patient history and label
co-occurrence — not visual diagnosis (neither the author nor the tooling is a radiologist).

- **Candidate label noise — chronic finding not restated (30 cases)**. High-confidence false
  positives where the same patient is positive for the finding in other scans. Consistent with
  ChestX-ray14's per-report labeling, where chronic findings are often not restated at each
  follow-up visit.
- **Co-occurrence / entangled-label confusion (27 cases)**. Errors that track a correlated
  finding (Effusion↔Cardiomegaly, Emphysema↔Pneumothorax). Notably, 9 of 15 Cardiomegaly false
  negatives co-list Effusion — one hypothesis (untested) is that shared visual features draw
  attention away from cardiomegaly when effusion dominates.
- **Genuine hard case (26 cases)**. False negatives with no corroborating history or
  co-occurrence story — treated at face value as legitimately subtle presentations.
- **Ambiguous / unexplained (7 cases)**. Errors not attributable to any of the above under the
  evidence rules, reported honestly as open.

**Illustrative case — patient 00000211**: 44 scans, Cardiomegaly-positive in 26. The model's
cardiomegaly probability stays in a stable ~0.4–0.7 band across follow-up visits regardless of
whether that visit's report restates "Cardiomegaly," including later visits where the label
shifts to Effusion / No Finding / Fibrosis. This is consistent with the model tracking a stable
signal across visits while the label intermittently drops the term — a concrete illustration of
the label-noise mechanism, not a claim about the patient's actual condition.

## Limitations

- **In-distribution calibration**. Calibration is fit and evaluated within ChestX-ray14 (same
  institution/scanners). Calibration is not guaranteed to transfer to other hospitals or
  imaging equipment.
- **Label-noise analysis is heuristic**. The taxonomy identifies cases consistent with label
  noise using patient-history proxies; it does not prove any individual label is incorrect.
  Conditions can genuinely change between visits.
- **No radiologist review**. All failure analysis is evidence-based (labels, patient history,
  co-occurrence), never a diagnostic read of the images.
- **Underpowered subgroups**. The 80+ age group is too small (n=252) for reliable subgroup
  metrics.
- **ECE binning**. ECE uses fixed 10-bin equal-width binning, which is noisier for low-prevalence
  classes; adaptive binning would be a robustness check.

## Repository

| File                | Purpose                                                         |
|---------------------|------------------------------------------------------------------|
| `data.py`           | Dataset, official split, label encoding, age handling            |
| `model.py`          | DenseNet-121 with 3-way multi-label head                         |
| `train.py`          | Training loop (weighted BCE, MPS, best-checkpoint saving)        |
| `evaluate.py`       | Test AUC, ECE, reliability diagrams, subgroup analysis           |
| `calibrate.py`      | Temperature and Platt scaling (fit on val, applied to test)      |
| `failure_analysis.py` | Surfaces highest-confidence errors for the taxonomy            |

Environment: Python 3.10, PyTorch, Apple Silicon MPS. The dataset and checkpoints are
gitignored.

### Reproduction

```bash
conda activate cxr

# Train (8 epochs used for the reported checkpoint; default is 10)
python3 train.py --epochs 8

# Evaluate: test AUC, calibration (raw/temperature/Platt), subgroup breakdowns
python3 evaluate.py

# Surface the 90 highest-confidence failure cases + manifest for the taxonomy
python3 failure_analysis.py
```

All three scripts default to `checkpoints/best_model.pt`; pass `--checkpoint <path>` to point at
a different one. `train.py` also supports `--subset N` for a fast smoke test on a small slice of
data before committing to a full run.
