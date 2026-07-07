# Chest X-Ray Multi-Label Classification with Calibration Analysis

A DenseNet-121 classifier for three thoracic findings on NIH ChestX-ray14. The emphasis is on
calibration and failure analysis, not just AUC. The short version: class-weighted training
gives you strong ranking but badly miscalibrated probabilities. Temperature scaling, the usual
fix, doesn't help. Platt scaling does, and it costs nothing in AUC.

## Summary

Test AUC lands at 0.84 for Pneumothorax, 0.82 for Effusion, 0.89 for Cardiomegaly, on the
official patient-disjoint split.

The interesting part is the calibration. Training with `pos_weight` pushed predicted
probabilities to 4–5× the base rate. Temperature scaling left that basically unchanged.
Per-class Platt scaling cut ECE by roughly 10× (Effusion went from 0.35 to 0.006) without
moving AUC.

There's also a failure taxonomy: an evidence-based walk through the model's 90
highest-confidence errors, using patient history and label co-occurrence instead of reading the
films. One patient's history turns out to be a clean illustration of label noise, where the
model holds a steady prediction across visits while the NIH label keeps dropping and re-adding
the finding.

## Dataset

NIH ChestX-ray14: 112,120 frontal chest X-rays from 30,805 patients, pre-resized to 224×224.
The three target findings (Pneumothorax, Effusion, Cardiomegaly) sit at roughly 5%, 12%, and
2.5% prevalence, so the classes are heavily imbalanced.

I used the official patient-level split and checked that it's actually patient-disjoint (no
patient in both train and test). The validation set is a 90/10 patient-disjoint carve-out of
`train_val`, used for checkpoint selection and for fitting the calibration.

A few data issues I had to handle:
- The labels are NLP-extracted from radiology reports, not checked by radiologists. That's a
  known source of noise, and it's what the failure analysis ends up circling back to.
- 16 records have impossible ages (over 100, one at 414) from a documented metadata bug.
  Filtered out.
- 27 records store sub-year ages in months or days. I converted these to fractional years
  instead of dropping them.

## Method

DenseNet-121 pretrained on ImageNet, classifier swapped for a 3-way multi-label head, fully
fine-tuned. The X-rays are single-channel and get expanded to 3 channels to match the pretrained
weights.

Loss is `BCEWithLogitsLoss` with per-class `pos_weight` (negatives over positives) for the
imbalance. AdamW at 1e-4, 8 epochs, keeping the best checkpoint by validation macro-AUC. I
stopped at 8 because the val curve started overfitting there: epoch 8's train loss kept
dropping while val loss ticked back up, and the best checkpoint landed at epoch 7 (val
macro-AUC 0.887). All on Apple Silicon MPS, which is slow but works.

Calibration is post-hoc, both temperature and Platt scaling, fit on validation logits and
applied to test. Evaluation covers per-class test AUC, ECE (10-bin), reliability diagrams, and
subgroup AUC and ECE by sex and age. Subgroups with fewer than 10 positives report NA instead of
a noisy number.

## Results

Discrimination on the test set (n = 25,592):

| Finding      | Test AUC |
|--------------|----------|
| Pneumothorax | 0.840    |
| Effusion     | 0.823    |
| Cardiomegaly | 0.885    |

Performance holds up across sex and across the age buckets that have enough data. The 80+
bucket (n=252) is too small to read into, so I report it but don't lean on it.

### Calibration

This is where the model looks worse than the AUC suggests. Weighted-loss training left it
badly miscalibrated. True negatives for Cardiomegaly averaged around 20% predicted probability
against a base rate near 4%, and the other two classes show the same inflation.

| Finding      | ECE raw | ECE temperature | ECE Platt |
|--------------|---------|------------------|-----------|
| Pneumothorax | 0.311   | 0.322            | 0.051     |
| Effusion     | 0.347   | 0.341            | 0.006     |
| Cardiomegaly | 0.178   | 0.196            | 0.018     |

The reason temperature scaling can't fix this comes down to what the miscalibration actually
is. It's a systematic upward shift, and temperature scaling only has a scale parameter, so it
can pull a sigmoid toward 0.5 but can't slide the whole distribution down. Platt scaling adds an
intercept via `sigmoid(a·logit + b)`, and the fitted intercepts came out strongly negative (b
around −2.3 to −3.6), which is exactly the downward shift the probabilities needed. Since every
fitted a is positive, the transform is monotonic and AUC is unchanged.

Per-subgroup ECE is in `results/eval_results.csv`. A few cells run high on small-sample noise,
most obviously the 80+ bucket (n=252, as few as 18 positives), so those are thin-cell variance
rather than a real subgroup effect. Reliability diagrams for all three classes, raw and both
scalings, are in `figures/`.

### Failure taxonomy

I pulled the model's 90 highest-confidence errors (15 false positives and 15 false negatives
per class, ranked on the calibrated probabilities) and sorted them using patient history and
label co-occurrence. I'm not a radiologist and neither is the tooling, so none of this rests on
reading the actual images.

Four groups came out of it:

- **Candidate label noise, 30 cases.** High-confidence false positives where the same patient
  is positive for the finding in their other scans. This fits how ChestX-ray14 was labeled,
  per-report, where a chronic finding often isn't restated at every follow-up.
- **Co-occurrence confusion, 27 cases.** The error tracks a correlated finding instead (Effusion
  with Cardiomegaly, Emphysema with Pneumothorax). One thing I noticed: 9 of the 15 Cardiomegaly
  false negatives also carry an Effusion label. My guess is the shared visual features pull
  attention toward effusion, but I didn't test that, so it's just a guess.
- **Genuine hard cases, 26 cases.** False negatives with no supporting patient history and no
  co-occurrence story. I took these at face value as subtle presentations.
- **Ambiguous, 7 cases.** Errors I couldn't attribute to any of the above under my own rules.
  Left open.

The clearest single example is patient 00000211: 44 scans, Cardiomegaly-positive in 26 of them.
The model's cardiomegaly probability stays in a steady 0.4–0.7 band across the follow-ups
whether or not that visit's report restates "Cardiomegaly," including later visits where the
label switches to Effusion, No Finding, or Fibrosis. The model seems to be tracking something
stable across the series while the label comes and goes. I can't say what it's tracking, and
this isn't a claim about the patient's actual heart, but it's a tidy picture of the label-noise
problem.

## Limitations

- The calibration is in-distribution. It's fit and tested inside ChestX-ray14, same institution
  and scanners, so it won't necessarily hold on a different hospital's images.
- The label-noise analysis is a heuristic. It flags cases that look consistent with label noise
  from patient history. It doesn't prove any single label is wrong, and findings really can
  change between visits.
- There's no radiologist review anywhere in this. Everything in the failure analysis comes from
  labels, patient history, and co-occurrence.
- The 80+ age group is too small to support subgroup metrics.
- ECE uses fixed 10-bin equal-width binning, which gets noisy on the low-prevalence classes.
  Adaptive binning would be a better robustness check, and I didn't get to it.

## Repository

| File                  | Purpose                                                    |
|-----------------------|-------------------------------------------------------------|
| `data.py`             | Dataset, official split, label encoding, age handling      |
| `model.py`            | DenseNet-121 with 3-way multi-label head                   |
| `train.py`            | Training loop (weighted BCE, MPS, best-checkpoint saving)  |
| `evaluate.py`         | Test AUC, ECE, reliability diagrams, subgroup analysis     |
| `calibrate.py`        | Temperature and Platt scaling (fit on val, applied to test)|
| `failure_analysis.py` | Surfaces highest-confidence errors for the taxonomy        |

Python 3.10, PyTorch, Apple Silicon MPS. Dataset and checkpoints are gitignored.

### Reproduction

```bash
conda activate cxr

python3 train.py --epochs 8      # default is 10; I used 8, see the note above
python3 evaluate.py              # test AUC, calibration, subgroup breakdowns
python3 failure_analysis.py      # surfaces the 90 failure cases + manifest
```

All three default to `checkpoints/best_model.pt`; use `--checkpoint <path>` for another.
`train.py` takes `--subset N` for a quick smoke test. Metrics will drift a little between runs
from training randomness.
