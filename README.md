# Multiclass AutoQC with two graders

Actual inputs:

- PreviousStudent: `data/multi_class_qc_dataset.csv` (not `multi_class_qc_data.csv`).
- Shane: `data/shane-image-file-listing.csv`.
- Images: `C:\Users\Shane\Desktop\images`.
- Shared saved split: `prepared_visus_nhp/manifest.csv`.

The original CSV already has patient splits, but 18 VISUS-NHP images are in
training and two in validation. Its copies in Downloads are identical. No valid
separate split was found in the searched project, Documents, Downloads or Desktop
directories. The original data and existing results remain untouched.

Preparation matched all 661 previousStudent identifiers to Shane labels and PNG
files by complete filename stem, retaining variant identifiers. The previousStudent
filenames end in `.tif`; both runs use the corresponding actual PNG. Identifier
matching and image readability were verified; equivalence to original TIFF pixels
was not checked. Eight extra Shane-only VISUS-NHP variants are excluded from the
matched comparison and never enter training or validation.

| Split | Images | Patients |
| --- | ---: | ---: |
| Train | 508 | 78 |
| Validation | 127 | 20 |
| Test: VISUS-NHP | 26 | 9 |

Use the same manifest for both runs. Both eyes, visits and variants stay together
by patient; patches inherit this assignment and flips are added only to training.
Every VISUS-NHP patient in the source labels is protected from training and
validation. Test images are not decoded during training. All five grades occur
in training and validation for both graders. Exclusions and counts are recorded
in `prepared_visus_nhp/split_report.json`.

Grades 1–5 map to classes 0–4; loss weights explicitly follow this order. Best
epoch weights are deep-copied, and `model.pth` remains a state dictionary.
Existing model folders and preparation outputs cause errors instead of being
overwritten. The model architecture and image preprocessing are unchanged.

Every completed training run now reloads its best saved checkpoint and evaluates
whole validation images. Each image's prediction is the argmax of the mean of
its nine patch softmax probability vectors; class indices 0–4 map to grades 1–5.
The run folder receives `validation_metrics.json`,
`validation_per_grade_metrics.csv`, `validation_confusion_matrix.csv`, and
`validation_predictions.csv`, followed by a printed summary. The existing patch
validation loss still controls best-epoch selection and early stopping.

To evaluate an existing saved run without retraining:

```powershell
.\.venv\Scripts\python.exe .\validation.py --run-dir .\runs\shane_20261013_121344_539_933fb0f3 --grader shane
```

Evaluation uses that run's saved manifest and model parameters. New runs also
save grader metadata, so `--grader` can be omitted for them. Only validation
images are opened. Existing validation output files cause an error to preserve
results. Undefined precision, recall and F1 are zero, with per-grade defined
flags; macro F1 averages over all five grades. Kappa is JSON `null` if expected
disagreement is zero. Accuracy and F1 are fractions, not percentages. Confusion
matrix rows are reference grades and columns are predicted grades.

## Exact one-epoch checks

Run in PowerShell from this project:

```powershell
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader previousStudent --output_dir .\runs --model_name previousStudent_one_epoch_check --epochs 1 --batch_size 16 --optimizer AdamW --seed 42
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader shane --output_dir .\runs --model_name shane_one_epoch_check --epochs 1 --batch_size 16 --optimizer AdamW --seed 42
```

Each runs one full training/validation epoch. Choose a new model name if that
output already exists. The existing environment has CPU-only PyTorch. No actual
training epoch was run during setup verification.

## Preparation and verification

The manifest is already prepared. To create another in a new output directory:

```powershell
.\.venv\Scripts\python.exe .\prepare_inputs.py --previous-student .\data\multi_class_qc_dataset.csv --shane .\data\shane-image-file-listing.csv --image-dir C:\Users\Shane\Desktop\images --output-dir .\prepared_visus_nhp_new
```

Preparation reuses source patient assignments if valid. Otherwise it creates a
deterministic split with seed 42 and approximately 20% validation patients,
balancing both graders and sites. `--split-file` accepts an existing manifest or
label CSV with patient assignments; an invalid explicit split fails. For example,
use `--split-file .\prepared_visus_nhp\manifest.csv` to reuse the saved split.

```powershell
.\.venv\Scripts\python.exe -m unittest test_setup -v
```

Three checks verify actual PNG readability and patient groups, reuse with a
different seed, and weight order plus best-epoch preservation for both graders.
Epoch updates are mocked; these checks do not train AutoQC or save model results.
The supplied `testing.py` still requires its legacy label/mode CSV format; final
test evaluation is outside this setup change.

Run `python -m unittest test_setup test_validation -v` for the complete setup and
evaluation regression checks, including equivalence to existing nine-patch
preprocessing, probability averaging across batch boundaries, image-ID alignment,
train/test image exclusion and undefined metrics.
