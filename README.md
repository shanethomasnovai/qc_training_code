# Five-grade QC training

Adapted from the supplied ds-auto-qc CNN. Two runs use the same actual image files and the same patient split; only the training grader changes. Grades 1–5 map internally to class indices 0–4.

## Setup on your work Mac

Place the two original CSV files in a `data` folder beside these scripts:

- `data/multi_class_qc_dataset.csv`
- `data/shane-image-file-listing.csv`

Run the following from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Prepare the split once

```bash
python prepare_data.py --previous-student data/multi_class_qc_dataset.csv --shane data/shane-image-file-listing.csv --image-dir /Users/shanethomas/Desktop/images --held-out VISUSS
```

The `project` field must contain the exact trial name VISUSS, ignoring case. If not, the script lists the available trial names and stops. It uses patient and site metadata from the previous student's CSV. Existing `mode` values are replaced by the requested trial/patient split.

The script matches the full filename stem, preserving image variants. It defaults to PNG files. If your folder contains only TIFFs, explicitly pass `--image-extension .tif`; both graders will then use the same TIFF file for a matched identifier. Confirm that corresponding PNG and TIFF files represent the same image content before using their labels interchangeably. Duplicate stems and missing images cause errors.

- Test: matched images from VISUSS.
- Validation: approximately 20% of the remaining patients.
- Training: the remaining patients.

Both eyes, visits and image variants stay together by `patient_id`. Any non-VISUSS images belonging to a VISUSS patient are excluded. All VISUSS patients from the previous student's CSV are protected, including patients without matched images.

Candidate patient splits are assessed for site representation and both graders' grade distributions, using only the non-test pool. Exact balance is not guaranteed. Inspect `prepared/split_report.json` for patient counts, image counts, grade counts, excluded images and unmatched identifiers. Both training runs require examples of all five grades in training. Do not alter the split after reviewing test performance. Preparation refuses to overwrite an existing manifest.

Only the images present in both CSVs are used for this controlled comparison. The manifest is local to this machine because its image paths are absolute.

## Short check

```bash
python train.py --grader shane --epochs 1 --output-dir runs/shane_check
```

This trains for one epoch and checks validation; VISUSS performance is not calculated. It may take time on a CPU. Automatic device selection uses CUDA, then Apple MPS, then CPU. If MPS operations fail, add `--device cpu`.

## Full runs

Agree training settings before final test evaluation. For example:

```bash
python train.py --grader previousStudent --epochs 50 --output-dir runs/previousStudent --evaluate-test
python train.py --grader shane --epochs 50 --output-dir runs/shane --evaluate-test
```

The default image batch size is four (36 patches). Reduce memory use with `--batch-size 1` in both runs. The optimiser and CNN hidden layers retain the supplied approach; the output is changed to five logits, without sigmoid. Training uses nine 256×256 grayscale patches from each resized 768×768 image, created after splitting, with random horizontal flips in training. This assumes each patch can inherit the whole image's grade; assess this assumption when interpreting results.

Validation and test predictions average the nine patch probability vectors into one image prediction. Early stopping uses image-level validation loss. Class weighting is disabled by default; `--class-weighting balanced` computes training-only weights in grade order. If enabled, both runs should use that setting, although the resulting weights depend on each grader's label distribution.

Outputs include the best checkpoint, run configuration and manifest hash, learning curve, validation metrics, and (with `--evaluate-test`) VISUSS image predictions, five-by-five confusion matrices and grade metrics. Test metrics include exact accuracy, within-one-grade accuracy, mean absolute error and quadratic weighted kappa.

Each model is evaluated against BOTH graders' VISUSS labels. This makes cross-grader comparisons possible but neither grader is an independent reference standard. Do not claim one model is more accurate solely because it matches its own training grader better. A shared independent test grading is needed for that claim. The 39 disagreement cases are a selected difficult-case sample rather than an unbiased test set.

Seeds are fixed for both runs; exact reproducibility across hardware is not guaranteed. Fresh output directories are required so runs cannot overwrite one another. Local datasets, manifests, outputs and checkpoints are excluded by `.gitignore`.

## Verification

Preparation has been checked on synthetic CSVs for patient separation, trial holdout, repeatable splits, missing-image failures and unknown-trial failures. Python syntax has been checked. Full training has not been executed here: this environment does not have PyTorch, the work Mac's image folder or its CSVs.
