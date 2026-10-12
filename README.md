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

## Full runs

Agree training settings before final test evaluation. For example:

```bash
python train.py --grader previousStudent --epochs 50 --output-dir runs/previousStudent --evaluate-test
python train.py --grader shane --epochs 50 --output-dir runs/shane --evaluate-test
```

## Verification

Preparation has been checked on synthetic CSVs for patient separation, trial holdout, repeatable splits, missing-image failures and unknown-trial failures. Python syntax has been checked. Full training has not been executed here: this environment does not have PyTorch, the work Mac's image folder or its CSVs.
