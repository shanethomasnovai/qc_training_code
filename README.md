# Five-grade QC training

Adapted from the supplied ds-auto-qc CNN. Two runs use the same actual image files and the same patient split; only the training grader changes. Grades 1–5 map internally to class indices 0–4.

The script matches the full filename stem, preserving image variants. It defaults to PNG files. If your folder contains only TIFFs, explicitly pass `--image-extension .tif`; both graders will then use the same TIFF file for a matched identifier. Confirm that corresponding PNG and TIFF files represent the same image content before using their labels interchangeably. Duplicate stems and missing images cause errors.

- Test: matched images from VISUS-NHP.
- Validation: approximately 20% of the remaining patients.
- Training: the remaining patients.

Both eyes, visits and image variants stay together by `patient_id`. Any non-VISUS images belonging to a VISUS patient are excluded. All VISUS patients from the previous student's CSV are protected, including patients without matched images.

## Full runs

Agree training settings before final test evaluation. For example:

```bash
python train.py --grader previousStudent --epochs 50 --output-dir runs/previousStudent --evaluate-test
python train.py --grader shane --epochs 50 --output-dir runs/shane --evaluate-test
```

## Verification

Preparation has been checked on synthetic CSVs for patient separation, trial holdout, repeatable splits, missing-image failures and unknown-trial failures. Python syntax has been checked. Full training has not been executed here: this environment does not have PyTorch, the work Mac's image folder or its CSVs.
