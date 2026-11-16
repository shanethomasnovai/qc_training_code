# Dataset preparation and handover

For normal training, edit [samples/config.yaml](samples/config.yaml). Its
`local_root` points to your local `images/`, `labels/`, `datasets/` and `models/`
folders. You do not rebuild the datasets for every training run.

## What the generator does

[tools/generate_datasets.py](tools/generate_datasets.py) follows four steps:

1. Read the recipe in [tools/datasets.yaml](tools/datasets.yaml).
2. Check the three source files against their recorded SHA-256 hashes.
3. Join the reviewed ratings to the saved split by the complete image ID.
4. Write five label workbooks and one `manifest.json` describing their sources.

The split CSV supplies image identity and patient assignments. Ganaav's CSV and
the reviewed ratings workbook supply the grades. Grades carried in the split CSV
are ignored; Shane's grades come only from the current reviewed workbook.

The generator never makes a new patient split. Missing files, changed source
hashes, invalid grades, duplicate images and patient overlap between splits cause
an error. Only unmatched image IDs explicitly listed in the recipe are allowed.

## Output files

Each workbook has one `labels` sheet and starts with `image_filename`.

| Workbook | Remaining columns / rule |
|---|---|
| `ganaav_labels.xlsx` | `label`: Ganaav's integer grade |
| `shane_labels.xlsx` | `label`: Shane's integer grade |
| `shane_ganaav_average_labels.xlsx` | `label`: exact mean of Ganaav and Shane; both required |
| `shane_ganaav_average_rounded_labels.xlsx` | `label`: that mean rounded **up**, using ceiling |
| `multigrader_labels.xlsx` | One column per grader; missing ratings stay blank |

The manifest records source hashes, grader names, label rules and workbook
hashes. Images remain in their original folder.

## Check or rebuild

Run these commands from the project root. Verify the datasets under your
configured `local_root` without changing them:

```powershell
.\.venv\Scripts\python.exe -m data.tools.generate_datasets --config data/samples/config.yaml --verify-existing
```

Rebuild into a **new** folder, replacing the example output path as needed:

```powershell
.\.venv\Scripts\python.exe -m data.tools.generate_datasets --config data/samples/config.yaml --output-dir C:/Users/Shane/Desktop/qc_data/rebuilt_datasets
```

The normal training config supplies `local_root` and the profile name. The build
recipe still supplies the approved source hashes and unmatched-image exceptions.
Existing output folders are never overwritten. For a deliberately updated ratings
source, review its changes and update its hash in `tools/datasets.yaml` before
building a new version. Preserve the original split CSV.

## Supporting code

- [labels.py](labels.py): shared validation, target encoding and workbook loading.
  Training derives ordinal probabilities, vote distributions or exact means here.
- [tools/generate_baseline_split.py](tools/generate_baseline_split.py): optional
  baseline/non-baseline selections, separate from the train/validation/test split.
- [Historical rerun code](../auto_qc/CNN/rerun.py): the older queue's eight-workbook
  builder, separate from the normal five-table generator.

Older data and review records remain under `model_dir/dataset_support/legacy_data/`.
# Experiment log

Each completed local training run updates `qc_data/experiment_log.xlsx` automatically.
The workbook has one filterable row per run, with its UTC run date, architecture,
head, dataset, hyperparameters, training history summary and validation metrics.
Binary-head and grade-derived binary results have separate columns. Missing older
values stay blank. Historical implementations and diagnostic runs are labelled;
their results may use different datasets or evaluation rules. The `Notes` column
is preserved when rows are refreshed.

To import previous runs or refresh the log after closing it in Excel:

```powershell
python -m auto_qc.CNN.experiment_log --local-root C:/Users/Shane/Desktop/qc_data
```

This scans the local `models` folder and the original project `runs` and `model_dir`
folders, excluding support copies. Existing rows are updated without duplication.
Keep Excel closed while training finishes so the log can be saved. If Excel locks
the workbook, the model output is still saved and the refresh command recovers
the missing entry. The workbook lives outside the seven-file run folders.

