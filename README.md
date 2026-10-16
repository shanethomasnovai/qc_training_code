# Multiclass AutoQC with single-grader and soft-label training

## Combined-reference validation

New soft-label runs automatically report these whole-image metrics together
against the fractions of available grader ratings:

- `earth_movers_distance`: Wasserstein-1, the sum of absolute cumulative
  probability differences at the four grade boundaries. Units are grade steps,
  range 0 to 4; lower is better. This is not divided by four or five.
- `soft_cross_entropy`: `-sum(target * ln(predicted_probability))`, averaged
  over images. Lower is better; a perfectly matched ambiguous distribution has
  nonzero cross-entropy. Compare models on the same reference distribution.
- `matches_any_grader_accuracy`: predicted grade equals at least one available
  selected grader rating. This is exact agreement, not within-one agreement.
  A minority rating counts; an unobserved grade between two ratings does not.
- `consensus_reference_exact_accuracy`, `consensus_reference_mae`, and
  `consensus_reference_quadratic_weighted_kappa`: prediction compared with the
  most frequent available rating. Kappa uses fixed grade order 1-5 and is JSON
  `null` with an explicit reason when expected disagreement is zero.

**Reference grades are the most frequent available grader rating; ties are resolved in favour of the lower grade.**
This is an explicit evaluation convention, not an adjudicated ground truth.
Integer vote counts identify ties; neither mean/median ratings nor rounded
probabilities determine the reference. Examples: 2/3/3 gives 3; 2/3 gives 2;
2/2/4/4 gives 2; 1/3/3/5/5 gives 3; 4/blank/blank gives 4.
Every single-grade metric uses the existing prediction: argmax of the mean of
nine patch probability vectors, mapped to grades 1-5.

Each image has equal base weight irrespective of rating count. No class weights
are applied to these evaluation metrics, even if training uses weighted loss.
Probability rows within 1e-5 of unit sum are normalised for rounding drift;
invalid distributions fail. Zero predicted probabilities use the recorded
float32 minimum-positive-normal floor before taking logs.

Missing ratings are ignored. Images with only the original two graders remain
eligible. No metric weights images by their number of ratings.

Soft runs add `distribution_metrics.json` and `distribution_per_image.csv`,
and include the combined-reference summary in `validation_metrics.json`. Summary
fields `reference_tie_count` and `reference_tie_percentage` report how many scored
images needed tie-breaking (percentage is on a 0-100 scale). Per-image outputs
contain `predicted_grade`, `consensus_reference_grade`, `reference_tied`,
`rating_count`, `matches_any_grader`, `consensus_exact_match`, and
`absolute_grade_error`, alongside original selected ratings, probabilities,
soft targets, EMD and soft cross-entropy. The automatic
`validation_predictions.csv` also includes the new columns. Individual-grader
reports remain available as secondary diagnostics. Model architecture, training
loss, early stopping and checkpoint selection are unchanged.

To compare saved hard and soft models fairly, use one completed soft run's
immutable `targets.csv` as the reference for every model. No retraining or image
decoding is needed. For the completed runs in this project:

```powershell
.\.venv\Scripts\python.exe .\score_distributions.py --reference-run .\runs\soft_20261016_122507_325_f156d443 --run-dir .\runs\previousStudent_20261013_130615_231_f9f4cb56 .\runs\shane_20261013_121344_539_933fb0f3 .\runs\soft_20261016_122507_325_f156d443 --output-dir .\runs\combined_reference_comparison_20261016_repeat
```

Choose a fresh output directory if that report already exists. This writes one
`comparison.csv` and a JSON summary and per-image CSV under each run name inside
the report directory, plus copies of the common `reference_targets.csv` and
`reference_config.json`. Original run files are never overwritten. Hashes identify
the common target snapshot and each model's saved predictions. Image IDs are
joined explicitly; all models must supply predictions for the same rated
validation images and have identical image populations, patients and splits.
Training/test predictions and conflicting prediction split fields are rejected.
Saved predicted grades must agree with the probability argmax. Additional
reviewers rated selected disagreements only;
these targets describe the observed panel, not an objective clinical truth.

The refreshed report in `runs/combined_reference_comparison_20261016` scores the
same 127 validation images for all three models. 63 images (49.6063%) need the
lower-grade tie rule. EMD and soft cross-entropy are unchanged from the earlier
`runs/distribution_comparison_20261016` report.

| Model | EMD | Soft CE | Matches-any | Reference exact | Reference MAE | Reference QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| previousStudent | 0.760806 | 1.257941 | 64.57% | 49.61% | 0.551181 | 0.732412 |
| shane | 0.687918 | 1.102816 | 81.10% | 55.12% | 0.480315 | 0.694977 |
| soft | 0.654200 | 1.116846 | 57.48% | 41.73% | 0.716535 | 0.665670 |

Checks (no full training):

```powershell
.\.venv\Scripts\python.exe -m unittest test_distribution_metrics test_validation test_soft_labels test_weighted_training test_setup -v
```

Focused checks cover all reference examples, missing ratings, minority and
unobserved-grade matches-any cases, tie statistics, hand-calculated accuracy/MAE,
kappa against scikit-learn, row-order-independent joins, invalid/missing/duplicate
predictions and unchanged distribution metric values.
EMD is independently checked against SciPy's Wasserstein distance. References:
[NIMA](https://arxiv.org/abs/1709.05424) evaluates ordered human-rating distributions
with EMD; [Matton et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13268597/)
use empirical clinician-rating distributions for soft ordinal learning.

## Multiple graders with ordinary cross-entropy

Use `C:\Users\Shane\Documents\multigrader_ratings.xlsx`, worksheet `Grades`.
Keep one row per image. Keep full `image_id` values, including variant suffixes
such as `1.0_0`. Alternatively a CSV can use `image_filename`; only its final
extension is removed. The workbook is read without modifying it.

Add future grader columns to `Grades`, named `<name>_grade` (for example
`reviewer4_grade`). Every column ending in `_grade` is discovered on the next
run, excluding generated `target_grade_1` through `target_grade_5`. Reserve this
suffix for ratings; metadata must use other names. Current columns are
`previousStudent_grade`, `shane_grade`, `reviewer1_grade`, `reviewer2_grade`, and
`reviewer3_grade`. Leave pending ratings blank. CSV strings such as `NA` and
`NaN` are invalid ratings, not blanks. Nonblank ratings must be integer values
1 through 5. Formulas are rejected; enter original numeric ratings.

Choose a subset with `-GraderColumns previousStudent_grade,shane_grade,reviewer1_grade`
or Python `--grader_columns previousStudent_grade shane_grade reviewer1_grade`.
Use `-RatingsSheet` / `--ratings_sheet` if the sheet name changes.

Targets are the five fractions of available grades, in order 1, 2, 3, 4, 5.
For example, 2/3/3 becomes `[0, 1/3, 2/3, 0, 0]`; 2/blank/3 becomes
`[0, .5, .5, 0, 0]`. Targets remain floating point probability vectors.
An image with no selected ratings is counted by split, retained in the saved
target table with `has_ratings=False` and an all-zero sentinel, and excluded
from loss and evaluation. An entirely unrated training or validation split fails.

Soft mode uses raw five-class logits with **CrossEntropyLoss**, unweighted by
default or with an explicitly selected hard-baseline weight vector as below.
No weights are derived from argmax labels. Each rated image contributes nine
patches (plus the same nine horizontal flips during training), so each image
has equal total weight regardless of rating count. Single-grader modes retain
their inverse training-class-frequency weights and use those weights for both
training and patch validation loss. Their reports and checkpoints stay compatible.

## Weighted soft-label comparison and reproducibility

`--class_weights_from <hard-run-directory>` reuses that run's exact five weights
in grade order 1-5. The run's `run_config.json` is checked first. If it contains
weights, they are reused without rescaling. The two historical baselines below
did not save weights; their vectors are reconstructed from the **training rows
of their own saved manifest**, using the original formula `N / (5 * n_grade)`.
Nine patches and their flips multiply all counts by 18 and cancel in the ratio.
Validation/test ratings and new reviewer votes do not enter this calculation.
The current image population and patient split must match the weight baseline.
The source grader must agree with saved metadata. The older Shane baseline has
no saved grader field, so `--weight_reference_grader shane` is required for it.

| Baseline | Training counts, grades 1-5 | Weights, grades 1-5 (display rounded) |
| --- | --- | --- |
| previousStudent | 73, 163, 156, 112, 4 | 1.39178082, 0.62331288, 0.65128205, 0.90714286, 25.4 |
| shane | 56, 176, 181, 87, 8 | 1.81428571, 0.57727273, 0.56132597, 1.16781609, 12.7 |

Run from this project in PowerShell. Define the common arguments and the two
available references (these lines do not launch training):

```powershell
$previousBaseline = '.\runs\previousStudent_20261013_130615_231_f9f4cb56'
$shaneBaseline = '.\runs\shane_20261013_121344_539_933fb0f3'
$softArgs = @('--dataset_csv_path', '.\prepared_visus_nhp\manifest.csv', '--grader', 'soft', '--ratings_csv_path', 'C:\Users\Shane\Documents\multigrader_ratings.xlsx', '--exclude_known_unmatched', '--output_dir', '.\runs', '--epochs', '50', '--batch_size', '16', '--learning_rate', '0.001', '--optimizer', 'AdamW', '--patience', '10', '--seed', '42', '--reduce_lr')
```

Short input/weight checks for either reference; no training or run files:

```powershell
.\.venv\Scripts\python.exe .\training.py @softArgs --class_weights_from $previousBaseline --weight_reference_grader previousStudent --check_only
.\.venv\Scripts\python.exe .\training.py @softArgs --class_weights_from $shaneBaseline --weight_reference_grader shane --check_only
```

Choose the reference explicitly when launching a full run (choose a fresh model
name if the output exists):

```powershell
.\.venv\Scripts\python.exe .\training.py @softArgs --class_weights_from $previousBaseline --weight_reference_grader previousStudent --model_name soft_weighted_previousStudent_seed42
.\.venv\Scripts\python.exe .\training.py @softArgs --class_weights_from $shaneBaseline --weight_reference_grader shane --model_name soft_weighted_shane_seed42
```

For an unweighted run omit both weight options, for example:

```powershell
.\.venv\Scripts\python.exe .\training.py @softArgs --model_name soft_unweighted_seed42
```

The launcher supports the same choices with `-ClassWeightsFrom` and
`-WeightReferenceGrader`. No reference is chosen automatically.

Each image target is still its own vote fractions. Missing additional ratings
do not contribute votes or disagreement; images with only the original two
graders remain eligible. No image is repeated according to its rating count.
Weights multiply the five target components inside PyTorch CrossEntropyLoss;
neither the target fractions nor the weights are renormalised. Class weighting
can change an image's effective loss according to its grade distribution, but
repeating the same distribution with more graders leaves its loss unchanged.

With `reduction='mean'`, probability-target loss is
`sum_i sum_c (-w_c * q_ic * log(p_ic)) / batch_size`. Weighted hard-target loss
divides its weighted sum by `sum_i w[y_i]` instead. Therefore even one-hot soft
targets generally have a different weighted mean loss from hard targets; their
unreduced/summed losses agree. **Raw weighted hard/soft losses are not directly
comparable.** This also changes gradient scale. The existing hard-mode epoch
loss remains the sample-count-weighted average of the individual batch means,
rather than a newly introduced global weighted mean. See
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

Best-checkpoint selection and the scheduler continue to use patch validation
loss with the selected training weights; target type determines its denominator.
Existing whole-image `soft_cross_entropy` remains explicitly unweighted for
compatibility. Weighted soft runs additionally report
`weighted_soft_cross_entropy` on the same mean patch probabilities, averaging
over images. Per-grader prediction metrics and aggregation are unchanged.

Every new run records the exact source weights, effective float32 weights,
baseline path/grader, source file hashes, reconstruction method/counts when
needed, loss reduction/denominator and initial-model fingerprint in
`run_config.json`.

The RNG correction applies to **both hard and soft modes**: class-counting
passes restore the global CPU and loader-generator states, the training and
validation loaders use separate generators seeded with `seed` and `seed + 1`,
and the model is seeded immediately before initialisation. New paired runs
with the same seed/settings and the same ordered eligible images/patches share
initial model weights and training sample order, independent of class counting
or validation loader iterations. Existing architecture, dropout, augmentation,
optimisers, scheduler and stopping rules are unchanged. Cross-device/version
bitwise training determinism is not promised; differing targets produce
different optimisation trajectories and potentially different stopping epochs.

Historical files are untouched. Their initialisation/shuffle sequence used the
old shared RNG behaviour, so they cannot be claimed as exactly paired with new
runs after this correction. For a controlled pair, rerun the corresponding
hard-label mode using the updated code, for example:

```powershell
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader previousStudent --output_dir .\runs --model_name previousStudent_paired_seed42 --epochs 50 --batch_size 16 --learning_rate 0.001 --optimizer AdamW --patience 10 --seed 42 --reduce_lr
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader shane --output_dir .\runs --model_name shane_paired_seed42 --epochs 50 --batch_size 16 --learning_rate 0.001 --optimizer AdamW --patience 10 --seed 42 --reduce_lr
```

The focused suite checks the original formula against reconstructed weights,
saved-vector precedence, partial coverage, unchanged loss for repeated vote
distributions, a manual weighted-loss example, both mean denominators, RNG-neutral
counting, and bitwise-equal initial states/sample orders/dropout draws in the
actual three training entry paths (epoch updates mocked). It also exercises
weighted save/reload and reporting and retains the real-image patch smoke check:

```powershell
.\.venv\Scripts\python.exe -m unittest test_setup test_validation test_soft_labels test_weighted_training -v
```

## Shared manifest and launch commands

The existing manifest remains authoritative: 508 train, 127 validation and 26
test images. No splits are generated by soft training. Duplicate IDs, ambiguous
paths, conflicting original grades/metadata/splits and new unknown IDs fail.
The workbook also includes the eight Shane-only variants already excluded in
`prepared_visus_nhp/split_report.json`. `-ExcludeKnownUnmatched` explicitly skips
only IDs recorded in that report and prints their identifiers. Without it these
extra rows fail validation. Any newly unknown ID still fails; adding images
requires a separately reviewed manifest with image, metadata and split checks.

Short input check (no images decoded, training, or run folder created):

```powershell
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader soft --ratings_csv_path 'C:\Users\Shane\Documents\multigrader_ratings.xlsx' --exclude_known_unmatched --output_dir .\runs --check_only
```

Full run, up to 50 epochs with early stopping and the existing AdamW settings:

```powershell
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader soft --ratings_csv_path 'C:\Users\Shane\Documents\multigrader_ratings.xlsx' --exclude_known_unmatched --output_dir .\runs --model_name soft_ce_seed42 --epochs 50 --batch_size 16 --optimizer AdamW --learning_rate 0.001 --patience 10 --seed 42 --reduce_lr
```

Use `--epochs 1` for a complete one-epoch trial. Choose a new `--model_name`
if it already exists. Original modes:

```powershell
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader previousStudent --output_dir .\runs --model_name previousStudent_ce_seed42 --epochs 50 --optimizer AdamW --seed 42 --reduce_lr
.\.venv\Scripts\python.exe .\training.py --dataset_csv_path .\prepared_visus_nhp\manifest.csv --grader shane --output_dir .\runs --model_name shane_ce_seed42 --epochs 50 --optimizer AdamW --seed 42 --reduce_lr
```

The existing `run_shane.ps1` also accepts `-Grader`, `-RatingsCsvPath`,
`-GraderColumns`, `-ExcludeKnownUnmatched` and `-CheckOnly`, and generates unique
run names. Direct Python commands above also work where PowerShell execution
policy blocks `.ps1` scripts; no policy change is required.

Soft runs save `targets.csv` with the exact selected original ratings, soft
targets, rating counts and all existing split assignments; `ratings_input.xlsx`
(or `.csv`) preserves the full input; the original manifest and split report
are copied too. `run_config.json` saves columns, input hashes (split identity),
seed, loss, class weights and model/training settings. Existing run files are
never overwritten. Do not edit inputs while a run is in progress.

Checkpoint selection uses mean **patch** validation cross-entropy against soft
targets. Whole-image reporting still averages nine patch softmax probabilities.
`validation_metrics.json` additionally reports unweighted soft cross-entropy of
those mean probabilities, and exact accuracy, within-one accuracy, MAE, quadratic
weighted kappa, macro F1 and weighted F1 separately for each selected grader,
with its evaluated image count. A grader with no validation ratings has count
zero and an explicit reason. Per-grade and confusion CSVs include a `grader`
column. `validation_predictions.csv` contains probabilities, predicted grade,
original selected ratings, soft targets and rating counts. No consensus grade is
created. Test images are never decoded or used for checkpoint selection.
Re-evaluation uses the saved target snapshot, independent of subsequent workbook
edits, through the existing `validation.py --run-dir ...` command.

Install dependencies if needed, then run all focused checks:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m unittest test_setup test_validation test_soft_labels -v
```

Checks cover missing/invalid grades, target fractions, duplicate IDs/headers,
variant joins, unchanged real image population and splits, hard/one-hot loss and
gradient equivalence, a real-image two-patch IQAnet optimizer step, save/reload
and per-grader reporting, plus all existing single-grader regression checks.
They use temporary outputs and do not launch a full training run.

## Existing single-grader data and workflow

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
