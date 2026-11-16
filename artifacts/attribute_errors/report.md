# Validation errors by image attribute

Compared 13 saved runs on the same 127 validation images. Grade errors are predictions differing from the saved consensus reference grade. All grades come from existing prediction files, never the supplied Shane CSV. No retraining or test predictions were used.

Best saved QWK: selected_soft_ordinal_coral_1_20261104_160210_897397_4a1d9225 (0.776).

Rates describe these validation images. Attributes can overlap; project, grade and patient differences may influence comparisons. Multiple runs reuse the same images and are not independent samples.

## Shane annotations

| Attribute | Validation images | Mean error rate across runs | Untagged error rate | Mean absolute grade error | Grade-adjusted excess error |
|---|---:|---:|---:|---:|---:|
| Blur | 26 | 45.0% | 39.7% | 0.464 | +7.5 percentage points |
| Uneven Illumination | 21 | 38.1% | 41.3% | 0.388 | +0.8 percentage points |
| Noise | 95 | 37.2% | 51.4% | 0.387 | -0.2 percentage points |
| Artifact | 21 | 29.3% | 43.0% | 0.293 | -3.9 percentage points |
| Dark | 11 | 20.3% | 42.7% | 0.203 | -7.2 percentage points |
| Low Contrast | 4 | 13.5% | 41.7% | 0.135 | -7.9 percentage points |

Grade-adjusted excess subtracts each run's error rate within each reference grade before comparing tagged and untagged images. It does not adjust for project or patient.

### Best-QWK model

| Attribute | Errors / tagged images | Tagged error rate | Untagged error rate | Mean absolute grade error | Binary-head error rate |
|---|---:|---:|---:|---:|---:|
| Noise | 49/95 | 51.6% | 28.1% | 0.526 | nan% |
| Blur | 12/26 | 46.2% | 45.5% | 0.462 | nan% |
| Uneven Illumination | 8/21 | 38.1% | 47.2% | 0.381 | nan% |
| Dark | 4/11 | 36.4% | 46.6% | 0.364 | nan% |
| Artifact | 7/21 | 33.3% | 48.1% | 0.333 | nan% |
| Low Contrast | 1/4 | 25.0% | 46.3% | 0.250 | nan% |

### Example errors (best-QWK model)

- Noise: ASTELLAS_JPN01_003_OD_V2_15122021_1.1_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V2_15122021_1.2_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V3_27122021_1.1_0: reference 1, predicted 3.
- Blur: ASTELLAS_JPN01_003_OD_V2_15122021_1.2_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V4_1112022_1.0_0: reference 2, predicted 1; ASTELLAS_JPN01_003_OS_V2_15122021_1.0_0: reference 2, predicted 3.
- Artifact: DARC-P2_GBR01_019_OS_V1_20102021_1.1_0: reference 4, predicted 3; DARC-P2_GBR01_025_OS_V1_20102021_1.0_0: reference 2, predicted 3; DARC-P2_GBR01_025_OS_V1_20102021_1.2_0: reference 3, predicted 4.
- Dark: ASTELLAS_JPN01_003_OD_V1_25102021_1.1_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V1_25102021_1.2_0: reference 4, predicted 3; DARC-P2_GBR01_015_OD_V1_20102021_1.0_0: reference 2, predicted 3.
- Uneven Illumination: ASTELLAS_JPN01_003_OD_V1_25102021_1.1_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V1_25102021_1.2_0: reference 4, predicted 3; ASTELLAS_JPN01_003_OD_V4_1112022_1.0_0: reference 2, predicted 1.
## Ganaav annotations

| Attribute | Validation images | Mean error rate across runs | Untagged error rate | Mean absolute grade error | Grade-adjusted excess error |
|---|---:|---:|---:|---:|---:|
| Blur | 97 | 38.9% | 46.7% | 0.405 | +15.1 percentage points |
| Noise | 82 | 34.5% | 52.1% | 0.361 | -0.1 percentage points |
| Dark | 3 | 30.8% | 41.0% | 0.308 | -14.5 percentage points |
| Low Contrast | 5 | 29.2% | 41.2% | 0.292 | -12.8 percentage points |
| Uneven Illumination | 6 | 28.2% | 41.4% | 0.282 | -3.3 percentage points |
| Artifact | 6 | 21.8% | 41.7% | 0.218 | -17.6 percentage points |

Grade-adjusted excess subtracts each run's error rate within each reference grade before comparing tagged and untagged images. It does not adjust for project or patient.

### Best-QWK model

| Attribute | Errors / tagged images | Tagged error rate | Untagged error rate | Mean absolute grade error | Binary-head error rate |
|---|---:|---:|---:|---:|---:|
| Uneven Illumination | 4/6 | 66.7% | 44.6% | 0.667 | nan% |
| Low Contrast | 3/5 | 60.0% | 45.1% | 0.600 | nan% |
| Noise | 39/82 | 47.6% | 42.2% | 0.488 | nan% |
| Blur | 45/97 | 46.4% | 43.3% | 0.474 | nan% |
| Artifact | 2/6 | 33.3% | 46.3% | 0.333 | nan% |
| Dark | 1/3 | 33.3% | 46.0% | 0.333 | nan% |

### Example errors (best-QWK model)

- Noise: ASTELLAS_JPN01_003_OD_V1_25102021_1.1_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V1_25102021_1.2_0: reference 4, predicted 3; ASTELLAS_JPN01_003_OD_V2_15122021_1.1_0: reference 2, predicted 3.
- Blur: ASTELLAS_JPN01_003_OD_V1_25102021_1.2_0: reference 4, predicted 3; ASTELLAS_JPN01_003_OD_V2_15122021_1.1_0: reference 2, predicted 3; ASTELLAS_JPN01_003_OD_V2_15122021_1.2_0: reference 2, predicted 3.
- Artifact: DARC-P2_GBR01_050_OD_V1_20102021_1.1_0: reference 2, predicted 1; DARC-P2_GBR01_050_OS_V1_20102021_1.1_0: reference 2, predicted 1.
- Dark: DARC-P2_GBR01_151_OD_V1_20102021_1.0_0: reference 5, predicted 4.
- Uneven Illumination: DARC-P2_GBR01_025_OD_V1_20102021_1.2_0: reference 4, predicted 3; DARC-P2_GBR01_025_OS_V1_20102021_1.2_0: reference 3, predicted 4; DARC-P2_GBR01_044_OD_V1_20102021_1.0_0: reference 2, predicted 3.

## Run coverage

- selected_soft_ordinal_coral_1_20261104_160210_897397_4a1d9225: QWK 0.776; exact grade errors 58/127.
- selected_or_soft_binary_1_20261105_101626_540011_3638a7b2: QWK 0.773; exact grade errors 47/127.
- selected_or_soft_binary_2_20261105_103202_692309_ee4e3879: QWK 0.770; exact grade errors 48/127.
- selected_soft_ordinal_coral_2_20261105_094701_831302_9e0d3d81: QWK 0.764; exact grade errors 56/127.
- resnet18_or_soft_binary_lambda1_seed42_20261106_155123_218636_65d14090: QWK 0.762; exact grade errors 53/127.
- selected_continuous_cdw_ce_1_20261104_143532_236158_8a130589: QWK 0.761; exact grade errors 48/127.
- selected_or_soft_1_20261104_145800_777691_fb65827f: QWK 0.759; exact grade errors 48/127.
- selected_or_soft_frozen_binary_1_20261105_104707_379291_2e5cc995: QWK 0.759; exact grade errors 48/127.
- selected_or_soft_2_20261104_151528_520807_12460ad6: QWK 0.755; exact grade errors 50/127.
- or_soft_binary_lambda2_seed42_20261106_144630_087530_30c95076: QWK 0.750; exact grade errors 50/127.
- or_soft_binary_lambda1_seed42_20261106_142229_367290_cc0bbd7e: QWK 0.745; exact grade errors 49/127.
- selected_soft_ce_1_20261105_105006_844617_0b14f684: QWK 0.742; exact grade errors 64/127.
- selected_continuous_cdw_ce_2_20261104_144640_914073_9a0a156f: QWK 0.717; exact grade errors 54/127.

Sources: Shane attribute CSV in Downloads; Ganaav original attribute CSV in qc_data/labels; validation_predictions.csv and validation_metrics.json from the listed completed runs.

## Individual model checks using Shane annotations

| Run | All grade errors | Blur grade errors | Non-blur grade errors | Blur binary errors | Non-blur binary errors |
|---|---:|---:|---:|---:|---:|
| selected_continuous_cdw_ce_1_20261104_143532_236158_8a130589 | 48/127 | 11/26 (42.3%) | 37/101 (36.6%) | n/a | n/a |
| selected_continuous_cdw_ce_2_20261104_144640_914073_9a0a156f | 54/127 | 11/26 (42.3%) | 43/101 (42.6%) | n/a | n/a |
| selected_or_soft_1_20261104_145800_777691_fb65827f | 48/127 | 10/26 (38.5%) | 38/101 (37.6%) | n/a | n/a |
| selected_or_soft_2_20261104_151528_520807_12460ad6 | 50/127 | 11/26 (42.3%) | 39/101 (38.6%) | n/a | n/a |
| selected_soft_ordinal_coral_1_20261104_160210_897397_4a1d9225 | 58/127 | 12/26 (46.2%) | 46/101 (45.5%) | n/a | n/a |
| selected_soft_ordinal_coral_2_20261105_094701_831302_9e0d3d81 | 56/127 | 12/26 (46.2%) | 44/101 (43.6%) | n/a | n/a |
| selected_or_soft_binary_1_20261105_101626_540011_3638a7b2 | 47/127 | 12/26 (46.2%) | 35/101 (34.7%) | 6/26 (23.1%) | 17/101 (16.8%) |
| selected_or_soft_binary_2_20261105_103202_692309_ee4e3879 | 48/127 | 12/26 (46.2%) | 36/101 (35.6%) | 6/26 (23.1%) | 16/101 (15.8%) |
| selected_or_soft_frozen_binary_1_20261105_104707_379291_2e5cc995 | 48/127 | 10/26 (38.5%) | 38/101 (37.6%) | 6/26 (23.1%) | 15/101 (14.9%) |
| selected_soft_ce_1_20261105_105006_844617_0b14f684 | 64/127 | 17/26 (65.4%) | 47/101 (46.5%) | n/a | n/a |
| or_soft_binary_lambda1_seed42_20261106_142229_367290_cc0bbd7e | 49/127 | 11/26 (42.3%) | 38/101 (37.6%) | 6/26 (23.1%) | 14/101 (13.9%) |
| or_soft_binary_lambda2_seed42_20261106_144630_087530_30c95076 | 50/127 | 11/26 (42.3%) | 39/101 (38.6%) | 6/26 (23.1%) | 16/101 (15.8%) |
| resnet18_or_soft_binary_lambda1_seed42_20261106_155123_218636_65d14090 | 53/127 | 12/26 (46.2%) | 41/101 (40.6%) | 7/26 (26.9%) | 15/101 (14.9%) |

Reference ambiguity: 56/127 saved consensus grade references are tied. For the lambda 0.25 two-head model, excluding tied grade references gives 2/10 blur errors versus 15/61 non-blur errors (20.0% versus 24.6%). Excluding tied binary references gives 1/16 blur errors versus 8/82 non-blur errors (6.3% versus 9.8%). The apparent blur weakness does not persist in these untied subsets, so interpret it as exploratory.

