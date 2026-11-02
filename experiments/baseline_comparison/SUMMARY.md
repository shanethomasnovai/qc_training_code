# Changes from the supervisor's Multi-class-qc baseline

Reference: the supplied `ds-auto-qc-Multi-class-qc (1).zip`. SHA-256:
`6ccfb237b8358e28caaa9a0cf152b9aa52dc3352371fbcda5d73fc6c185a6c34`.
The original commit is unknown; the ZIP's commit-like comment has not been independently verified.

## Retained foundation — verified

- Original `auto_qc/CNN` layout and five core module names.
- The original `IQAnet` constructor and feature operations: two convolutions, max/min pooling, FC 600/400, optional batch normalisation and dropout.
- Resize 768×768, one grayscale channel, ToTensor, nine row-major 256×256 patches, and one horizontal-flip copy per training patch.
- The original five-logit model remains available for hard/soft categorical CE and continuous CDW-CE.

Constructor and feature-operation syntax matches the archive. Eight synthetic forward cases produce **exactly equal outputs** for the original and current five-grade IQAnet, covering BN on/off, sigmoid on/off and train/eval modes. Synthetic PNG tests also confirm exact equality of all nine patches and all 18 original/flip training inputs. These checks take no optimiser steps and use no real patient images.

## Code changes to review

| Module | Changes from the original |
|---|---|
| `model.py` | Directly extended: exposes `forward_features`; adds ordered CORAL, independent OR-Soft, joint binary and frozen binary heads using the same features. |
| `dataset.py` | Substantially rewritten: versioned Excel inputs, explicit targets and grade indices, missing-rating handling, patient/split checks, lazy patch loading. |
| `training.py` | Substantially rewritten: shared configured entry point, explicit objectives, seeds, cloned best weights, unique run directories and complete saved outputs. |
| `testing.py` | Substantially rewritten: saved-architecture loading, whole-image probability averaging, consensus/distribution/binary evaluation and separate optional test evaluation. |
| `utils.py` | Replaced small loader-count helpers with explicit ordered weights, RNG/provenance and checkpoint helpers. |

Added supporting modules: `losses.py`, `metrics.py`, `generate_datasets.py`, `rerun.py`, `collect_results.py`; plus configurations and focused tests. Original `kfold.py` is retained in the reference copy only; the current pipeline requires fixed patient assignments before patches.

## Scientific changes — distinguish from implementation fixes

- New Ganaav/Shane and selected multigrader targets: hard grades, ceiling/exact means, vote fractions and cumulative probabilities. Exact means use the existing continuous CDW-CE definition, not integer class indices.
- New output heads/objectives. Joint training adds a weighted binary objective; frozen training updates only its binary head. OR-Soft's inference-only PAVA is our explicit adaptation.
- The same 661 full image IDs and patient IDs are retained, but **332 split assignments differ**: original main CSV is 440/75/146 train/validation/test images; later saved assignments are 508/127/26 with all VISUS held out.
- Current active workbooks use reviewed-v2; retrospective reruns retain their recorded rating versions. These versions must be identified in comparisons.
- Original CNN testing scores patches; current evaluation averages patch probabilities to images. Optimiser, weighting, LR and scheduler choices are explicit experiment settings, not universally unchanged baseline defaults.

Implementation fixes include best-state copying, class-indexed weights, seeding, mandatory patient splits, architecture-aware loading and output-overwrite protection. The existing test file covers these separately from experimental method choices.

## Review directly

`original/` contains byte-exact CNN source from the archive. `changes.patch` shows all CNN source additions/modifications/removals; `verification.json` records checks and source hashes. Ordinary Git diff uses the later repository HEAD; use this archive comparison for the supervisor's baseline.

```powershell
code --diff .\experiments\baseline_comparison\original\auto_qc\CNN\model.py .\auto_qc\CNN\model.py
.\.venv\Scripts\python.exe -B .\experiments\baseline_comparison\verify_baseline.py --refresh-diff
```

Scope/limits: preprocessing parity uses synthetic PNGs in the installed environment; original TIFF-to-PNG equivalence and reproduction of historical scores are not established. This comparison did not alter runtime code, README, datasets, checkpoints or the running experiments.
