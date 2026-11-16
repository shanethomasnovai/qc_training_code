# AutoQC

CNN image-quality training with ordinal, two-head and classification models.

## Layout

```text
auto_qc/CNN/             Training, models and evaluation
data/tools/      Dataset preparation code
data/labels.py          Shared label validation, target encoding and workbook loading
data/datasets/          Current generated label tables
data/sources/           Inputs used to rebuild those tables
data/models/            New training outputs
data/samples/config.yaml Editable example training config
data/tools/datasets.yaml   Dataset build recipe
tests/                  Pipeline regression checks
```

## Train

Install the dependencies with `python -m pip install -r requirements.txt`.
Edit [data/samples/config.yaml](data/samples/config.yaml). Images remain in your
local folder; set `local_root` to the folder containing `images/`, `labels/`,
`datasets/` and `models/`. Dataset paths in the example are relative to
`local_root/datasets/`.

```powershell
python -m auto_qc.CNN.training --config data/samples/config.yaml --check-only
python -m auto_qc.CNN.training --config data/samples/config.yaml
```

`or_soft_binary` uses ordinal and binary heads. `or_soft` and `coral` use one
ordinal head. Training reuses the saved patient split before extracting patches.
The current data contains 508 training and 127 validation images, with VISUS-NHP
held out. Graders are read from the dataset metadata.

With the sample's `local_root`, each new run is saved under
`<local_root>/models/` with exactly seven files: `checkpoint.pth`, a model-named
config such as `or_soft.yaml`, `training_history.png`,
`validation_metrics.json`, `validation_predictions.csv`, `dataset.xlsx` and
`training_history.csv`. The dataset snapshot includes the labels, splits and
cohort flags used by the run; images remain in their original folder. The history
CSV records each epoch's training/validation losses and learning rate. Metrics
JSON contains summary scores, while predictions CSV contains per-image results.
The saved YAML follows the input config's simple layout, preserving its comments
when no command-line overrides are used. Internal settings stay in the checkpoint.
The checkpoint embeds the dataset snapshot needed for later evaluation.
Progress is printed in the terminal. Plotting caches use Matplotlib's normal
cache location outside the model folder. Legacy configurations without
`local_root` retain their historical output and results-collection behavior.

## Prepare data

[data/tools/datasets.yaml](data/tools/datasets.yaml) specifies the pinned annotation
and ratings sources under `data/sources/`. It uses the saved split crosswalk in
`prepared_visus_nhp/manifest.csv` and writes label tables under `data/datasets/`.

```powershell
python -m data.tools.generate_datasets --verify-existing
python -m data.tools.generate_datasets --output-dir data/datasets/new_labels
```

The label tables represent different training targets: individual ratings,
exact means, rounded means and multigrader labels. The multigrader table supports
ordinal, soft-classification and continuous-mean training.
See [data/README.md](data/README.md) for the source-to-label workflow, label rules
and commands using your local data folder. Historical rebuilding lives in the
rerun module; normal dataset generation only writes the five current label tables.

## ResNet-18 comparison

Use the universal `data/samples/config.yaml` for either architecture. Set
`model.backbone: resnet18` and `model.pretrained: true` for torchvision's
ImageNet-pretrained ResNet-18. For IQAnet, use `backbone: iqanet` and
`pretrained: false`. Leave IQAnet-specific layer settings out when using ResNet.
The head and loss are selected separately: the two-head comparison uses
`model.head: or_soft_binary`, `loss.name: joint_bce`, and
`loss.binary_loss_weight: 1.0`. The lower-learning-rate comparison uses
`optimizer.lr: 0.00001`.
It fine-tunes the full backbone. The model repeats each grayscale patch into
three channels and applies ImageNet normalization internally; the original 256px
patches are used when `data.input_mode: patches`. Set `data.input_mode: whole_image`
to resize the entire image to 256 x 256 and use one prediction per image instead
of averaging nine patch predictions. Both modes keep horizontal flips for
training, the saved patient split, heads and loss. The mode is saved in the
checkpoint, readable YAML and Excel experiment log. ResNet's standard batch
normalization layers are used. This compares a pretrained backbone with IQAnet
trained from scratch, so it measures architecture and pretraining together.

```powershell
python -m auto_qc.CNN.training --config data/samples/config.yaml
```

The first run downloads the official torchvision weights to the normal Torch
cache. Saved checkpoints reload without downloading weights. The run produces
the same seven files, with config named after the architecture and head
(for example, `resnet18_or_soft_binary.yaml`).

## Four-convolution IQAnet comparison

Set `model.backbone: iqanet4` in the universal config to add two 3 x 3,
stride-1 convolutions with ReLU after the original second convolution.
Both keep the same channel count and spatial size. The original max/min pooling,
fully connected layers, heads, patch-level loss and equal patch averaging are
retained. `backbone: iqanet` still selects the original two-convolution model.

This is a custom depth experiment informed by compact CNN approaches such as
[basic IQDN](https://doi.org/10.1016/j.jvcir.2026.104801), not a reproduction of
that paper's architecture or loss. The comparison keeps the same saved split,
seed, patch batch size, optimizer and scheduler. Saved checkpoints and the Excel
log identify the architecture as `iqanet4`; the readable output config is named
`iqanet4_or_soft_binary.yaml` for the two-head setup.

## Verify

```powershell
python -m unittest discover -s tests -v
```

Older runs and review records remain under `runs/`, `model_dir/` and
`experiments/`. Older data-folder contents are preserved in
`model_dir/dataset_support/legacy_data/`.
Generated historical training configurations live under `experiments/configs/`.
