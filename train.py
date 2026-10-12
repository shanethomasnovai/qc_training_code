import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import IQAnet


class ImageDataset(Dataset):
    def __init__(self, rows, grader, training=False):
        self.rows = rows.reset_index(drop=True)
        self.grader = grader
        self.training = training
        self.transform = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(row.image_path) as source:
            image = source.convert("L")
            if self.training and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = self.transform(image)
        patches = torch.stack([
            image[:, top:top + 256, left:left + 256]
            for top in (0, 256, 512) for left in (0, 256, 512)
        ])
        return patches, int(row[f"{self.grader}_grade"]) - 1


def evaluate(model, loader, device):
    model.eval()
    losses, probabilities = [], []
    with torch.no_grad():
        for patches, targets in loader:
            batch, count = patches.shape[:2]
            logits = model(patches.flatten(0, 1).to(device))
            probs = logits.softmax(dim=1).reshape(batch, count, 5).mean(dim=1)
            loss = nn.functional.nll_loss(probs.clamp_min(1e-12).log(), targets.to(device), reduction="sum")
            losses.append(loss.item())
            probabilities.extend(probs.cpu().numpy())
    return sum(losses) / len(loader.dataset), np.asarray(probabilities)


def grade_metrics(reference, prediction):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    kappa = None
    if len(set(reference.tolist()) | set(prediction.tolist())) > 1:
        value = cohen_kappa_score(reference, prediction, labels=[1, 2, 3, 4, 5], weights="quadratic")
        kappa = float(value) if np.isfinite(value) else None
    return {
        "images": len(reference),
        "exact_accuracy_percent": accuracy_score(reference, prediction) * 100,
        "within_one_grade_percent": float(np.mean(np.abs(reference - prediction) <= 1) * 100),
        "mean_absolute_grade_error": mean_absolute_error(reference, prediction),
        "quadratic_weighted_kappa": kappa,
        "confusion_matrix": confusion_matrix(reference, prediction, labels=[1, 2, 3, 4, 5]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("prepared/manifest.csv"))
    parser.add_argument("--grader", choices=["previousStudent", "shane"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out", default="VISUSS")
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, patience and learning rate must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory to avoid overwriting a run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    rows = pd.read_csv(args.manifest, dtype={"image_id": str, "patient_id": str, "project": str, "project_site": str})
    required = {"image_id", "image_path", "patient_id", "project", "split", "previousStudent_grade", "shane_grade"}
    if not required.issubset(rows.columns) or rows[list(required)].isna().any().any():
        raise ValueError("Manifest has missing columns or missing values")
    if rows.image_id.duplicated().any() or not rows.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Duplicate images or invalid split names")
    if rows.groupby("patient_id")["split"].nunique().gt(1).any():
        raise ValueError("Patient leakage across splits")
    held_out = rows.project.str.casefold().eq(args.held_out.casefold())
    if not held_out.equals(rows.split.eq("test")):
        raise ValueError("The test set must contain exactly the held-out trial")
    for grader in ("previousStudent", "shane"):
        if not rows[f"{grader}_grade"].isin([1, 2, 3, 4, 5]).all():
            raise ValueError(f"Invalid {grader} grades")
    missing = [path for path in rows.image_path if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image files, including {missing[:3]}")
    subsets = {split: rows.loc[rows.split.eq(split)].reset_index(drop=True) for split in ("train", "validation", "test")}
    if any(subset.empty for subset in subsets.values()):
        raise ValueError("Training, validation and test sets must all be nonempty")
    counts = np.bincount(subsets["train"][f"{args.grader}_grade"].to_numpy(dtype=int) - 1, minlength=5)
    if (counts == 0).any():
        raise ValueError(f"No training examples for grades {(np.flatnonzero(counts == 0) + 1).tolist()}. Review your data and split before a five-grade run")
    loaders = {
        split: DataLoader(ImageDataset(subset, args.grader, training=split == "train"), batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
        for split, subset in subsets.items()
    }
    model = IQAnet().to(device)
    weights = torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32, device=device) if args.class_weighting == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({"device_used": str(device), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "training_grade_counts": counts.tolist(), "patch_aggregation": "mean probabilities"})
    (args.output_dir / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    history, best, best_loss, stale = [], None, float("inf"), 0
    print(f"Grader: {args.grader}; device: {device}; training images: {len(subsets['train'])}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_patches = 0.0, 0
        for patches, targets in loaders["train"]:
            count = patches.shape[1]
            inputs = patches.flatten(0, 1).to(device)
            targets = targets.repeat_interleave(count).to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            total_patches += len(inputs)
        validation_loss, probabilities = evaluate(model, loaders["validation"], device)
        validation_accuracy = accuracy_score(subsets["validation"][f"{args.grader}_grade"], probabilities.argmax(axis=1) + 1) * 100
        history.append({"epoch": epoch, "train_patch_loss": total_loss / total_patches, "validation_image_loss": validation_loss, "validation_accuracy_percent": validation_accuracy})
        pd.DataFrame(history).to_csv(args.output_dir / "learning_curve.csv", index=False)
        print(f"Epoch {epoch}: train loss {total_loss / total_patches:.4f}; validation loss {validation_loss:.4f}; validation accuracy {validation_accuracy:.1f}%")
        if not np.isfinite(validation_loss):
            raise RuntimeError("Non-finite validation loss")
        if validation_loss < best_loss:
            best_loss, stale = validation_loss, 0
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best, "configuration": configuration, "best_epoch": epoch}, args.output_dir / "best_model.pth")
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best)
    _, probabilities = evaluate(model, loaders["validation"], device)
    validation_metrics = grade_metrics(subsets["validation"][f"{args.grader}_grade"], probabilities.argmax(axis=1) + 1)
    (args.output_dir / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
    if args.evaluate_test:
        _, probabilities = evaluate(model, loaders["test"], device)
        predictions = subsets["test"].copy()
        predictions["predicted_grade"] = probabilities.argmax(axis=1) + 1
        for index in range(5):
            predictions[f"probability_grade_{index + 1}"] = probabilities[:, index]
        predictions.to_csv(args.output_dir / "VISUSS_predictions.csv", index=False)
        metrics = {grader: grade_metrics(predictions[f"{grader}_grade"], predictions.predicted_grade) for grader in ("previousStudent", "shane")}
        for grader, result in metrics.items():
            pd.DataFrame(result["confusion_matrix"], index=range(1, 6), columns=range(1, 6)).to_csv(args.output_dir / f"VISUSS_confusion_matrix_against_{grader}.csv", index_label="reference_grade")
        (args.output_dir / "VISUSS_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    else:
        print("Test evaluation skipped. Use --evaluate-test only when your training settings are fixed.")
    print(f"Saved run: {args.output_dir}")


if __name__ == "__main__":
    main()
