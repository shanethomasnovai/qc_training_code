
import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


from model import IQAnet
from dataset import ImageQualityDataset, subsample_subimages
from utils import get_class_counts


def calculate_cm_metrics(tn, fp, fn, tp):

    results = {}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    results["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    results["precision"] = precision
    results["recall"] = recall
    results["f1_score"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results["true_positive"] = tp
    results["true_negative"] = tn
    results["false_positive"] = fp
    results["false_negative"] = fn
    results["support"] = tp + fn

    return results


def calculate_weighted_metrics(y_true, y_pred, y_pred_proba, num_classes=2):
    metric_results = []
    
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    for weighting_type in ['macro', 'micro', 'weighted']:

        results = {}

        results["weighting_type"] = weighting_type

        results["precision"] = precision_score(y_true, y_pred, average=weighting_type, zero_division=0)
        results["recall"] = recall_score(y_true, y_pred, average=weighting_type, zero_division=0)
        results["f1_score"] = f1_score(y_true, y_pred, average=weighting_type, zero_division=0)
        results["roc_auc"] = roc_auc_score(y_true_bin, y_pred_proba, average=weighting_type, multi_class='ovr')
        results["average_precision"] = average_precision_score(y_true_bin, y_pred_proba, average=weighting_type)

        metric_results.append(results)

    return pd.DataFrame(metric_results)


def calculate_per_class_metrics(y_true, y_pred, y_pred_proba, num_classes=2):
    metric_results = []

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    for class_id in range(num_classes):
        y_true_binary = [1 if label == class_id else 0 for label in y_true]
        y_pred_binary = [1 if label == class_id else 0 for label in y_pred]

        results = {}
        results["class"] = f"Class_{class_id}"

        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results.update(calculate_cm_metrics(tn, fp, fn, tp))
        
        if len(np.unique(y_true_bin[:, class_id])) > 1:
            results["roc_auc"] = roc_auc_score(y_true_bin[:, class_id], y_pred_proba[:, class_id])
            results["pr_auc"] = average_precision_score(y_true_bin[:, class_id], y_pred_proba[:, class_id])
        else:
            results["roc_auc"] = np.nan
            results["pr_auc"] = np.nan

        metric_results.append(results)

    return pd.DataFrame(metric_results)


def get_model_predictions(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    y_all = []
    ypred_all = []
    ypred_prob_all = []

    print("Running inference on test set...")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            test_loss += loss.item() * inputs.size(0)

            # Get probabilities
            proba = torch.softmax(logits, dim=1)
            ypred_prob_all.extend(proba.cpu().numpy())

            _, preds = torch.max(logits, 1)
            y_all.extend(targets.cpu().numpy())
            ypred_all.extend(preds.cpu().numpy())

            print(f"Batch {i+1}/{len(test_loader)}", end='\r')

    test_loss /= len(test_loader.dataset)
    print("\nTest set evaluation complete!")

    return test_loss, np.array(y_all), np.array(ypred_all), np.array(ypred_prob_all)


def main():

    parser = argparse.ArgumentParser(description='Test a trained model on held-out test set')
    parser.add_argument(
        '--model_filepath',
        type=str,
        required=True,
        help='Path to saved model.pth file'
    )
    parser.add_argument(
        '--dataset_filepath',
        type=str,
        required=True,
        help='Path to split CSV file'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Base directory containing the images'
    )
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save test results')

    args = parser.parse_args()

    # Setup device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    print("\n")
    print("Loading dataset")
    print("="*60)

    df = pd.read_csv(args.dataset_filepath)

    num_classes = len(set(df['image_grading'].unique()))

    print('Loading dataset...')
    dataset = ImageQualityDataset(df, transform, base_image_dir=args.images_dir)

    print('Patching dataset...')
    patched_dataset, df = subsample_subimages(dataset, method='all', df=df)

    if len(patched_dataset[0]) == 3:
        # Split dataset using the mode column
        test_dataset = [(images, labels) for (images, labels, mode) in patched_dataset if mode == 'test']
    else:
        # Randomly split
        n_samples = len(patched_dataset)
        val_size = int(n_samples * args.val_split)
        test_size = int(n_samples * args.test_split)

        train_dataset, x_remaining = train_test_split(
            patched_dataset,
            test_size=val_size+test_size,
            random_state=args.seed
        )

        val_dataset, test_dataset = train_test_split(
            x_remaining,
            test_size=test_size,
            random_state=args.seed
        )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Test set size: {len(test_dataset)}")
    print(f"Image size (H, W): {patched_dataset[0][0].shape[1:]}")

    print("\nLoading trained model")

    if not os.path.exists(args.model_filepath):
        print(f"ERROR: Model file not found at {args.model_filepath}")
        print("\nPlease provide the correct path to your model.pth file")
        return

    print(f"Loading model from: {args.model_filepath}")

    # Load the saved model
    model = torch.load(args.model_filepath, map_location=device)

    # If state_dict, load into model
    if isinstance(model, dict):
        print("Detected state_dict, loading into model...")
        print(num_classes)
        full_model = IQAnet(batch_normalization=True, output_sigmoid=False, num_classes=num_classes)
        full_model.load_state_dict(model)
        model = full_model

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    class_counts = get_class_counts(test_loader)

    print("\nTest set class distribution:")
    for class_label, count in class_counts.items():
        print(f"  Class {class_label}: {count} samples")

    criterion = nn.CrossEntropyLoss()

    print("\nEvaluating on test set")

    test_loss, y_true, y_pred, y_pred_proba = get_model_predictions(model, test_loader, criterion, device)

    weighted_metric_df = calculate_weighted_metrics(y_true, y_pred, y_pred_proba, num_classes=num_classes)
    per_class_metric_df = calculate_per_class_metrics(y_true, y_pred, y_pred_proba, num_classes=num_classes)

    os.makedirs(args.output_dir, exist_ok=True)

    weighted_metric_csv_path = os.path.join(args.output_dir, 'weighted_metrics.csv')
    weighted_metric_df.to_csv(weighted_metric_csv_path, index=False)

    per_class_metric_csv_path = os.path.join(args.output_dir, 'per_class_metrics.csv')
    per_class_metric_df.to_csv(per_class_metric_csv_path, index=False)

    if num_classes == 2:
        labels = [1, 0]
    else:
        labels = list(range(num_classes))

    # Confusion matrix
    cm_df = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f'True_Class_{i}' for i in labels],
        columns=[f'Pred_Class_{i}' for i in labels]
    )
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix: {cm_path}")

    # Predictions
    predictions_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': [1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)]
    })
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"All predictions saved: {predictions_path}")


if __name__ == '__main__':
    main()