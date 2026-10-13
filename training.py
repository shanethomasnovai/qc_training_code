import os
import argparse
import pandas as pd
import shutil
import copy
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


from model import IQAnet
from dataset import ImageQualityDataset, subsample_subimages
from utils import get_class_counts
from validation import evaluate_saved_run


def class_balance(train_loader, val_loader):
    set_class_weights = {'Training': None, 'Validation': None}

    for loader_label, loader in zip(['Training', 'Validation'], [train_loader, val_loader]):
        class_counts = get_class_counts(loader)

        label_sum = sum(class_counts.values())
        n_classes = len(class_counts)

        class_weight = {}
        for class_label, count in class_counts.items():
            print(f"{loader_label} Class {class_label}: {count} samples")
            class_weight[class_label] = label_sum / (n_classes * count)

        for weight_label, weight in class_weight.items():
            print(f"{loader_label} Class {weight_label} Weight: {weight:.4f}")

        set_class_weights[loader_label] = class_weight

    return set_class_weights['Training'], set_class_weights['Validation']


def train(model, train_loader, val_loader, criterion, optimizer, device):
    # Training
    model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        # Move inputs and targets to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        class_prob = model(inputs)
        loss = criterion(class_prob, targets)

        # Backward pass / optimization
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        print(f"Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            class_prob = model(inputs)
            loss = criterion(class_prob, targets)
            val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

    return train_loss, val_loss


def save_model(
    model,
    output_dir,
    best_val_loss,
    best_epoch,
    total_epochs,
    epoch_df,
    dataset_file,
    args,
    model_name=None,
    num_classes=2,
    summary_csv_path=None
):

    # Creating folder name
    if model_name is None:
        if num_classes == 2:
            class_type = "singleclass"
        else:
            class_type = "multiclass"
        model_name = (
            f"Auto_qc_{class_type}_lr_{args.learning_rate}_batch_{args.batch_size}"
            f"patience_{args.patience}_seed_{args.seed}_optim_AdamW"
        )
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=False)

    print(f"\n{'='*60}")
    print(f"Saving: {model_name}")
    print(f"{'='*60}")

    # Save  weight
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model, model_path)
    print(f"model: {model_path}")

    # Save learning curve
    curves_path = os.path.join(model_dir, "learning_curves.csv")
    epoch_df.to_csv(curves_path, index=False)
    print(f"Learning curves: {curves_path}")

    shutil.copy(dataset_file, os.path.join(model_dir, os.path.basename(dataset_file)))
    
    # Save metrics summary
    model_params_data = {
        'model_name': [model_name],
        'grader': [args.grader],
        'dataset_manifest': [os.path.basename(dataset_file)],
        
        # Hyperparameters 
        'learning_rate': [args.learning_rate],
        'batch_size': [args.batch_size],
        'patience': [args.patience],
        'epochs_max': [args.epochs],
        'seed': [args.seed],
        'batch_normalization': [args.batch_normalization],
        'output_sigmoid': [args.output_sigmoid]
    }

    model_params_df = pd.DataFrame(model_params_data)
    model_params_path = os.path.join(model_dir, "model_params.csv")
    model_params_df.to_csv(model_params_path, index=False)
    print(f"Model Params: {model_params_path}")

    metrics_data = {
        'model_name': [model_name],
        'best_epoch': [best_epoch],
        'best_val_loss': [best_val_loss],
        'epochs_trained': [total_epochs],
        'train_loss_final': [epoch_df.iloc[-1]['train_loss']],
        'val_loss_final': [epoch_df.iloc[-1]['val_loss']],
        
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(model_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics: {metrics_path}")
    
    # 4. Update experiments summary CSV
    if summary_csv_path is not None:
    
        summary_data = {
            'folder_name': model_name,
            
            # Hyperparameters
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'patience': args.patience,
            'epochs_max': args.epochs,
            'seed': args.seed,
            
            # Performance
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'epochs_trained': total_epochs,
            'train_loss_final': epoch_df.iloc[-1]['train_loss'],
            'val_loss_final': epoch_df.iloc[-1]['val_loss'],
            
        }
        
        if os.path.exists(summary_csv_path):
            summary_df = pd.read_csv(summary_csv_path)
            summary_df = pd.concat([summary_df, pd.DataFrame([summary_data])], ignore_index=True)
        else:
            summary_df = pd.DataFrame([summary_data])
        
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"summary updated: {summary_csv_path}")
        
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train a model to classify the quality of images')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', choices={'SGD', 'sgd', 'AdamW', 'adamw'}, help='optimizer to use')
    parser.add_argument('--reduce_lr', action='store_true', help='whether to use ReduceLROnPlateau scheduler')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--batch_normalization', action='store_true', help='whether to use batch normalization')
    parser.add_argument('--output_sigmoid', action='store_true', help='whether to apply sigmoid activation to output layer (for binary classification)')
    parser.add_argument('--test_split', type=float, default=0.10, help='percentage of dataset to use for testing')
    parser.add_argument('--val_split', type=float, default=0.20, help='percentage of dataset to use for testing')
    parser.add_argument('--dataset_csv_path', type=str, required=True, help='path to datset csv file')
    parser.add_argument('--images_dir', type=str, help='base dir path for images (if not included in csv)')
    parser.add_argument('--output_dir', type=str, required=True, help='dir path to save models')
    parser.add_argument('--model_name', type=str, default=None, help='name for model folder (default: auto-generated based on hyperparameters)')
    parser.add_argument('--grader', choices=['previousStudent', 'shane'], required=True)
  
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.output_sigmoid:
        parser.error('Five-grade CrossEntropyLoss requires logits; omit --output_sigmoid')
    if args.model_name is None:
        args.model_name = f'Auto_qc_{args.grader}_{args.optimizer}_seed_{args.seed}'
    if Path(args.output_dir, args.model_name).exists():
        raise FileExistsError('Model output already exists; choose a new --model_name to preserve results')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    df = pd.read_csv(args.dataset_csv_path)
    required = {'image_path', 'patient_id', 'project', 'split', 'previousStudent_grade', 'shane_grade'}
    if not required.issubset(df.columns):
        raise ValueError('Use the shared manifest from prepare_inputs.py')
    if df[list(required)].isna().any().any() or df.empty:
        raise ValueError('Manifest contains missing values or is empty')
    if not set(df['split']).issubset({'train', 'validation', 'test'}):
        raise ValueError('Unknown manifest split')
    if df.groupby('patient_id')['split'].nunique().max() != 1:
        raise ValueError('Patient leakage in manifest')
    held_out = df['project'].str.casefold() == 'visus-nhp'
    if not held_out.any() or (df.loc[held_out, 'split'] != 'test').any():
        raise ValueError('Entire VISUS-NHP trial must be held out')
    for grader in ('previousStudent', 'shane'):
        if not df[f'{grader}_grade'].isin(range(1, 6)).all():
            raise ValueError(f'Invalid {grader} grades')
    if not all((df['split'] == mode).any() for mode in ('train', 'validation')):
        raise ValueError('Empty train or validation split')
    if not df['image_path'].map(lambda path: Path(path).is_file()).all():
        raise ValueError('Manifest image files are missing')
    # Test images never need decoding or patching during training.
    test_image_count = (df['split'] == 'test').sum()
    df = df[df['split'] != 'test'].copy().reset_index(drop=True)
    df['image_filename'] = df['image_path']
    df['image_grading'] = df[f'{args.grader}_grade']
    df['mode'] = df['split'].replace({'validation': 'val'})

    print('Loading dataset...')
    dataset = ImageQualityDataset(df, transform, label_mapping={grade: grade - 1 for grade in range(1, 6)})

    print('Patching dataset...')
    patched_dataset, df = subsample_subimages(dataset, method='all', df=df)

    # Every patch inherits its patient's prepared split.
    train_dataset = [(images, labels) for (images, labels, mode) in patched_dataset if mode == 'train']
    val_dataset = [(images, labels) for (images, labels, mode) in patched_dataset if mode == 'val']

    # augment training data
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
    ])

    # append
    for i in range(len(train_dataset)):
        train_dataset.append((
            augment(train_dataset[i][0]), train_dataset[i][1]
        ))

    # data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"\nOriginal Dataset size: {len(dataset)}, Patched Dataset size: {len(patched_dataset)}")
    print(f"\nTrain patches: {len(train_dataset)}, Val patches: {len(val_dataset)}; held-out test images: {test_image_count} (not loaded)")
    print(f"\nImage size (H, W): {patched_dataset[0][0].shape[1:]}")

    train_class_weights, _ = class_balance(train_loader, val_loader)
    missing = set(range(5)) - set(train_class_weights)
    if missing:
        raise ValueError(f'No training examples for grades {[index + 1 for index in sorted(missing)]}')
    train_class_weights = [train_class_weights[index] for index in range(5)]

    num_classes = 5

    # outputs
    epoch_df = pd.DataFrame()

    # Initialize the model
    model = IQAnet(
        num_classes=num_classes,
        batch_normalization=args.batch_normalization,
        output_sigmoid=args.output_sigmoid
    )
    model.to(device)

    # Define loss function and optimizer
    weight_tensor = torch.tensor(train_class_weights, dtype=torch.float32).to(device)
    print(f"\nUsing class weights: {weight_tensor}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    scheduler = None
    if args.reduce_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            threshold=0.001
        )

    # Initialise variables for tracking best model and early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    best_model = None
    early_stopping_counter = 0

    # epoch loop
    for epoch in range(args.epochs):

        # Train & val the model for this epoch
        train_loss, val_loss = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device
        )

        print(f"Epoch: {epoch + 1}/{args.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        # save the learning curve
        epoch_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        epoch_df = pd.concat(
            [epoch_df, pd.DataFrame(epoch_dict, index=[0])],
            ignore_index=True,
            axis=0
        )

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print("Early stopping triggered!")
                break

    # Load best model
    model.load_state_dict(best_model)
    model.eval()

    save_model(
        best_model,
        args.output_dir,
        best_val_loss,
        best_epoch,
        epoch + 1,
        epoch_df,
        args.dataset_csv_path,
        args,
        num_classes=num_classes,
        model_name=args.model_name
    )
    evaluate_saved_run(Path(args.output_dir) / args.model_name, grader=args.grader,
                       device=device, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
