import os
import argparse
import pandas as pd
import shutil
import copy
import random
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader


from .model import IQAnet
from .dataset import ImageQualityDataset, subsample_subimages
from .validation import evaluate_saved_run, image_transform
from .soft_labels import TARGET_COLUMNS, construct_continuous_targets, load_soft_manifest, sha256
from .class_weights import baseline_class_weights
from .losses import make_criterion, validate_alpha


def make_loaders(train_dataset, val_dataset, batch_size, seed):
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(val_dataset, batch_size=batch_size,
                       generator=torch.Generator().manual_seed(seed + 1)))


def initialise_model(seed, batch_normalization=False, output_sigmoid=False):
    torch.manual_seed(seed)
    return IQAnet(num_classes=5, batch_normalization=batch_normalization, output_sigmoid=output_sigmoid)


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
    num_classes=5,
    summary_csv_path=None
):

    # Creating folder name
    if model_name is None:
        model_name = f'Auto_qc_soft_{args.loss}_{args.optimizer}_seed_{args.seed}'
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
    settings = vars(args).copy()
    settings.update({'loss': 'continuous_cdw_ce' if args.loss == 'continuous_cdw_ce' else 'CrossEntropyLoss',
                     'loss_selection': args.loss, 'class_weights': args.class_weights,
                     'class_weight_mode': 'baseline' if args.class_weights_from else 'none',
                     'loss_reduction': 'mean',
                     'loss_mean_denominator': 'batch_size',
                     'manifest_sha256': sha256(dataset_file), 'num_classes': 5,
                     'patches_per_image': 9, 'training_horizontal_flip': True,
                     'torch_version': torch.__version__, 'numpy_version': np.__version__})
    if sha256(dataset_file) != args.input_manifest_sha256 or sha256(args.ratings_csv_path) != args.input_ratings_sha256:
        raise ValueError('Inputs changed during training; refusing to save misleading provenance')
    args.target_table.to_csv(Path(model_dir) / 'targets.csv', index=False)
    settings.pop('target_table')
    settings['ratings_sha256'] = sha256(args.ratings_csv_path)
    shutil.copyfile(args.ratings_csv_path, Path(model_dir) / ('ratings_input' + Path(args.ratings_csv_path).suffix))
    split_report = Path(dataset_file).parent / 'split_report.json'
    if split_report.is_file():
        shutil.copyfile(split_report, Path(model_dir) / 'split_report.json')
    (Path(model_dir) / 'run_config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    
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
    parser = argparse.ArgumentParser(description='Train five-grade image quality models from multiple graders')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', choices={'SGD', 'sgd', 'AdamW', 'adamw'}, help='optimizer to use')
    parser.add_argument('--reduce_lr', action='store_true', help='whether to use ReduceLROnPlateau scheduler')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--batch_normalization', action='store_true', help='whether to use batch normalization')
    parser.add_argument('--dataset_csv_path', type=str, required=True, help='path to datset csv file')
    parser.add_argument('--output_dir', type=str, required=True, help='dir path to save models')
    parser.add_argument('--model_name', type=str, default=None, help='name for model folder (default: auto-generated based on hyperparameters)')
    parser.add_argument('--grader', choices=['soft'], default='soft', help='Compatibility flag for combined-grader runs; single-grader training is retired')
    parser.add_argument('--ratings_csv_path', required=True, help='Combined ratings CSV or XLSX joined to the existing manifest')
    parser.add_argument('--grader_columns', nargs='+', help='Explicit *_grade columns; default: discover *_grade in ratings CSV')
    parser.add_argument('--ratings_sheet', default='Grades', help='Worksheet for .xlsx ratings')
    parser.add_argument('--exclude_known_unmatched', action='store_true', help='Explicitly exclude only IDs recorded as unmatched in the saved split_report.json')
    parser.add_argument('--check_only', action='store_true', help='Validate inputs and targets without decoding images or training')
    parser.add_argument('--class_weights_from', help='Soft mode: hard-label baseline run directory supplying the five class weights; omit for unweighted')
    parser.add_argument('--weight_reference_grader', choices=['previousStudent', 'shane'], help='Required if the weight baseline has no saved grader metadata')
    parser.add_argument('--loss', choices=['cross_entropy', 'continuous_cdw_ce'], default='cross_entropy')
    parser.add_argument('--alpha', type=float, default=1.0, help='Positive finite distance exponent for continuous_cdw_ce')
  
    args = parser.parse_args()
    args.output_sigmoid = False
    try:
        validate_alpha(args.alpha)
    except ValueError as error:
        parser.error(str(error))
    if args.loss == 'continuous_cdw_ce':
        if args.class_weights_from or args.weight_reference_grader:
            parser.error('continuous_cdw_ce uses distance weights only; omit class-weight options')
    elif args.alpha != 1.0:
        parser.error('--alpha is only configurable with --loss continuous_cdw_ce')
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.weight_reference_grader and not args.class_weights_from:
        parser.error('--weight_reference_grader requires --class_weights_from')
    if args.model_name is None:
        loss_suffix = '_continuous_cdw_ce' if args.loss == 'continuous_cdw_ce' else ''
        args.model_name = f'Auto_qc_{args.grader}{loss_suffix}_{args.optimizer}_seed_{args.seed}'
    if Path(args.output_dir, args.model_name).exists():
        raise FileExistsError('Model output already exists; choose a new --model_name to preserve results')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print('Using device:', device)

    transform = image_transform()

    args.input_manifest_sha256 = sha256(args.dataset_csv_path)
    args.input_ratings_sha256 = sha256(args.ratings_csv_path)
    df, args.grader_columns, args.unrated_counts = load_soft_manifest(
        args.dataset_csv_path, args.ratings_csv_path, args.grader_columns,
        sheet=args.ratings_sheet, exclude_known_unmatched=args.exclude_known_unmatched)
    if args.loss == 'continuous_cdw_ce':
        df = construct_continuous_targets(df, args.grader_columns)
    args.target_table = df.copy()
    print(f'Selected grader columns: {args.grader_columns}')
    train_class_weights = None
    args.class_weight_source = {'method': 'none'}
    if args.class_weights_from:
        train_class_weights, args.class_weight_source = baseline_class_weights(
            args.class_weights_from, args.weight_reference_grader, args.dataset_csv_path)
        print(f'Baseline class weights (grades 1-5): {train_class_weights}; source: {args.class_weight_source}')
    else:
        print(f'{args.loss}; no class-balancing weights; reduction=mean (over patches)')
    args.target_construction = ('arithmetic mean of available selected integer ratings; unrounded, 1-5 scale'
                                if args.loss == 'continuous_cdw_ce' else
                                'fractions of available selected grader votes in grade order 1-5')
    if args.check_only:
        print('Input and split checks passed; no training performed.')
        return
    # Test images never need decoding or patching during training.
    test_image_count = (df['split'] == 'test').sum()
    df = df[df['split'] != 'test'].copy().reset_index(drop=True)
    df = df[df.has_ratings].reset_index(drop=True)
    df['image_filename'] = df['image_path']
    df['mode'] = df['split'].replace({'validation': 'val'})

    print('Loading dataset...')
    dataset = ImageQualityDataset(df, transform,
                                  target_columns=TARGET_COLUMNS if args.loss == 'cross_entropy' else None,
                                  scalar_target_column='continuous_target' if args.loss == 'continuous_cdw_ce' else None)

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
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, args.batch_size, args.seed)

    print(f"\nOriginal Dataset size: {len(dataset)}, Patched Dataset size: {len(patched_dataset)}")
    print(f"\nTrain patches: {len(train_dataset)}, Val patches: {len(val_dataset)}; held-out test images: {test_image_count} (not loaded)")
    print(f"\nImage size (H, W): {patched_dataset[0][0].shape[1:]}")

    args.class_weights = train_class_weights

    num_classes = 5

    # outputs
    epoch_df = pd.DataFrame()

    # Initialize the model
    model = initialise_model(args.seed, args.batch_normalization, args.output_sigmoid)
    args.initial_model_sha256 = model_fingerprint(model)
    args.randomness = {'policy': 'separate loader generators; reset seed immediately before model initialisation',
                       'model_seed': args.seed, 'training_loader_seed': args.seed, 'validation_loader_seed': args.seed + 1}
    model.to(device)
    args.model_settings = str(model)
    args.preprocessing = {'resize': [768, 768], 'grayscale_channels': 1,
                          'patch_size': 256, 'grid_size': 3, 'method': 'all'}

    # Define loss function and optimizer
    weight_tensor = (torch.tensor(train_class_weights, dtype=torch.float32).to(device)
                     if train_class_weights is not None else None)
    args.effective_class_weights_float32 = weight_tensor.detach().cpu().tolist() if weight_tensor is not None else None
    print(f"\nUsing class weights: {weight_tensor}")

    criterion = make_criterion(args.loss, weight_tensor, args.alpha)

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    args.optimizer_settings = optimizer.defaults.copy()

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
