import os
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from model import IQAnet
from dataset import ImageQualityDataset, subsample_subimages




def extract_metrics(y, yhat):

    metrics = {
        'cm' : str(confusion_matrix(y, yhat, labels=[1, 0])),
        'precision' : precision_score(y, yhat, pos_label=1),
        'recall' : recall_score(y, yhat, pos_label=1),
        'f1' : f1_score(y, yhat),
        'accuracy' : sum([1 if label == prediction else 0 for label, prediction in zip(y, yhat)]) / len(y)
    }

    return metrics


def class_balance(train_loader, val_loader):

    set_class_weights = {'Training': None, 'Validation': None}
    
    for loader_label, loader in zip(['Training', 'Validation'], [train_loader, val_loader]):
        # print the class distribution for the fold
        class_counts = {}
        for _, labels in loader:
            for label in labels:

                if label.item() not in class_counts:
                    class_counts[label.item()] = 0
                class_counts[label.item()] += 1

        label_sum = sum(class_counts.values())
        n_classes = len(class_counts)

        # Print the class distribution
        class_weight = {}
        for class_label, count in class_counts.items():
            print(f"{loader_label} Class {class_label}: {count} samples")
            class_weight[class_label] = label_sum / (n_classes * count)

        for weight_label, weight in class_weight.items():
            print(f"{loader_label} Class {weight_label} Weight: {weight}")

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




def main():
    parser = argparse.ArgumentParser(description='Train a model to classify the quality of images')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--k_folds', type=int, default=10, help='number of folds for k-fold cross validation')
    parser.add_argument('--test_split', type=float, default=0.10, help='percentage of dataset to use for testing')
    parser.add_argument('--csv_path', type=str, help='path to label csv file')
    parser.add_argument('--model_path', type=str, help='dir path to save models')
    parser.add_argument('--learning_path', type=str, help='dir path to save csv outputs')
    args = parser.parse_args()


    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print('Using device:', device)


    label_mapping = {
        'Good': 0, 
        'Bad': 1
    }

    transform = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # setup output dirs
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.learning_path):
        os.makedirs(args.learning_path)

    df = pd.read_csv(args.csv_path)

    print('Loading dataset...')
    dataset = ImageQualityDataset(df, transform, label_mapping)
    print('Patching dataset...')
    patched_dataset, df = subsample_subimages(dataset, method='all', df=df)

    # split test from train
    test_size = int(len(patched_dataset) * args.test_split)
    train_size = len(patched_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(patched_dataset, [train_size, test_size])



    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"\nOriginal Dataset size: {len(dataset)}, Patched Dataset size: {len(patched_dataset)}")
    print(f"\nTrain size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    print(f"\nImage size (H, W): {patched_dataset[0][0].shape[1:]}")


    # outputs
    epoch_df = pd.DataFrame()
    test_df = pd.DataFrame()

    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
    ])    


    # kfold loop
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_ix, val_ix) in enumerate(kfold.split(train_dataset)):
        
        # augment training data
        train_set = [patched_dataset[i] for i in train_ix]

        # append
        for i in range(len(train_set)):
            train_set.append((augment(train_set[i][0]), train_set[i][1]))

        # Create data loaders for this fold
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size
        )
        val_loader = DataLoader(
            [patched_dataset[i] for i in val_ix], 
            batch_size=args.batch_size
        )

        print(f"\nFold {fold + 1}, Training size: {len(train_ix)}, Augmented size: {len(train_set)}, Validation size: {len(val_ix)}")

        # Print the class distribution for this fold
        train_class_ratio,_ = class_balance(train_loader, val_loader)

        # Initialize the model
        model = IQAnet()
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(list(train_class_ratio.values()))).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

        # Initialize variables for tracking best model and early stopping
        best_val_loss = float('inf')
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

            print(f"Fold {fold + 1} Epoch {epoch + 1}/{args.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # save the learning curve
            epoch_dict = {
                'fold' : fold + 1, 
                'epoch' : epoch + 1, 
                'train_loss' : train_loss, 
                'val_loss' : val_loss
            }

            epoch_df = pd.concat(
                [epoch_df, pd.DataFrame(epoch_dict, index=[0])], 
                ignore_index=True, 
                axis=0
            )
            
            pd.DataFrame(epoch_df).to_csv(
                f"{args.learning_path}/kfold_learning_curves.csv", 
                index=False
            )


            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print("Early stopping triggered!")
                    break


        # Save the best model from this fold
        best_model_path = f"{args.model_path}/best_model_fold_{fold}.pth"
        torch.save(best_model, best_model_path)

        # Load the best model from this fold
        model.load_state_dict(best_model)
        model.eval()

        # Evaluate the best model from this fold on the test set
        test_loss = 0.0

        y_all = []
        yhat_all = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                class_prob = model(inputs)
                loss = criterion(class_prob, targets)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(class_prob, 1)
                y_all.extend(targets.cpu().numpy())
                yhat_all.extend(preds.cpu().numpy())
            test_loss /= len(test_loader.dataset)
            print(f"Test Loss = {test_loss:.4f}")


        print('Assesing held-out test set...') 
        test_dict = {
            'fold' : fold + 1,
            'epoch' : epoch + 1,
            'test_loss' : test_loss,
        }

        # extract metrics & save 
        test_dict = test_dict | extract_metrics(y_all, yhat_all)
        
        test_df = pd.concat(
            [test_df, pd.DataFrame(test_dict, index=[0])], 
            ignore_index=True, 
            axis=0
        )

        test_df.to_csv(f"{args.learning_path}/kfold_test_metrics.csv", index=False)
        print(f"Fold {fold + 1} Test Loss: {test_loss:.4f}")




if __name__ == '__main__':
    main()