
import argparse
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage


class ImageQualityDataset(Dataset):
    def __init__(self, data, transform=None, label_mapping=None, base_image_dir=None, target_columns=None):
        self.transform = transform
        self.target_columns = target_columns
        self.data = [pd.read_csv(data) if isinstance(data, str) else data][0]
        if label_mapping is not None:
            self.label_mapping = label_mapping
        elif not target_columns:
            self._set_label_mapping()

        self.base_image_dir = base_image_dir

    def _set_label_mapping(self):
        unique_labels = self.data['image_grading'].unique()
        if set(unique_labels) == set(["Good", "Bad"]):
            self.label_mapping = {'Good': 0, 'Bad': 1}
        else:
            self.label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, 'image_filename']
        if self.base_image_dir:
            image_path = os.path.join(self.base_image_dir, image_path)
        quality = (self.data.loc[idx, self.target_columns].to_numpy(dtype=np.float32)
                   if self.target_columns else self.data.loc[idx, 'image_grading'])

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        # define label index
        if self.target_columns:
            return image, torch.tensor(quality, dtype=torch.float32)
        quality = self.label_mapping[quality]

        return image, torch.tensor(quality, dtype=torch.long)


def patch_dataset(dataset):
    '''Extract patches from a dataset of images.

    Args:
        dataset (torch.utils.data.Dataset): dataset of images

    Returns:
        list: list of tuples containing subimages and labels
    '''

    transformed_data = []

    for i in range(len(dataset)):

        image, label = dataset[i]

        # Calculate patch size
        _, height, width = image.size()
        sub_image_width = width // 2
        sub_image_height = height // 2

        # Reshape the image tensor using unfold
        unfolded = torch.nn.functional.unfold(
            image,
            (sub_image_height, sub_image_width),
            stride=(sub_image_height, sub_image_width)
            )

        # Reshape the unfolded tensor to get sub-images
        num_patches = unfolded.shape[1]
        sub_images = unfolded.view(-1, 1, sub_image_height, sub_image_width)

        # Duplicate labels for each sub-image
        labels = [label] * num_patches

        # Append sub-images and labels to the transformed data
        transformed_data.extend(zip(sub_images, labels))

    return transformed_data


def subsample_subimages(
        dataset,
        method='+',
        patch_size=256,
        df=None,
        class_labels=True,
        grid_size=3
        ):

    '''Subsample subimages from a dataset of images.

    Args:
        dataset (torch.utils.data.Dataset): dataset of images
        method (str): Subsample method. Options are '+', 'x', 'all'
        patch_size (int): size of subimages
        df (pandas.DataFrame): dataframe containing image paths and labels
        class_labels (bool): whether dataset contains labels
        grid_size (int): size of grid to subsample subimages from

    Returns:
        list: list of tuples containing subimages and labels
        pandas.DataFrame: Image paths, labels, and subimage positions
    '''

    # Define the positions of the subimages in the 3x3 grid
    if method == '+':
        subimage_positions = [2, 4, 6, 8]
    elif method == 'x':
        subimage_positions = [1, 3, 5, 7, 9]
    elif method == 'all':
        subimage_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    num_patches = len(subimage_positions)

    # Define the transformation to extract subimages
    crop_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    n_samples = len(dataset)

    has_mode = df is not None and 'mode' in df.columns

    transformed_data = []
    for i in range(len(dataset)):

        print(f' {i+1}/{n_samples}', end='\r')

        if class_labels:
            image, label = dataset[i]
        else:
            image = dataset[i]
        
        if has_mode:
            mode = df.loc[i, 'mode']

        subimages = []
        for position in subimage_positions:
            # Calculate the row and column indices of the subimage position
            left = ((position - 1) % grid_size) * patch_size
            top = ((position - 1) // grid_size) * patch_size
            right = left + patch_size
            bottom = top + patch_size

            # Crop and transform the subimage
            subimage = crop_transform(ToPILImage()(image)
                                      .crop((left, top, right, bottom)))
            subimages.append(subimage)

        # Append sub-images and labels (and mode) to the transformed data
        if class_labels:
            labels = [label] * num_patches
            if has_mode:
                modes = [mode] * num_patches
                transformed_data.extend(zip(subimages, labels, modes))
            else:
                transformed_data.extend(zip(subimages, labels))
        else:
            transformed_data = subimages

    if df is not None:
        n = len(df)
        # duplicate rows len(subimage_positions) times
        df = df.loc[df.index.repeat(len(subimage_positions))] \
            .reset_index(drop=True)
        # add subimage positions
        df['subimage_position'] = subimage_positions * n

    else:
        df = pd.DataFrame(
            {
                'subimage_position': subimage_positions * n_samples
            }
        )

    return transformed_data, df


# Creating data splits
def create_split(df, seed):

    np.random.seed(seed)

    df['project'] = df['image_filename'].str.split('_').str[0]
    df['patient_id'] = df['image_filename'].str.split('_').str[:3].str.join('_')

    # Getting patient info
    patient_info = df.groupby('patient_id').agg({
        'project': 'first',
        'image_filename': 'count'
    }).reset_index()
    patient_info.columns = ['patient_id', 'project', 'num_images']

    test_patients = []
    train_patients = []
    val_patients = []

    filtered_df = patient_info[patient_info['project'] == "ASTELLAS"]
    test_patients += list(filtered_df['patient_id'])

    filtered_df = df[df['image_grading'] == 5]
    filtered_df = filtered_df[filtered_df['project'] != "ASTELLAS"]
    prefixes = (
        filtered_df["image_filename"]
        .str.split("_")
        .str[:3]
        .str.join("_")
    )
    grade_5_patients = list(set(prefixes))
    np.random.shuffle(grade_5_patients)

    train_patients.append(grade_5_patients[0])
    val_patients.append(grade_5_patients[1])
    test_patients.append(grade_5_patients[2])

    #  Split remaining patients randomly 
    assigned = train_patients + val_patients + test_patients
    remaining = [p for p in patient_info['patient_id'] if p not in assigned]
    remaining_info = patient_info[patient_info['patient_id'].isin(remaining)]
    
    remaining_test = max(0.1*len(patient_info) - len(test_patients), 0)
    remaining_test_size = remaining_test / len(remaining)

    # Split remaining patients: first separate test (20%)
    try:
        train_val_pool, add_test = train_test_split(
            remaining,
            test_size=remaining_test_size,  # 20% to test
            random_state=seed,
            stratify=remaining_info['project']
        )
        test_patients.extend(add_test)

        # Split rest into train 87.5% and val 12.5%
        pool_info = patient_info[patient_info['patient_id'].isin(train_val_pool)]
        add_train, add_val = train_test_split(
            train_val_pool,
            test_size=0.125,  # 12.5% of the 80% = 10% overall
            random_state=seed,
            stratify=pool_info['project']
        )

    except ValueError:
        train_val_pool, add_test = train_test_split(
            remaining,
            test_size=0.20,
            random_state=seed
        )
        test_patients.extend(add_test)

        add_train, add_val = train_test_split(
            train_val_pool,
            test_size=0.125,
            random_state=seed
        )

    train_patients.extend(add_train)
    val_patients.extend(add_val)

    print(len(train_patients), len(val_patients), len(test_patients))

    # Assign mode
    df['mode'] = df['patient_id'].apply(
        lambda x: 'train' if x in train_patients else 'val' if x in val_patients else 'test'
    )

    return df


# Validate splits
def validate_split(df):
    # Minimum counts required
    min_counts = {
        'train': {'1-4': 10, '5': 1},
        'val': {'1-4': 5, '5': 1},
        'test': {'1-4': 3, '5': 1}
    }

    for split_name in ['train', 'val', 'test']:
        split_data = df[df['mode'] == split_name]
        print(len(split_data))

        # Check each grade distribution
        for grade in range(1, 6):
            count = len(split_data[split_data['image_grading'] == grade])
            required = min_counts[split_name]['5'] if grade == 5 else min_counts[split_name]['1-4']

            if count < required:
                return False, f"{split_name} has only {count} Grade {grade} images (need {required})"

        # Check split size 
        pct = len(split_data) / len(df) * 100
        if split_name == 'train' and not (65 <= pct <= 75):
            return False, f"train size {pct:.1f}% out of range (65-75%)"
        elif split_name == 'val' and not (8 <= pct <= 15):
            return False, f"val size {pct:.1f}% out of range (8-15%)"
        elif split_name == 'test' and not (15 <= pct <= 25):
            return False, f"test size {pct:.1f}% out of range (15-25%)"
    

    # Check if patient leakage
    train_pts = set(df[df['mode'] == 'train']['patient_id'].unique())
    val_pts = set(df[df['mode'] == 'val']['patient_id'].unique())
    test_pts = set(df[df['mode'] == 'test']['patient_id'].unique())

    if len(train_pts & val_pts) > 0 or len(train_pts & test_pts) > 0 or len(val_pts & test_pts) > 0:
        return False, "Patient leakage"

    return True, "Valid"


# Generate splits
def generate_monte_carlo_splits(csv_path, output_dir, n_valid_splits=5, max_attempts=100, seed=None):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"generating {n_valid_splits} Monte Carlo splits (Target: 70/10/20)")

    # Load data once
    df_original = pd.read_csv(csv_path)
    
    valid_splits = []
    attempt = 1

    if seed is None:
        seed = np.random.randint(0, 100000)
    
    while len(valid_splits) < n_valid_splits and attempt < max_attempts:
        
        # Create split
        df_split = df_original.copy()
        df_split = create_split(df_split, seed)
        
        # Validate
        is_valid, message = validate_split(df_split)
        
        if is_valid:
            # Save split
            output_file = os.path.join(output_dir, f'multi_class_qc_dataset_seed_{seed}.csv')
            df_split.to_csv(output_file, index=False)
            
            # Get split percentages
            train_pct = len(df_split[df_split['mode'] == 'train']) / len(df_split) * 100
            val_pct = len(df_split[df_split['mode'] == 'val']) / len(df_split) * 100
            test_pct = len(df_split[df_split['mode'] == 'test']) / len(df_split) * 100
            
            valid_splits.append(seed)
            print(f"Attempt {attempt}: Seed {seed} valid ({train_pct:.1f}/{val_pct:.1f}/{test_pct:.1f}) saved")
        else:
            print(f"Attempt {attempt}: Seed {seed} failed ({message})")
        
        attempt += 1
        seed += 1
    
    # Summary
    print("\n")
    print("summary of Monte Carlo splits (RANDOM TEST SET)")
    print(f"\nValid seeds: {valid_splits}")
    
    # Save summary
    summary_file = os.path.join(output_dir, 'splits_summary_random.txt')
    with open(summary_file, 'w') as f:
        f.write("Monte Carlo Splits Summary - RANDOM TEST SET\n")
        f.write("\n")
        f.write("Target split: 70/10/20 (train/val/test)\n")
        f.write("Valid splits: {len(valid_splits)}\n")
        f.write("\nValid seeds:\n")
        for seed in valid_splits:
            f.write(f"  - {seed}\n")
    
    print(f"\n Summary saved to: {summary_file}")


# Run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Monte Carlo data splits')
    parser.add_argument('--dataset_csv_path', type=str, required=True, help='path to dataset containing images and labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to output csv files')
    parser.add_argument('--start_seed', type=int, default=1)

    args = parser.parse_args()

    generate_monte_carlo_splits(
        csv_path=args.dataset_csv_path,
        output_dir=args.output_dir,
        n_valid_splits=1,
        max_attempts=100,
        seed=args.start_seed
    )
