"""Original image loader and patch functions, with prepared patches for current runs."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from data.labels import VOTES, ORDINAL


class ImageQualityDataset(Dataset):
    def __init__(self, data, transform=None, label_mapping=None):
        self.transform = transform
        self.data = pd.read_csv(data) if isinstance(data, (str, Path)) else data
        self.data = self.data.reset_index(drop=True)
        self.label_mapping = label_mapping
        # Accept the pasted original columns and the archived branch's names.
        self.image_column = 'Image' if 'Image' in self.data else 'image_filename'
        self.label_column = 'Quality' if 'Quality' in self.data else 'image_grading'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.loc[idx, self.image_column]
        quality = self.data.loc[idx, self.label_column]

        with Image.open(image_path) as source:
            image = self.transform(source) if self.transform else source.copy()

        if self.label_mapping:
            quality = self.label_mapping[quality]

        return image, quality


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
        # Unfold stores patches as columns; transpose before reshaping.
        sub_images = unfolded.transpose(0, 1).reshape(
            num_patches, image.size(0), sub_image_height, sub_image_width)

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
        grid_size=3,
        verbose=True
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
    else:
        raise ValueError(f'Unknown patch method: {method}')

    num_patches = len(subimage_positions)

    # Define the transformation to extract subimages
    crop_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    n_samples = len(dataset)

    has_mode = df is not None and 'mode' in df.columns

    transformed_data = []
    for i in range(len(dataset)):

        if verbose:
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
            transformed_data.extend(subimages)

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

# Current experiments supply precomputed scalar or vector targets.


def target_tensor(frame, kind):
    if kind in ('hard_ganaav', 'hard_shane', 'ceiling_mean'):
        return torch.tensor(frame.target_class_index.to_numpy(dtype=np.int64), dtype=torch.long)
    columns = ORDINAL if kind == 'ordinal' else VOTES if kind == 'soft_categorical' else ['target_mean_grade']
    values = torch.tensor(frame[columns].to_numpy(dtype=np.float32))
    return values.squeeze(1) if kind == 'exact_mean' else values


def image_transform(input_mode='patches'):
    if input_mode not in ('patches', 'whole_image'):
        raise ValueError('input_mode must be patches or whole_image')
    size = 256 if input_mode == 'whole_image' else 768
    return transforms.Compose([transforms.Resize((size, size)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])


def extract_patches(path):
    with Image.open(path) as image:
        tensor = image_transform()(image)
    patches, _ = subsample_subimages([tensor], method='all', class_labels=False, verbose=False)
    return torch.stack(patches)


def extract_inputs(path, input_mode='patches'):
    """Use identical preprocessing during training and checkpoint evaluation."""
    if input_mode == 'patches':
        return extract_patches(path)
    with Image.open(path) as image:
        return image_transform(input_mode)(image).unsqueeze(0)


class ImagePatchDataset(Dataset):
    """Prepare nine original crops or one resized whole image, with flips on access.

    ToPILImage -> ToTensor already quantises pixels to uint8 / 255. Store those
    exact bytes, then use the same float division on access. This uses a quarter
    of the float cache's memory, with no duplicated cache for horizontal flips.
    """
    def __init__(self, frame, kind, augment=False, input_mode='patches'):
        self.frame = frame.reset_index(drop=True)
        self.targets = target_tensor(self.frame, kind)
        self.augment = augment
        self.inputs_per_image = 1 if input_mode == 'whole_image' else 9
        self.base_length = len(frame) * self.inputs_per_image
        if frame.empty or frame.split.nunique() != 1 or not frame.eligible.all():
            raise ValueError('Supply one nonempty eligible split before extracting patches')
        data = self.frame.assign(Image=self.frame.image_path, Quality=list(self.targets))
        self.images = ImageQualityDataset(data, transform=image_transform(input_mode))
        self._patches = []
        for image_index in range(len(self.images)):
            if input_mode == 'whole_image':
                pixels = self.images[image_index][0].unsqueeze(0)
            else:
                patches, _ = subsample_subimages([self.images[image_index]], method='all', verbose=False)
                pixels = torch.stack([image for image, _ in patches])
            self._patches.append(pixels.mul(255).round_().to(torch.uint8))
        self.cache_nbytes = sum(p.numel() * p.element_size() for p in self._patches)

    def __len__(self):
        return self.base_length * (2 if self.augment else 1)

    def patches(self, image_index):
        return self._patches[image_index].to(torch.float32).div_(255)

    def __getitem__(self, index):
        flipped = index >= self.base_length
        image_index, patch = divmod(index % self.base_length, self.inputs_per_image)
        value = self._patches[image_index][patch].to(torch.float32).div_(255)
        return (value.flip(-1) if flipped else value), self.targets[image_index]
