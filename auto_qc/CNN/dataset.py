"""Image-level workbook records and the original 3x3 IQAnet patch extraction."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

VOTES = [f'target_grade_{k}' for k in range(1, 6)]
ORDINAL = [f'target_above_grade_{k}' for k in range(1, 5)]
KINDS = ('hard_ganaav', 'hard_shane', 'ceiling_mean', 'exact_mean', 'soft_categorical', 'ordinal')


def image_key(value):
    text = str(value).strip().replace('\\', '/')
    path = Path(text)
    return path.stem if path.suffix.lower() in ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp') else path.name


def canonical_graders(columns):
    return ['ganaav_grade' if c == 'previousStudent_grade' else c for c in columns]


def canonical_columns(frame):
    frame = frame.copy()
    if 'previousStudent_grade' in frame:
        if 'ganaav_grade' in frame and not np.allclose(pd.to_numeric(frame.ganaav_grade, errors='coerce'),
                pd.to_numeric(frame.previousStudent_grade, errors='coerce'), equal_nan=True):
            raise ValueError('Conflicting ganaav / previousStudent aliases')
        frame = frame.rename(columns={'previousStudent_grade': 'ganaav_grade'}) if 'ganaav_grade' not in frame else frame.drop(columns='previousStudent_grade')
    if 'split' not in frame and 'mode' in frame:
        frame['split'] = frame['mode']
    if 'split' in frame:
        frame['split'] = frame['split'].replace({'val': 'validation'})
        frame['mode'] = frame['split'].replace({'validation': 'val'})
    return frame


def ratings_array(frame, graders):
    graders = canonical_graders(graders)
    if not graders or len(set(graders)) != len(graders) or not set(graders).issubset(frame.columns):
        raise ValueError('Select unique existing grader columns')
    arrays = []
    for name in graders:
        raw = frame[name]
        missing = raw.isna() | raw.astype(str).str.strip().eq('')
        values = pd.to_numeric(raw.where(~missing), errors='coerce')
        invalid = ~missing & (~values.isin(range(1, 6)) | raw.map(lambda v: isinstance(v, (bool, np.bool_))))
        if invalid.any():
            raise ValueError(f'Invalid integer 1..5 rating in {name}: {frame.loc[invalid, "image_id"].tolist()[:5]}')
        arrays.append(values.to_numpy(dtype=float))
    return np.stack(arrays, axis=1)


def generate_targets(frame, kind, graders, mean_missing_policy='require_all'):
    if kind not in KINDS:
        raise ValueError(f'Unsupported target kind: {kind}')
    result = canonical_columns(frame)
    graders = canonical_graders(graders)
    ratings = ratings_array(result, graders)
    counts = np.isfinite(ratings).sum(1)
    votes = np.stack([(ratings == k).sum(1) for k in range(1, 6)], axis=1)
    votes = np.divide(votes, counts[:, None], out=np.full_like(votes, np.nan, dtype=float), where=counts[:, None] > 0)
    result[VOTES] = votes
    result[ORDINAL] = np.stack([votes[:, k:].sum(1) for k in range(1, 5)], axis=1)
    result['target_binary_unacceptable'] = result[ORDINAL[1]]
    result['rating_count'] = counts
    result['eligible'] = counts > 0
    if kind in ('ceiling_mean', 'exact_mean'):
        if mean_missing_policy not in ('require_all', 'available'):
            raise ValueError('Explicit mean missing-rating policy required')
        result['eligible'] = counts == len(graders) if mean_missing_policy == 'require_all' else counts > 0
        mean = np.divide(np.nansum(ratings, axis=1), counts, out=np.full(len(frame), np.nan), where=counts > 0)
        result['target_mean_grade'] = mean
        if kind == 'ceiling_mean':
            result['target_grade'] = np.ceil(mean)
    elif kind in ('hard_ganaav', 'hard_shane'):
        if graders != (['ganaav_grade'] if kind == 'hard_ganaav' else ['shane_grade']):
            raise ValueError('Hard target grader does not match kind')
        result['target_grade'] = ratings[:, 0]
    if 'target_grade' in result:
        result['target_class_index'] = result.target_grade - 1
        result['image_grading'] = result.target_grade
    elif kind == 'exact_mean':
        result['image_grading'] = result.target_mean_grade
    result['target_kind'] = kind
    result['exclusion_reason'] = np.where(result.eligible, '', 'missing required selected ratings')
    target_names = VOTES + ORDINAL + ['target_binary_unacceptable', 'target_mean_grade', 'target_grade', 'target_class_index']
    for name in set(target_names) & set(result):
        result.loc[~result.eligible, name] = np.nan
    return result


def validate_records(frame, check_files=False, held_out='VISUS-NHP'):
    required = ['image_id', 'image_path', 'patient_id', 'project', 'project_site', 'split']
    if frame.empty or not set(required).issubset(frame):
        raise ValueError('Explicit image/patient/project/site/split records required before patching')
    if frame[required].isna().any().any() or frame[required].astype(str).apply(lambda c: c.str.strip().eq('')).any().any():
        raise ValueError('Empty identity / split metadata')
    if frame.image_id.duplicated().any() or frame.image_path.map(lambda p: str(Path(p).resolve()).casefold()).duplicated().any():
        raise ValueError('Duplicate image ID or path')
    if not frame.image_id.eq(frame.image_path.map(image_key)).all():
        raise ValueError('Image ID/path mismatch; preserve full variant suffixes')
    if not frame.split.isin(('train', 'validation', 'test')).all() or frame.groupby('patient_id').split.nunique().gt(1).any():
        raise ValueError('Invalid split or patient overlap')
    if not frame.patient_id.eq(frame.image_id.str.split('_').str[:3].str.join('_')).all():
        raise ValueError('Patient identity differs from original filename crosswalk')
    if held_out and frame.loc[frame.project.eq(held_out), 'split'].ne('test').any():
        raise ValueError(f'{held_out} must remain held out')
    if check_files and not frame.image_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError('Missing image paths')


def logical_hash(frame):
    data = frame.sort_values('image_id').sort_index(axis=1).copy()
    for name in data.select_dtypes(include='number'):
        data[name] = data[name].astype(float)
    return hashlib.sha256(data.to_json(orient='split', index=False, double_precision=15).encode()).hexdigest()


def save_workbook(path, frame, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    metadata = dict(metadata, logical_content_sha256=logical_hash(frame))
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        frame.to_excel(writer, index=False, sheet_name='dataset')
        pd.DataFrame({'key': list(metadata), 'value': [json.dumps(v, allow_nan=False) for v in metadata.values()]}).to_excel(writer, index=False, sheet_name='metadata')


def load_workbook(path, check_files=False):
    frame = canonical_columns(pd.read_excel(path, sheet_name='dataset')).fillna({'exclusion_reason': ''})
    values = pd.read_excel(path, sheet_name='metadata')
    metadata = {row.key: json.loads(row.value) for row in values.itertuples()}
    validate_records(frame, check_files, metadata.get('held_out_project'))
    expected = generate_targets(frame.drop(columns=[c for c in frame if c.startswith('target_')] + ['image_grading'], errors='ignore'),
                                metadata['target_kind'], metadata['graders'], metadata.get('mean_missing_policy', 'require_all'))
    for name in [c for c in expected if c.startswith('target_') and c != 'target_kind'] + ['rating_count', 'eligible']:
        if name not in frame or not np.allclose(pd.to_numeric(frame[name]), pd.to_numeric(expected[name]), rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f'Workbook consumed target disagrees with source ratings: {name}')
    if logical_hash(frame.fillna({'exclusion_reason': ''})) != metadata['logical_content_sha256']:
        raise ValueError('Workbook logical content hash mismatch')
    return frame, metadata


def target_tensor(frame, kind):
    if kind in ('hard_ganaav', 'hard_shane', 'ceiling_mean'):
        return torch.tensor(frame.target_class_index.to_numpy(dtype=np.int64), dtype=torch.long)
    columns = ORDINAL if kind == 'ordinal' else VOTES if kind == 'soft_categorical' else ['target_mean_grade']
    values = torch.tensor(frame[columns].to_numpy(dtype=np.float32))
    return values.squeeze(1) if kind == 'exact_mean' else values


def image_transform():
    return transforms.Compose([transforms.Resize((768, 768)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])


def extract_patches(path):
    # Keep original Resize -> Grayscale -> Tensor -> PIL -> grid-crop sequence.
    with Image.open(path) as image:
        tensor = image_transform()(image)
    pil = transforms.ToPILImage()(tensor)
    return torch.stack([transforms.ToTensor()(pil.crop((c*256, r*256, (c+1)*256, (r+1)*256))) for r in range(3) for c in range(3)])


class ImageQualityDataset(Dataset):
    """Already split image records exposed as patches, then appended training flips."""
    def __init__(self, frame, kind, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.targets = target_tensor(self.frame, kind)
        self.augment = augment
        self.base_length = len(frame) * 9
        if frame.empty or frame.split.nunique() != 1 or not frame.eligible.all():
            raise ValueError('Supply one nonempty eligible split before extracting patches')

    def __len__(self):
        return self.base_length * (2 if self.augment else 1)

    @lru_cache(maxsize=32)
    def patches(self, image_index):
        return extract_patches(self.frame.iloc[image_index].image_path)

    def __getitem__(self, index):
        flipped = index >= self.base_length
        image_index, patch = divmod(index % self.base_length, 9)
        value = self.patches(image_index)[patch]
        return (value.flip(-1) if flipped else value), self.targets[image_index]
