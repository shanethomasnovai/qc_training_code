"""Run provenance, reproducible RNG and checkpoint helpers."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    # JSON is a YAML 1.2 subset; emitted .yaml files need no extra parser dependency.
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_copy(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fingerprint(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def code_version():
    command = ['git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT)]
    result = subprocess.run(command + ['rev-parse', 'HEAD'], capture_output=True, text=True)
    status = subprocess.run(command + ['status', '--porcelain'], capture_output=True, text=True)
    return {'commit': result.stdout.strip() or 'unknown', 'dirty': bool(status.stdout.strip()),
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'auto_qc').rglob('*.py'))},
            'dependencies': {name: importlib.metadata.version(name) for name in
                             ('torch', 'torchvision', 'numpy', 'pandas', 'Pillow', 'openpyxl', 'scikit-learn')}}


def class_weights(frame, specification):
    mode = specification['mode']
    if mode == 'none':
        return None
    if mode == 'explicit':
        weights = np.asarray(specification['values'], dtype=float)
    elif mode == 'inverse_frequency':
        if 'target_class_index' not in frame:
            raise ValueError('Inverse frequency requires hard training labels')
        counts = np.bincount(frame.target_class_index.to_numpy(dtype=int), minlength=5)
        if (counts == 0).any():
            raise ValueError('Cannot estimate inverse-frequency weights with an absent training class')
        weights = len(frame) / (5 * counts)
    else:
        raise ValueError(f'Unknown weight mode: {mode}')
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Five positive finite weights required in grade order 1..5')
    return torch.tensor(weights, dtype=torch.float32)


def load_checkpoint(run_dir, device='cpu'):
    from .model import make_model
    run_dir = Path(run_dir)
    config = read_config(run_dir / 'config.yaml')
    saved = torch.load(run_dir / 'checkpoint.pth', map_location='cpu', weights_only=True)
    if saved.get('schema') != 'autoqc_cnn_v1' or saved.get('architecture') != config['model']:
        raise ValueError('Checkpoint architecture/configuration mismatch')
    if saved['dataset_sha256'] != sha256(run_dir / 'dataset.xlsx'):
        raise ValueError('Checkpoint dataset snapshot mismatch')
    model = make_model(config['model'], config['training']['seed'], load_source=False)
    model.load_state_dict(saved['model_state'], strict=True)
    return model.to(device).eval(), config, saved
