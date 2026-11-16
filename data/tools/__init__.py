"""Dataset generation and selection scripts."""
from pathlib import Path

LOCAL_FOLDERS = {'images': 'images', 'labels': 'labels', 'datasets': 'datasets', 'models': 'models'}
SOURCE_FILENAMES = {'ganaav': 'ganaav_labels.csv', 'ratings': 'ratings.xlsx',
                    'split': 'manifest.csv', 'manifest': 'manifest.json'}


def local_paths(local_root, config_path):
    if not isinstance(local_root, str) or not local_root.strip():
        raise ValueError('local_root must be a nonempty directory path')
    root = Path(local_root).expanduser()
    if not root.is_absolute():
        root = Path(config_path).resolve().parent / root
    root = root.resolve()
    return root, {name: str(root / folder) for name, folder in LOCAL_FOLDERS.items()}


def require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {description}. Expected file: {path}')
    return path


def root_input(local_root, role):
    folder = 'datasets' if role in ('split', 'manifest') else 'labels'
    return Path(local_root) / LOCAL_FOLDERS[folder] / SOURCE_FILENAMES[role]
