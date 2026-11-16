"""Read editable training settings and save resolved experiment configurations."""
import copy
import io
import math
import re
import tomllib
from pathlib import Path

from data.tools import local_paths, require_file, root_input, SOURCE_FILENAMES
from data.labels import KINDS, canonical_graders, load_workbook
from .utils import ROOT, read_config, sha256

MODEL_DEFAULTS = {'ker_size': 5, 'n_kers': 64, 'n1_nodes': 600, 'n2_nodes': 400,
                  'num_grades': 5, 'batch_normalization': False, 'output_sigmoid': False,
                  'dropout_p': .5}
HEAD_LOSSES = {'or_soft': 'ordinal_bce', 'coral': 'ordinal_bce',
               'or_soft_binary': 'joint_bce', 'frozen_binary': 'binary_bce'}
TARGET_LOSSES = {'hard_ganaav': 'hard_ce', 'hard_shane': 'hard_ce', 'ceiling_mean': 'hard_ce',
                 'soft_categorical': 'soft_ce', 'exact_mean': 'continuous_cdw_ce'}
TRAINING_DEFAULTS = {'seed': 42, 'epochs': 100, 'batch_size': 16, 'patience': 10, 'device': 'auto'}


def preprocessing_settings(input_mode='patches'):
    if input_mode not in ('patches', 'whole_image'):
        raise ValueError('data.input_mode must be patches or whole_image')
    settings = {'resize': [768, 768], 'grayscale_channels': 1, 'patch_size': 256,
                'grid_size': 3, 'method': 'all', 'horizontal_flip': True}
    if input_mode == 'whole_image':
        settings.update(input_mode=input_mode, resize=[256, 256], grid_size=1)
    return settings


def _keys(mapping, allowed, section):
    if not isinstance(mapping, dict):
        raise ValueError(f'{section} must be a mapping')
    unknown = mapping.keys() - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}')


def load_training_config(path, overrides=None):
    path = Path(path).resolve()
    if path.suffix.lower() == '.toml':
        with path.open('rb') as handle:
            settings = tomllib.load(handle)
    else:
        settings = read_config(path)
    if isinstance(settings, dict) and settings.get('schema') == 'autoqc_cnn_config_v1':
        if 'local_root' in settings and settings.get('dataset', {}).get('image_directory') != settings.get('effective_paths', {}).get('images'):
            raise ValueError('Resolved local_root config has conflicting image_directory')
        config = copy.deepcopy(settings)
        _apply_overrides(config, overrides)
        return config
    _keys(settings, {'local_root', 'experiment_name', 'model', 'data', 'training', 'optimizer', 'scheduler', 'loss', 'evaluation'}, 'config sections')
    data = settings['data']
    _keys(data, {'dataset', 'target', 'graders', 'image_directory', 'input_mode'}, 'data')
    preprocessing = preprocessing_settings(data.get('input_mode', 'patches'))
    root = None
    paths = {}
    if 'local_root' in settings:
        if 'image_directory' in data:
            raise ValueError('local_root conflicts with legacy data.image_directory; remove image_directory')
        root, paths = local_paths(settings['local_root'], path)
        relative = Path(data['dataset'])
        dataset = (Path(paths['datasets']) / relative).resolve()
        if relative.is_absolute() or not dataset.is_relative_to(Path(paths['datasets'])):
            raise ValueError('With local_root, data.dataset must be relative to its datasets/ folder')
        require_file(dataset, 'dataset workbook')
        image_directory = paths['images']
    else:
        dataset = ROOT / data['dataset']
        image_directory = data['image_directory']
    if data['target'] not in KINDS:
        raise ValueError(f'data.target must be one of {KINDS}')
    graders = canonical_graders(data['graders']) if 'graders' in data else None
    options = {'target_kind': data['target']}
    if graders is not None:
        options['graders'] = graders
    if root is not None:
        options['local_root'] = root
    frame, metadata = load_workbook(dataset, **options)
    model_options = settings['model']
    _keys(model_options, {*MODEL_DEFAULTS, 'head', 'source', 'backbone', 'pretrained'}, 'model')
    model = {**MODEL_DEFAULTS, **model_options}
    backbone = model.get('backbone', 'iqanet')
    if backbone not in ('iqanet', 'iqanet4', 'resnet18'):
        raise ValueError('model.backbone must be iqanet, iqanet4 or resnet18')
    if not isinstance(model.get('pretrained', False), bool):
        raise ValueError('model.pretrained must be true or false')
    if backbone == 'resnet18':
        if model['head'] == 'frozen_binary':
            raise ValueError('ResNet-18 does not support frozen_binary')
        if set(model_options) & {'ker_size', 'n_kers', 'n1_nodes', 'n2_nodes', 'batch_normalization'}:
            raise ValueError('ResNet-18 uses its standard layers and batch normalization; remove IQAnet-specific settings')
    elif model.get('pretrained', False):
        raise ValueError('Pretrained weights are only supported for resnet18')
    if model['head'] not in {*HEAD_LOSSES, 'five_logits'}:
        raise ValueError('model.head must be five_logits, or_soft, coral, or_soft_binary or frozen_binary')
    for key in ('ker_size', 'n_kers', 'n1_nodes', 'n2_nodes', 'num_grades'):
        if isinstance(model[key], bool) or not isinstance(model[key], int) or model[key] <= 0:
            raise ValueError(f'model.{key} must be a positive integer')
    if model['num_grades'] != 5 or model['output_sigmoid'] is not False:
        raise ValueError('Supported methods require five grades and raw logits (output_sigmoid: false)')
    if not isinstance(model['dropout_p'], (int, float)) or not 0 <= model['dropout_p'] < 1:
        raise ValueError('model.dropout_p must be in [0, 1)')
    if not isinstance(model['batch_normalization'], bool):
        raise ValueError('model.batch_normalization must be boolean')
    expected_loss = HEAD_LOSSES.get(model['head']) if data['target'] == 'ordinal' else TARGET_LOSSES[data['target']]
    if (data['target'] == 'ordinal') != (model['head'] != 'five_logits'):
        raise ValueError('Target/head mismatch: ordinal needs an ordinal head; other targets need five_logits')
    loss_options = settings.get('loss', {})
    _keys(loss_options, {'name', 'class_weights', 'binary_loss_weight', 'alpha'}, 'loss')
    loss = {'name': expected_loss, 'class_weights': {'mode': 'none'},
            'threshold_weights': [1, 1, 1, 1] if data['target'] == 'ordinal' else None,
            'epoch_aggregation': 'example_mean', **loss_options}
    if loss['name'] != expected_loss:
        raise ValueError(f'Target/head/loss mismatch: expected {expected_loss}')
    weights = loss['class_weights']
    _keys(weights, {'mode', 'values'}, 'loss.class_weights')
    if weights.get('mode') not in ('none', 'explicit', 'inverse_frequency'):
        raise ValueError('Class weight mode must be none, explicit or inverse_frequency')
    if weights['mode'] != 'none' and data['target'] in ('ordinal', 'exact_mean'):
        raise ValueError('Ordinal/CDW objectives do not accept class weighting')
    if weights['mode'] == 'inverse_frequency' and expected_loss != 'hard_ce':
        raise ValueError('inverse_frequency requires a hard-label target; use explicit weights for soft_ce')
    if weights['mode'] == 'explicit':
        if not isinstance(weights.get('values'), list) or len(weights['values']) != 5:
            raise ValueError('Explicit weights require five values in grade order 1..5')
        for value in weights['values']:
            _positive(value, 'class weight')
    elif 'values' in weights:
        raise ValueError('class_weights.values is only supported with mode: explicit')
    if expected_loss == 'hard_ce' and weights['mode'] != 'none':
        loss['epoch_aggregation'] = 'global_target_weight_sum'
    if expected_loss == 'joint_bce':
        loss.setdefault('binary_loss_weight', 1.)
        _positive(loss['binary_loss_weight'], 'binary_loss_weight', True)
    elif 'binary_loss_weight' in loss_options:
        raise ValueError('binary_loss_weight is only used with or_soft_binary')
    if expected_loss == 'continuous_cdw_ce':
        loss.setdefault('alpha', 1.)
        _positive(loss['alpha'], 'alpha')
    elif 'alpha' in loss_options:
        raise ValueError('alpha is only used with continuous_cdw_ce')
    optimizer = dict(settings.get('optimizer', {}))
    optimizer_name = optimizer.pop('name', 'AdamW')
    if optimizer_name not in ('AdamW', 'SGD'):
        raise ValueError('optimizer.name must be AdamW or SGD')
    allowed_optimizer = ({'lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused'}
                         if optimizer_name == 'AdamW' else {'lr', 'weight_decay', 'momentum', 'dampening', 'nesterov', 'maximize', 'foreach', 'differentiable', 'fused'})
    _keys(optimizer, allowed_optimizer, 'optimizer')
    optimizer.setdefault('lr', .0001)
    _positive(optimizer['lr'], 'optimizer.lr')
    if 'weight_decay' in optimizer:
        _positive(optimizer['weight_decay'], 'optimizer.weight_decay', True)
    scheduler = dict(settings.get('scheduler', {}))
    _keys(scheduler, {'enabled', 'name', 'mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lr', 'eps'}, 'scheduler')
    enabled = scheduler.pop('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('scheduler.enabled must be boolean')
    if scheduler.pop('name', 'ReduceLROnPlateau') != 'ReduceLROnPlateau':
        raise ValueError('Only ReduceLROnPlateau is supported')
    if scheduler.get('mode', 'min') != 'min':
        raise ValueError('The validation-loss scheduler requires mode: min')
    if not 0 < scheduler.get('factor', .1) < 1:
        raise ValueError('scheduler.factor must be between 0 and 1')
    if scheduler.get('threshold_mode', 'rel') not in ('rel', 'abs'):
        raise ValueError('scheduler.threshold_mode must be rel or abs')
    training_options = settings.get('training', {})
    _keys(training_options, TRAINING_DEFAULTS, 'training')
    training = {**TRAINING_DEFAULTS, **training_options, 'optimizer': optimizer_name,
                'optimizer_settings': optimizer, 'scheduler': {'enabled': enabled, 'settings': scheduler}}
    name = settings.get('experiment_name', model['head'] if path.stem == 'config' else path.stem)
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('experiment_name must contain only letters, numbers, underscores or hyphens')
    evaluation = settings.get('evaluation', {})
    _keys(evaluation, {'enable_test'}, 'evaluation')
    if not isinstance(evaluation.get('enable_test', False), bool):
        raise ValueError('evaluation.enable_test must be boolean')
    config = {
        'schema': 'autoqc_cnn_config_v1', 'experiment_id': name, 'status': 'ready',
        'dataset': {'path': str(dataset), 'sha256': sha256(dataset),
                    'target_kind': metadata['target_kind'], 'graders': metadata['graders'],
                    'image_directory': image_directory},
        'model': model, 'loss': loss, 'training': training,
        'preprocessing': preprocessing,
        'cohort': {'training_image_ids': frame.loc[frame.split.eq('train') & frame.eligible, 'image_id'].tolist(),
                   'monitor_image_ids': frame.loc[frame.split.eq('validation') & frame.eligible, 'image_id'].tolist()},
        'evaluation': {'image_ids': frame.loc[frame.split.eq('validation') & frame.eligible, 'image_id'].tolist(),
                       'subsets': {}, 'enable_test': evaluation.get('enable_test', False)},
        'selection': {'criterion': 'strict minimum primary validation loss'},
    }
    if root is not None:
        config['local_root'] = str(root)
        config['effective_paths'] = {**paths, 'dataset': str(dataset),
                                     'split': str(root_input(root, 'split')),
                                     'ganaav_labels': str(root_input(root, 'ganaav')),
                                     'ratings': str(root_input(root, 'ratings')),
                                     'dataset_manifest': str(dataset.parent / SOURCE_FILENAMES['manifest'])}
    if model['head'] == 'frozen_binary':
        source = model.get('source')
        if not source or not {'checkpoint', 'sha256', 'architecture'}.issubset(source):
            raise ValueError('frozen_binary needs model.source with checkpoint, sha256 and architecture')
        source = copy.deepcopy(source)
        source_path = Path(source['checkpoint'])
        if root is not None and not source_path.is_absolute():
            source_path = Path(paths['models']) / source_path
        source['checkpoint'] = str(require_file(source_path, 'frozen source checkpoint'))
        if sha256(source_path) != source['sha256'] or source['architecture'].get('head') != 'or_soft':
            raise ValueError('Frozen source hash or architecture mismatch')
        config['model']['source'] = source
    _apply_overrides(config, overrides)
    if root is not None:
        editable = copy.deepcopy(settings)
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if key == 'learning_rate':
                editable.setdefault('optimizer', {})['lr'] = value
            else:
                editable.setdefault('training', {})[key] = value
        if editable == settings and path.suffix.lower() in ('.yaml', '.yml'):
            config['input_config_text'] = path.read_text(encoding='utf-8')
        else:
            config['input_config_text'] = editable_yaml(editable)
    return config


def _apply_overrides(config, overrides):
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    _keys(overrides, {*TRAINING_DEFAULTS, 'learning_rate'}, 'command-line overrides')
    for key, value in overrides.items():
        if key == 'learning_rate':
            config['training']['optimizer_settings']['lr'] = value
        else:
            config['training'][key] = value
    training = config.get('training', {})
    if not training:
        return
    for key in ('epochs', 'batch_size', 'patience', 'seed'):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError(f'training.{key} must be an integer >= {0 if key == "seed" else 1}')
    if training['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('training.device must be auto, cpu or cuda')
    _positive(training['optimizer_settings']['lr'], 'learning rate')


def model_config_name(config):
    prefix = 'resnet18_' if config['model'].get('backbone') == 'resnet18' else ''
    if config['model'].get('backbone') == 'iqanet4':
        prefix = 'iqanet4_'
    return f"{prefix}{config['model']['head']}.yaml"


def checkpoint_settings(config):
    """Keep model/evaluation settings, without provenance or duplicated image lists."""
    settings = {key: config[key] for key in ('experiment_id', 'model', 'loss', 'training', 'preprocessing', 'selection')}
    for key in ('local_root', 'effective_paths'):
        if key in config:
            settings[key] = config[key]
    settings['dataset'] = {key: value for key, value in config['dataset'].items()
                           if key in ('target_kind', 'graders', 'profile', 'image_directory')}
    settings['evaluation'] = {key: config['evaluation'][key] for key in ('enable_test', 'subsets')}
    settings['cohort'] = {key: value for key, value in config['cohort'].items() if key == 'matched_sampling'}
    if config.get('historical', {}).get('run'):
        settings['historical'] = {'run': config['historical']['run']}
    return settings


def editable_yaml(settings):
    """Keep the same small sections as the editable input file."""
    import yaml
    validate_editable_output(settings)
    return '\n'.join(yaml.safe_dump({key: value}, sort_keys=False, allow_unicode=True)
                     for key, value in settings.items())


def validate_editable_output(settings):
    """Refuse internal metadata in the human-readable output file."""
    sections = {
        'model': {*MODEL_DEFAULTS, 'head', 'source', 'backbone', 'pretrained'},
        'data': {'dataset', 'target', 'graders', 'image_directory', 'input_mode'},
        'training': set(TRAINING_DEFAULTS),
        'optimizer': {'name', 'lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize',
                      'foreach', 'capturable', 'differentiable', 'fused', 'momentum', 'dampening', 'nesterov'},
        'scheduler': {'enabled', 'name', 'mode', 'factor', 'patience', 'threshold',
                      'threshold_mode', 'cooldown', 'min_lr', 'eps'},
        'loss': {'name', 'class_weights', 'binary_loss_weight', 'alpha'},
        'evaluation': {'enable_test'},
    }
    _keys(settings, {'local_root', 'experiment_name', *sections}, 'output config')
    for name, allowed in sections.items():
        if name in settings:
            _keys(settings[name], allowed, f'output config.{name}')
    if 'class_weights' in settings.get('loss', {}):
        _keys(settings['loss']['class_weights'], {'mode', 'values'}, 'output config.loss.class_weights')


def editable_settings(config):
    """Recover the input-shaped settings for older expanded run configs."""
    training = config['training']
    dataset = Path(config['dataset']['path'])
    settings = {
        'experiment_name': config['experiment_id'],
        'model': {key: config['model'][key] for key in ('head', 'dropout_p', 'batch_normalization')},
        'data': {'dataset': (dataset.relative_to(Path(config['local_root']) / 'datasets').as_posix()
                             if 'local_root' in config else str(dataset)),
                 'target': config['dataset']['target_kind'],
                 'input_mode': config.get('preprocessing', {}).get('input_mode', 'patches')},
        'training': {key: training[key] for key in TRAINING_DEFAULTS},
        'optimizer': {'name': training['optimizer'], **{key: value for key, value in
                      training['optimizer_settings'].items() if key in ('lr', 'weight_decay')}},
        'scheduler': {'enabled': training['scheduler']['enabled'], **{key: value for key, value in
                      training['scheduler']['settings'].items() if key in ('factor', 'patience', 'min_lr')}},
        'loss': {'name': config['loss']['name'], 'class_weights': {
            key: value for key, value in config['loss']['class_weights'].items() if key in ('mode', 'values')}},
        'evaluation': {'enable_test': config['evaluation']['enable_test']},
    }
    if 'local_root' in config:
        settings = {'local_root': config['local_root'], **settings}
    else:
        settings['data']['image_directory'] = config['dataset'].get('image_directory', 'images/')
    for key in ('binary_loss_weight', 'alpha'):
        if key in config['loss']:
            settings['loss'][key] = config['loss'][key]
    if config['model'].get('source'):
        settings['model']['source'] = config['model']['source']
    if config['model'].get('backbone') == 'resnet18':
        settings['model'].pop('batch_normalization', None)
        settings['model'].update(backbone='resnet18', pretrained=config['model'].get('pretrained', False))
    if config['model'].get('backbone') == 'iqanet4':
        settings['model'].update(backbone='iqanet4', pretrained=False)
    return settings


def load_run_config(run_dir, checkpoint=None):
    """Read checkpoint settings, with compatibility for older saved runs."""
    import copy
    import pandas as pd
    import torch
    from .utils import run_config_path
    run_dir = Path(run_dir)
    for resolved in [run_dir / 'config.yaml', *sorted(run_dir.glob('*.yaml'))]:
        if not resolved.is_file():
            continue
        config = read_config(resolved)
        if config.get('schema') == 'autoqc_cnn_config_v1':
            return config
    if checkpoint is None:
        checkpoint = torch.load(run_dir / 'checkpoint.pth', map_location='cpu', weights_only=True)
    if 'run_settings' not in checkpoint:
        return read_config(run_config_path(run_dir))
    config = copy.deepcopy(checkpoint['run_settings'])
    snapshot = (io.BytesIO(checkpoint['dataset_snapshot']) if 'dataset_snapshot' in checkpoint
                else run_dir / 'dataset.xlsx')
    frame = pd.read_excel(snapshot, sheet_name='dataset')
    for role, flag in [('training', 'training_eligible'), ('monitor', 'monitor_eligible')]:
        config['cohort'][role + '_image_ids'] = frame.loc[frame[flag], 'image_id'].tolist()
    config['evaluation']['image_ids'] = frame.loc[frame.evaluation_eligible & frame.split.eq('validation'), 'image_id'].tolist()
    return config


def write_model_config(directory, config):
    """Write only the validated, input-shaped config; never serialize internal settings."""
    import yaml
    content = config.get('input_config_text') or editable_yaml(editable_settings(config))
    validate_editable_output(yaml.safe_load(content))
    path = Path(directory) / model_config_name(config)
    temporary = path.with_suffix('.yaml.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)
    return path
