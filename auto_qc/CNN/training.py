"""One configured training entry point for original and later IQAnet methods."""
import argparse
import copy
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import tempfile
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ImagePatchDataset
from .configuration import checkpoint_settings, editable_yaml, load_training_config, model_config_name, preprocessing_settings, write_model_config
from data.labels import load_workbook, save_workbook, validate_records
from .losses import loss_components
from .model import make_model
from .testing import evaluate
from .utils import ROOT, class_weights, code_version, fingerprint, load_checkpoint, seed_everything, sha256, source_files, state_copy, write_json

PATHS = {'hard_ganaav': ('five_logits', 'hard_ce'), 'hard_shane': ('five_logits', 'hard_ce'),
         'ceiling_mean': ('five_logits', 'hard_ce'), 'exact_mean': ('five_logits', 'continuous_cdw_ce'),
         'soft_categorical': ('five_logits', 'soft_ce')}


def preflight(config):
    if config.get('status') != 'ready':
        raise ValueError(f'Configuration is blocked: {config.get("blocked_reasons")}')
    dataset = ROOT / config['dataset']['path']
    if sha256(dataset) != config['dataset']['sha256']:
        raise ValueError('Configured workbook hash mismatch')
    frame, metadata = load_workbook(dataset, check_files=False,
                                    target_kind=config['dataset']['target_kind'], graders=config['dataset']['graders'],
                                    **({'local_root': config['local_root']} if 'local_root' in config else {}))
    if config['dataset'].get('image_directory'):
        image_directory = ROOT / config['dataset']['image_directory']
        frame['image_path'] = frame.image_path.map(lambda path: str(image_directory / Path(path).name))
    kind = metadata['target_kind']
    if kind != config['dataset']['target_kind'] or metadata['graders'] != config['dataset']['graders']:
        raise ValueError('Dataset target kind / graders differ from configuration')
    if kind == 'ordinal':
        mapping = {'coral': 'ordinal_bce', 'or_soft': 'ordinal_bce', 'or_soft_binary': 'joint_bce', 'frozen_binary': 'binary_bce'}
        if mapping.get(config['model']['head']) != config['loss']['name']:
            raise ValueError('Ordinal target/head/loss mismatch')
        if config['loss'].get('threshold_weights') != [1, 1, 1, 1]:
            raise ValueError('Historical ordinal objectives use four unweighted thresholds')
    elif PATHS.get(kind) != (config['model']['head'], config['loss']['name']):
        raise ValueError('Target/head/loss mismatch (fractional means are not class indices)')
    if kind in ('ordinal', 'exact_mean') and config['loss']['class_weights']['mode'] != 'none':
        raise ValueError('Ordinal/CDW objectives do not accept class weighting')
    if config['loss']['name'] == 'joint_bce':
        value = config['loss']['binary_loss_weight']
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError('Binary loss weight must be finite and nonnegative')
    for role, split in [('training', 'train'), ('monitor', 'validation')]:
        ids = config['cohort'][role+'_image_ids']
        if len(ids) != len(set(ids)) or not set(ids).issubset(frame.loc[frame.split.eq(split), 'image_id']):
            raise ValueError(f'Invalid {role} IDs / split leakage')
        frame[role+'_eligible'] = frame.image_id.isin(ids) & frame.eligible
        if not frame[role+'_eligible'].any():
            raise ValueError(f'Empty {role} population')
    evaluation_ids = config['evaluation']['image_ids']
    if len(evaluation_ids) != len(set(evaluation_ids)) or not set(evaluation_ids).issubset(frame.loc[frame.split.eq('validation'), 'image_id']):
        raise ValueError('Evaluation IDs must be unique saved validation IDs')
    frame['evaluation_eligible'] = frame.image_id.isin(evaluation_ids) & frame.eligible
    # Test remains metadata only unless explicitly enabled by a separate invocation.
    frame.loc[frame.split.eq('test'), 'evaluation_eligible'] = bool(config['evaluation']['enable_test']) & frame.loc[frame.split.eq('test'), 'eligible']
    validate_records(frame, True)
    preprocessing = config['preprocessing']
    if preprocessing != preprocessing_settings(preprocessing.get('input_mode', 'patches')):
        raise ValueError('Unsupported preprocessing; refusing an implicit experimental change')
    return frame, metadata


def epoch_pass(model, loader, optimizer, loss, weights, device, reduction):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    count = 0
    weighted_numerator = weighted_denominator = 0.
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = loss_components(model(inputs), targets, loss, weights)
            if not all(torch.isfinite(value) for value in values):
                raise RuntimeError('Non-finite loss')
            if training:
                values[2].backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Non-finite gradients')
                optimizer.step()
            totals += np.array([v.detach().item() for v in values]) * len(inputs)
            count += len(inputs)
            if loss['name'] == 'hard_ce' and weights is not None:
                denominator = float(weights[targets].sum())
                weighted_numerator += float(values[0].detach()) * denominator
                weighted_denominator += denominator
    totals /= count
    if loss['name'] == 'hard_ce' and weights is not None and reduction == 'global_target_weight_sum':
        totals[0] = totals[2] = weighted_numerator / weighted_denominator
    elif reduction not in ('batch_size_weighted_batch_means', 'example_mean', 'global_target_weight_sum'):
        raise ValueError('Unknown epoch loss aggregation')
    return totals.tolist()


def history_figure(history, path, config=None, selected_epoch=None):
    """Plot recorded epoch losses without smoothing or changing their values."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    config = config or {}
    loss = config.get('loss', {}).get('name')
    titles = {'ordinal_bce': 'Ordinal head', 'joint_bce': 'Ordinal head',
              'binary_bce': 'Frozen-feature binary head', 'hard_ce': 'Classification head',
              'soft_ce': 'Soft-label classification head', 'continuous_cdw_ce': 'CDW-CE head'}
    panels = [('primary', '')]
    if loss == 'joint_bce':
        panels = [('primary', titles[loss]), ('binary', 'Binary head'), ('total', 'Total loss')]
    figure = Figure(figsize=(15.5 if len(panels) == 3 else 7.2, 4.5), dpi=180, layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, len(panels), squeeze=False)[0]
    epochs = history['epoch'].to_numpy()
    for axis, (component, title) in zip(axes, panels):
        for prefix, label, colour in [('train', 'Training loss', '#2563EB'), ('val', 'Validation loss', '#EA580C')]:
            axis.plot(epochs, history[f'{prefix}_{component}_loss'], label=label,
                      color=colour, linewidth=1.9, marker='o' if len(history) == 1 else None)
        if selected_epoch is not None:
            axis.axvline(selected_epoch, color='#64748B', linestyle='--', linewidth=1.2,
                         label=f'Selected checkpoint: epoch {selected_epoch}')
        axis.set(title=title, xlabel='Epoch', ylabel='Loss')
        axis.set_xlim((epochs[0], epochs[-1]) if len(epochs) > 1 else (epochs[0]-.5, epochs[0]+.5))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(axis='y', color='#E2E8F0', linewidth=.7)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.legend(frameon=False, fontsize=8, loc='best')
    name = config.get('experiment_id', config.get('model', {}).get('head', 'IQAnet'))
    figure.suptitle(f'Train/Validation loss for model {name}', fontsize=14, fontweight='bold')
    path = Path(path)
    temporary = path.with_suffix('.plot.tmp')
    figure.savefig(temporary, format='png', facecolor='white')
    temporary.replace(path)
    figure.clear()


def train(config_path, output_root=None, smoke=False, check_only=False, overrides=None):
    config = copy.deepcopy(load_training_config(config_path, overrides))
    if 'local_root' in config:
        models = Path(config['effective_paths']['models'])
        if output_root is not None:
            models = Path(output_root).expanduser()
            if not models.is_absolute():
                models = Path(config_path).resolve().parent / models
            models = models.resolve()
            config['effective_paths']['models'] = str(models)
    else:
        models = ROOT / (output_root if output_root is not None else Path('data/models'))
    frame, metadata = preflight(config)
    if check_only:
        print(f'CHECK PASS: {config["experiment_id"]}; train={int(frame.training_eligible.sum())}, monitor={int(frame.monitor_eligible.sum())}, evaluation={int(frame.evaluation_eligible.sum())}')
        return None
    if config.get('diagnostic_only') and not smoke:
        raise ValueError('Diagnostic configurations may run only with --smoke')
    if smoke:
        # A declared diagnostic cohort, never labelled a scientific rerun.
        for flag in ('training_eligible', 'monitor_eligible', 'evaluation_eligible'):
            ids = frame.loc[frame[flag], 'image_id'].iloc[:2]
            frame[flag] = frame.image_id.isin(ids)
        config['training']['epochs'] = 1
        if config.get('input_config_text'):
            import yaml
            editable = yaml.safe_load(config['input_config_text'])
            if editable.get('training', {}).get('epochs') != 1:
                editable.setdefault('training', {})['epochs'] = 1
                config['input_config_text'] = editable_yaml(editable)
        config['diagnostic_smoke'] = {'real_images_per_split': 2, 'original_config': str(config_path), 'not_a_retrospective_result': True}
        for key, flag in [('training', 'training_eligible'), ('monitor', 'monitor_eligible')]:
            config['cohort'][key+'_image_ids'] = frame.loc[frame[flag], 'image_id'].tolist()
        config['evaluation']['image_ids'] = frame.loc[frame.evaluation_eligible, 'image_id'].tolist()
    name = f'{config["experiment_id"]}_{"smoke_" if smoke else ""}{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}'
    run = models / name
    run.mkdir(parents=True, exist_ok=False)
    compact = 'local_root' in config
    scratch = tempfile.TemporaryDirectory(prefix='autoqc-') if compact else None
    support = Path(scratch.name) if scratch else run
    snapshot = run / 'dataset.xlsx'
    if 'local_root' in config:
        config['effective_paths'].update(run_dir=str(run),
                                         snapshot=str(snapshot),
                                         checkpoint=str(run / 'checkpoint.pth'), config=str(run / model_config_name(config)),
                                         training_history=str(run / 'training_history.csv'),
                                         learning_curve=str(run / 'training_history.png'),
                                         validation_predictions=str(run / 'validation_predictions.csv'),
                                         validation_metrics=str(run / 'validation_metrics.json'))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handlers = [logging.StreamHandler()]
    if not compact:
        handlers.insert(0, logging.FileHandler(run / 'training.log', encoding='utf-8'))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    status = {'status': 'preparing', 'experiment_id': config['experiment_id'], 'run_dir': str(run), 'smoke': smoke}
    write_json(support / 'run_status.json', status)
    try:
        config['code_version'] = code_version()
        for source in ([] if compact else source_files()):
            destination = support / 'source_code' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        config['dataset']['source_workbook_sha256'] = sha256(ROOT / config['dataset']['path'])
        save_workbook(snapshot, frame, metadata)
        # Ensure the exact saved records are the records training consumes.
        frame, metadata = load_workbook(snapshot, check_files=True)
        config['dataset']['run_snapshot_sha256'] = sha256(snapshot)
        write_model_config(run, config)
        seed = config['training']['seed']
        seed_everything(seed)
        device = config['training']['device']
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ordered = frame.set_index('image_id', drop=False)
        train_frame = ordered.loc[config['cohort']['training_image_ids']].loc[lambda f: f.training_eligible].reset_index(drop=True)
        validation = ordered.loc[config['cohort']['monitor_image_ids']].loc[lambda f: f.monitor_eligible].reset_index(drop=True)
        weights = class_weights(train_frame, config['loss']['class_weights'])
        if weights is not None:
            weights = weights.to(device)
        input_mode = config['preprocessing'].get('input_mode', 'patches')
        loaders = [DataLoader(ImagePatchDataset(part, metadata['target_kind'], augment=flip, input_mode=input_mode),
                             batch_size=config['training']['batch_size'], shuffle=flip, num_workers=0,
                             generator=torch.Generator().manual_seed(seed+offset))
                   for part, flip, offset in [(train_frame, True, 0), (validation, False, 1)]]
        seed_everything(seed)
        model = make_model(config['model'], seed).to(device)
        initial = fingerprint(model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        options = dict(config['training']['optimizer_settings'])
        optimizer_name = config['training']['optimizer']
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(parameters, **options)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(parameters, **options)
        else:
            raise ValueError('Explicit supported optimiser required')
        scheduler_settings = config['training']['scheduler']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_settings['settings']) if scheduler_settings['enabled'] else None
        config['effective_runtime'] = {'device': device, 'optimizer': optimizer.defaults, 'initial_model_sha256': initial,
                                       'class_weights': None if weights is None else weights.cpu().tolist(),
                                       'input_mode': input_mode,
                                       'batch_size_unit': 'images' if input_mode == 'whole_image' else 'patches',
                                       'train_patches': len(loaders[0].dataset),
                                       'monitor_patches': len(loaders[1].dataset), 'workers': 0,
                                       'patch_cache': 'precomputed once; lossless uint8 in memory; flips on access',
                                       'patch_cache_bytes': sum(loader.dataset.cache_nbytes for loader in loaders),
                                       'training_loader_seed': seed, 'validation_loader_seed': seed+1,
                                       'resize_interpolation': 'bilinear', 'resize_antialias': True,
                                       'normalization': 'ToTensor uint8 scaling only; no mean/std normalization',
                                       'scheduler': None if scheduler is None else {name: getattr(scheduler, name) for name in
                                           ('mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lrs', 'eps')}}
        if config['model'].get('backbone') == 'resnet18':
            config['effective_runtime']['normalization'] = 'grayscale repeated to RGB; ImageNet mean/std inside model'
        logger.info('Input mode=%s device=%s train_images=%d validation_images=%d train_inputs=%d validation_inputs=%d',
                    input_mode, device, len(train_frame), len(validation),
                    len(loaders[0].dataset), len(loaders[1].dataset))
        write_model_config(run, config)
        frozen_initial = fingerprint(model.source) if hasattr(model, 'source') else None
        status.update(status='running', initial_model_sha256=initial)
        write_json(support / 'run_status.json', status)
        best, best_epoch, stale, best_state = float('inf'), None, 0, None
        history = []
        for epoch in range(1, config['training']['epochs']+1):
            train_values = epoch_pass(model, loaders[0], optimizer, config['loss'], weights, device, config['loss']['epoch_aggregation'])
            val_values = epoch_pass(model, loaders[1], None, config['loss'], weights, device, config['loss']['epoch_aggregation'])
            row = dict(epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'])
            row.update({f'{prefix}_{component}_loss': v for prefix, values in [('train', train_values), ('val', val_values)]
                        for component, v in zip(('primary', 'binary', 'total'), values)})
            history.append(row)
            pd.DataFrame(history).to_csv(run / 'training_history.csv', index=False)
            selected = val_values[0]
            if selected < best:
                best, best_epoch, stale, best_state = selected, epoch, 0, state_copy(model)
                torch.save({'schema': 'autoqc_cnn_v1', 'architecture': config['model'], 'model_state': best_state,
                            'run_settings': checkpoint_settings(config),
                            'selected_epoch': epoch, 'selection_loss': selected, 'selection_rule': config['selection'],
                            'dataset_sha256': sha256(snapshot),
                            **({'dataset_snapshot': snapshot.read_bytes()} if compact else {})}, run / 'checkpoint.pth')
            else:
                stale += 1
            if frozen_initial and (fingerprint(model.source) != frozen_initial or any(m.training for m in model.source.modules())):
                raise RuntimeError('Frozen source changed or left evaluation mode')
            if scheduler:
                scheduler.step(selected)
            status.update(epoch=epoch, selected_epoch=best_epoch, best_selection_loss=best)
            write_json(support / 'run_status.json', status)
            logger.info('epoch=%d train=%.6f val=%.6f best_epoch=%s', epoch, train_values[0], selected, best_epoch)
            if stale >= config['training']['patience']:
                break
        model.load_state_dict(best_state, strict=True)
        reloaded, _, _ = load_checkpoint(run, device)
        if fingerprint(model) != fingerprint(reloaded):
            raise RuntimeError('Saved best checkpoint reload differs')
        model.eval()
        inputs = next(iter(loaders[1]))[0].to(device)
        with torch.inference_mode():
            original, restored = model(inputs), reloaded(inputs)
            original = original if isinstance(original, tuple) else (original,)
            restored = restored if isinstance(restored, tuple) else (restored,)
            if not all(torch.equal(a, b) for a, b in zip(original, restored)):
                raise RuntimeError('Fixed-input checkpoint predictions differ')
        metrics, predictions = evaluate(reloaded, frame, config, run, device=device, **({'compact': True} if compact else {}))
        if len(predictions) != int((frame.evaluation_eligible & frame.split.eq('validation')).sum()):
            raise RuntimeError('Evaluation population changed')
        if config['evaluation']['enable_test']:
            evaluate(reloaded, frame, config, support if compact else run, split='test', device=device)
        history_figure(pd.DataFrame(history), run / 'training_history.png', config, best_epoch)
        config['effective_runtime'].update(epochs_completed=epoch, selected_epoch=best_epoch)
        write_model_config(run, config)
        required = ['checkpoint.pth', 'dataset.xlsx', model_config_name(config), 'training_history.csv', 'training.log',
                    'validation_predictions.csv', 'validation_metrics.json', 'run_status.json']
        if compact:
            required = ['checkpoint.pth', model_config_name(config), 'training_history.png',
                        'dataset.xlsx', 'training_history.csv',
                        'validation_predictions.csv', 'validation_metrics.json']
        if any(not (run / name).is_file() for name in required):
            raise RuntimeError('Missing required output')
        status.update(status='completed', required_outputs_verified=True, validation_images=len(predictions),
                      final_model_sha256=fingerprint(reloaded), test_evaluated=config['evaluation']['enable_test'],
                      checkpoint_output_equality_verified=True)
        write_json(support / 'run_status.json', status)
        if compact:
            from .experiment_log import run_row, update_log
            try:
                log_path, _ = update_log(config['local_root'], [run_row(run, config)])
                logger.info('Experiment log updated: %s', log_path)
            except (OSError, ValueError, KeyError):
                logger.exception('Model saved, but experiment log update failed; close the workbook and run '
                                 'python -m auto_qc.CNN.experiment_log --local-root "%s"', config['local_root'])
        logger.info('COMPLETED %s', run)
        return run
    except BaseException as error:
        status.update(status='failed', error=str(error), traceback=traceback.format_exc())
        write_json(support / 'run_status.json', status)
        logger.exception('FAILED; diagnostic artifacts preserved')
        raise
    finally:
        for handler in handlers:
            handler.close()
        if scratch:
            scratch.cleanup()
            logger.removeHandler(handler)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Editable YAML settings or an internal run configuration')
    parser.add_argument('--output-root', type=Path, default=None, help='Optional override of config models location')
    for option in ('epochs', 'batch-size', 'patience', 'seed'):
        parser.add_argument('--' + option, type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default=None)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One epoch; two real training and two validation images; explicitly diagnostic')
    return parser


def main():
    args = argument_parser().parse_args()
    overrides = {key: getattr(args, key) for key in ('epochs', 'batch_size', 'patience', 'seed', 'learning_rate', 'device')}
    run = train(args.config, args.output_root, args.smoke, args.check_only, overrides)
    if run is not None and not args.smoke:
        from .configuration import load_run_config
        if 'local_root' not in load_run_config(run):
            from .collect_results import collect_run
            result = collect_run(run)
            print(f'Results: {result}')


if __name__ == '__main__':
    main()
