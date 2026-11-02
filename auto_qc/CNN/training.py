"""One configured training entry point for original and later IQAnet methods."""
import argparse
import copy
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import traceback
import uuid

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ImageQualityDataset, load_workbook, save_workbook, validate_records
from .losses import loss_components
from .model import make_model
from .testing import evaluate
from .utils import ROOT, class_weights, code_version, fingerprint, load_checkpoint, read_config, seed_everything, sha256, state_copy, write_json

PATHS = {'hard_ganaav': ('five_logits', 'hard_ce'), 'hard_shane': ('five_logits', 'hard_ce'),
         'ceiling_mean': ('five_logits', 'hard_ce'), 'exact_mean': ('five_logits', 'continuous_cdw_ce'),
         'soft_categorical': ('five_logits', 'soft_ce')}


def preflight(config):
    if config.get('status') != 'ready':
        raise ValueError(f'Configuration is blocked: {config.get("blocked_reasons")}')
    dataset = ROOT / config['dataset']['path']
    if sha256(dataset) != config['dataset']['sha256']:
        raise ValueError('Configured workbook hash mismatch')
    frame, metadata = load_workbook(dataset, check_files=True)
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
    if preprocessing != {'resize': [768, 768], 'grayscale_channels': 1, 'patch_size': 256, 'grid_size': 3, 'method': 'all', 'horizontal_flip': True}:
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


def history_figure(history, path):
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (640, 360), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((20, 10), 'Train (blue) and validation (red) selection losses', fill='black')
    maximum = max(1e-8, float(history[['train_primary_loss', 'val_primary_loss']].to_numpy().max()))
    for name, colour in [('train_primary_loss', 'blue'), ('val_primary_loss', 'red')]:
        points = [(40+i*560/max(1, len(history)-1), 320-280*v/maximum) for i, v in enumerate(history[name])]
        if len(points) > 1:
            draw.line(points, fill=colour, width=2)
        else:
            x, y = points[0]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=colour)
    image.save(path)


def train(config_path, output_root=Path('model_dir'), smoke=False, check_only=False):
    config = copy.deepcopy(read_config(config_path))
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
        config['diagnostic_smoke'] = {'real_images_per_split': 2, 'original_config': str(config_path), 'not_a_retrospective_result': True}
        for key, flag in [('training', 'training_eligible'), ('monitor', 'monitor_eligible')]:
            config['cohort'][key+'_image_ids'] = frame.loc[frame[flag], 'image_id'].tolist()
        config['evaluation']['image_ids'] = frame.loc[frame.evaluation_eligible, 'image_id'].tolist()
    name = f'{config["experiment_id"]}_{"smoke_" if smoke else ""}{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:8]}'
    run = ROOT / output_root / name
    run.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(run / 'training.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    status = {'status': 'preparing', 'experiment_id': config['experiment_id'], 'run_dir': str(run), 'smoke': smoke}
    write_json(run / 'run_status.json', status)
    try:
        config['code_version'] = code_version()
        for source in sorted((ROOT / 'auto_qc').rglob('*.py')):
            destination = run / 'source_code' / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        config['dataset']['source_workbook_sha256'] = sha256(ROOT / config['dataset']['path'])
        save_workbook(run / 'dataset.xlsx', frame, metadata)
        # Ensure the exact saved records are the records training consumes.
        frame, metadata = load_workbook(run / 'dataset.xlsx', check_files=True)
        config['dataset']['run_snapshot_sha256'] = sha256(run / 'dataset.xlsx')
        write_json(run / 'config.yaml', config)
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
        loaders = [DataLoader(ImageQualityDataset(part, metadata['target_kind'], augment=flip),
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
                                       'batch_size_unit': 'patches', 'train_patches': len(loaders[0].dataset),
                                       'monitor_patches': len(loaders[1].dataset), 'workers': 0,
                                       'training_loader_seed': seed, 'validation_loader_seed': seed+1,
                                       'resize_interpolation': 'bilinear', 'resize_antialias': True,
                                       'normalization': 'ToTensor uint8 scaling only; no mean/std normalization',
                                       'scheduler': None if scheduler is None else {name: getattr(scheduler, name) for name in
                                           ('mode', 'factor', 'patience', 'threshold', 'threshold_mode', 'cooldown', 'min_lrs', 'eps')}}
        write_json(run / 'config.yaml', config)
        frozen_initial = fingerprint(model.source) if hasattr(model, 'source') else None
        status.update(status='running', initial_model_sha256=initial)
        write_json(run / 'run_status.json', status)
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
                            'selected_epoch': epoch, 'selection_loss': selected, 'selection_rule': config['selection'],
                            'dataset_sha256': sha256(run / 'dataset.xlsx')}, run / 'checkpoint.pth')
            else:
                stale += 1
            if frozen_initial and (fingerprint(model.source) != frozen_initial or any(m.training for m in model.source.modules())):
                raise RuntimeError('Frozen source changed or left evaluation mode')
            if scheduler:
                scheduler.step(selected)
            status.update(epoch=epoch, selected_epoch=best_epoch, best_selection_loss=best)
            write_json(run / 'run_status.json', status)
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
        metrics, predictions = evaluate(reloaded, frame, config, run, device=device)
        if len(predictions) != int((frame.evaluation_eligible & frame.split.eq('validation')).sum()):
            raise RuntimeError('Evaluation population changed')
        if config['evaluation']['enable_test']:
            evaluate(reloaded, frame, config, run, split='test', device=device)
        history_figure(pd.DataFrame(history), run / 'training_history.png')
        required = ['checkpoint.pth', 'dataset.xlsx', 'config.yaml', 'training_history.csv', 'training.log',
                    'validation_predictions.csv', 'validation_metrics.json', 'run_status.json']
        if any(not (run / name).is_file() for name in required):
            raise RuntimeError('Missing required output')
        status.update(status='completed', required_outputs_verified=True, validation_images=len(predictions),
                      final_model_sha256=fingerprint(reloaded), test_evaluated=config['evaluation']['enable_test'],
                      checkpoint_output_equality_verified=True)
        write_json(run / 'run_status.json', status)
        logger.info('COMPLETED %s', run)
        return run
    except BaseException as error:
        status.update(status='failed', error=str(error), traceback=traceback.format_exc())
        write_json(run / 'run_status.json', status)
        logger.exception('FAILED; diagnostic artifacts preserved')
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='JSON-formatted YAML 1.2 configuration')
    parser.add_argument('--output-root', type=Path, default=Path('model_dir'))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One epoch; two real training and two validation images; explicitly diagnostic')
    args = parser.parse_args()
    train(args.config, args.output_root, args.smoke, args.check_only)


if __name__ == '__main__':
    main()
