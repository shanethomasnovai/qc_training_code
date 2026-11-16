"""Check editable YAML settings and compatibility with saved run configs."""
import tempfile
import unittest
import os
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
import numpy as np
from PIL import Image

from auto_qc.CNN.configuration import load_training_config, load_run_config, write_model_config
from auto_qc.CNN.utils import ROOT, write_json, sha256
from data.tools import SOURCE_FILENAMES, LOCAL_FOLDERS
from data.tools.generate_datasets import VARIANTS, RULES
from data.labels import load_workbook


class EditableConfig(unittest.TestCase):
    def expand(self, settings):
        records = pd.DataFrame({'image_id': ['train_1', 'val_1', 'test_1'],
                                'split': ['train', 'validation', 'test'],
                                'eligible': [True, True, True]})
        metadata = {'target_kind': 'ordinal', 'graders': ['ganaav_grade', 'shane_grade']}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(settings), encoding='utf-8')
            with patch('auto_qc.CNN.configuration.load_workbook', return_value=(records, metadata)), \
                    patch('auto_qc.CNN.configuration.sha256', return_value='dataset_hash'):
                return load_training_config(path)

    def settings(self):
        settings = yaml.safe_load((ROOT / 'data/samples/config.yaml').read_text(encoding='utf-8'))
        settings.pop('local_root')
        settings.pop('experiment_name')
        # This fixture tests single-head settings independently of the sample's active run.
        settings['model'].update(backbone='iqanet', pretrained=False, head='or_soft')
        settings['loss']['name'] = 'ordinal_bce'
        settings['loss'].pop('binary_loss_weight', None)
        settings['data']['input_mode'] = 'patches'
        settings['data']['image_directory'] = 'legacy_images'
        return settings

    def test_example_expands_to_two_heads_and_saved_splits(self):
        settings = self.settings()
        settings['model']['head'] = 'or_soft_binary'
        settings['loss']['name'] = 'joint_bce'
        config = self.expand(settings)
        self.assertEqual(config['experiment_id'], 'or_soft_binary')
        self.assertEqual(config['loss']['name'], 'joint_bce')
        self.assertEqual(config['cohort']['training_image_ids'], ['train_1'])
        self.assertEqual(config['evaluation']['image_ids'], ['val_1'])
        self.assertFalse(config['evaluation']['enable_test'])

    def test_edits_change_head_learning_rate_and_image_directory(self):
        settings = self.settings()
        settings['model']['head'] = 'or_soft'
        settings['optimizer']['lr'] = .002
        settings['data']['image_directory'] = 'another/image_folder'
        config = self.expand(settings)
        self.assertEqual(config['loss']['name'], 'ordinal_bce')
        self.assertEqual(config['training']['optimizer_settings']['lr'], .002)
        self.assertEqual(config['dataset']['image_directory'], 'another/image_folder')

    def test_unknown_section_is_rejected(self):
        settings = self.settings()
        settings['trainnig'] = {}
        with self.assertRaisesRegex(ValueError, 'Unknown config sections'):
            self.expand(settings)

    def test_whole_image_preprocessing_and_invalid_mode(self):
        settings = self.settings()
        settings['data']['input_mode'] = 'whole_image'
        config = self.expand(settings)
        self.assertEqual(config['preprocessing']['resize'], [256, 256])
        self.assertEqual(config['preprocessing']['grid_size'], 1)
        self.assertEqual(config['preprocessing']['input_mode'], 'whole_image')
        settings['data']['input_mode'] = 'typo'
        with self.assertRaisesRegex(ValueError, 'input_mode'):
            self.expand(settings)

    def test_saved_internal_config_is_loaded_unchanged(self):
        config = {'schema': 'autoqc_cnn_config_v1', 'model': {'head': 'or_soft'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            write_json(path, config)
            self.assertEqual(load_training_config(path), config)


class LocalRootConfig(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'local data with spaces'
        self.config = self.base / 'settings' / 'config.yaml'
        self.config.parent.mkdir()
        self.variant = self.root / 'datasets/reviewed_v2'
        self.variant.mkdir(parents=True)
        (self.root / 'images').mkdir()
        ids = ['DARC-P2_SITE_001_OD_V1_2020_1.1_0',
               'DARC-P2_SITE_002_OD_V1_2020_1.0_0',
               'VISUS-NHP_SITE_003_OD_D07_2020_5.1_0']
        self.records = pd.DataFrame({
            'image_id': ids, 'image_path': ['Z:/old_machine/images/' + value + '.png' for value in ids],
            'patient_id': ['_'.join(value.split('_')[:3]) for value in ids],
            'project': ['DARC-P2', 'DARC-P2', 'VISUS-NHP'], 'project_site': ['SITE'] * 3,
            'split': ['train', 'validation', 'test']})
        split = self.root / 'datasets' / SOURCE_FILENAMES['split']
        self.records.to_csv(split, index=False)
        for name in ids:
            Image.new('L', (16, 16), 100).save(self.root / 'images' / (name + '.png'))
        self.graders = ['ganaav_grade', 'shane_grade', 'reviewer1_grade', 'reviewer2_grade', 'reviewer3_grade']
        labels = pd.DataFrame({'image_filename': [value + '.png' for value in ids]})
        for grader in self.graders:
            labels[grader] = [2, 3, 4]
        self.variants = {}
        for name, (target, graders) in VARIANTS.items():
            table = labels.copy() if name == 'multigrader_labels.xlsx' else labels[['image_filename']].assign(label=[2, 3, 4])
            table.to_excel(self.variant / name, index=False, sheet_name='labels')
            self.variants[name] = {'target_kind': target, 'graders': graders or self.graders,
                                   'mean_missing_policy': 'available' if graders is None else 'require_all',
                                   'label_rule': RULES[target], 'sha256': sha256(self.variant / name)}
        self.manifest = {'schema': 'autoqc_labels_v1', 'graders': self.graders,
                         'ratings_file': 'multigrader_labels.xlsx', 'held_out_project': 'VISUS-NHP',
                         'rating_profile': 'reviewed_v2', 'variants': self.variants,
                         'sources': {'split': {'path': 'Z:/historical/split.csv', 'sha256': sha256(split)}}}
        write_json(self.variant / 'manifest.json', self.manifest)
        self.settings = yaml.safe_load((ROOT / 'data/samples/config.yaml').read_text(encoding='utf-8'))
        self.settings['experiment_name'] = 'or_soft'
        self.settings['model'].update(backbone='iqanet', pretrained=False, head='or_soft')
        self.settings['loss']['name'] = 'ordinal_bce'
        self.settings['loss'].pop('binary_loss_weight', None)
        self.settings['data']['input_mode'] = 'patches'
        self.settings['local_root'] = '../local data with spaces'
        self.save()

    def save(self):
        self.config.write_text(yaml.safe_dump(self.settings, sort_keys=False), encoding='utf-8')

    def test_relative_root_is_config_relative_from_another_working_directory(self):
        original = self.config.read_bytes()
        old = Path.cwd()
        try:
            os.chdir(self.root / 'images')
            config = load_training_config(self.config)
        finally:
            os.chdir(old)
        self.assertEqual(config['local_root'], str(self.root.resolve()))
        for name, folder in LOCAL_FOLDERS.items():
            self.assertEqual(config['effective_paths'][name], str(self.root / folder))
        self.assertEqual(config['dataset']['image_directory'], str(self.root / 'images'))
        self.assertEqual(self.config.read_bytes(), original)
        self.assertFalse((self.root / 'models').exists())

    def test_sample_uses_supported_choices_and_local_root(self):
        from auto_qc.CNN.configuration import HEAD_LOSSES, TARGET_LOSSES
        from data.labels import KINDS
        text = (ROOT / 'data/samples/config.yaml').read_text(encoding='utf-8')
        sample = yaml.safe_load(text)
        self.assertIsInstance(sample['local_root'], str)
        self.assertTrue(sample['local_root'].strip())
        self.assertNotIn('image_directory', sample['data'])
        self.assertIn(sample['model']['head'], {*HEAD_LOSSES, 'five_logits'})
        self.assertIn(sample['data']['target'], KINDS)
        self.assertIn(sample['loss']['name'], {*HEAD_LOSSES.values(), *TARGET_LOSSES.values()})
        self.assertIn(sample['optimizer']['name'], ('AdamW', 'SGD'))
        self.assertIn(sample['training']['device'], ('auto', 'cpu', 'cuda'))

    def test_check_only_entry_point_from_another_directory(self):
        import subprocess
        import sys
        environment = os.environ.copy()
        # Make the existing module importable, as when launched in an installed
        # environment; local input/output resolution must not depend on cwd.
        environment['PYTHONPATH'] = str(ROOT)
        original = self.config.read_bytes()
        result = subprocess.run([sys.executable, '-m', 'auto_qc.CNN.training', '--config', str(self.config), '--check-only'],
                                cwd=self.root, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CHECK PASS', result.stdout)
        self.assertEqual(self.config.read_bytes(), original)
        self.assertFalse((self.root / 'models').exists())

    def test_historical_split_paths_are_rebased_without_rewriting_metadata(self):
        original = (self.variant / 'manifest.json').read_bytes()
        frame, _ = load_workbook(self.variant / 'multigrader_labels.xlsx', local_root=self.root, check_files=True)
        self.assertTrue(all(Path(path).parent == self.root / 'images' for path in frame.image_path))
        self.assertEqual((self.variant / 'manifest.json').read_bytes(), original)
        split = self.root / 'datasets/manifest.csv'
        split.write_text(split.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'split hash mismatch'):
            load_training_config(self.config)

    def test_snapshot_does_not_silently_ignore_explicit_target_or_graders(self):
        from data.labels import save_workbook
        frame, metadata = load_workbook(self.variant / 'multigrader_labels.xlsx', local_root=self.root)
        snapshot = self.root / 'datasets/snapshot.xlsx'
        save_workbook(snapshot, frame, metadata)
        with self.assertRaisesRegex(ValueError, 'target differs'):
            load_workbook(snapshot, target_kind='soft_categorical', local_root=self.root)
        with self.assertRaisesRegex(ValueError, 'graders differ'):
            load_workbook(snapshot, graders=['ganaav_grade'], local_root=self.root)

    def test_missing_dataset_split_and_images_report_expected_locations(self):
        self.settings['data']['dataset'] = 'reviewed_v2/missing.xlsx'
        self.save()
        with self.assertRaises(FileNotFoundError) as error:
            load_training_config(self.config)
        self.assertIn(str(self.variant / 'missing.xlsx'), str(error.exception))
        self.settings['data']['dataset'] = 'reviewed_v2/multigrader_labels.xlsx'
        self.save()
        (self.root / 'datasets/manifest.csv').unlink()
        with self.assertRaises(FileNotFoundError) as error:
            load_training_config(self.config)
        self.assertIn(str(self.root / 'datasets/manifest.csv'), str(error.exception))

    def test_missing_image_is_reported_by_preflight(self):
        from auto_qc.CNN.training import preflight
        config = load_training_config(self.config)
        image = self.root / 'images' / (self.records.image_id.iloc[0] + '.png')
        image.unlink()
        with self.assertRaises(FileNotFoundError) as error:
            preflight(config)
        self.assertIn(str(image), str(error.exception))

    def test_legacy_path_conflicts_and_dataset_escape_are_errors(self):
        self.settings['data']['image_directory'] = 'other_images'
        self.save()
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            load_training_config(self.config)
        del self.settings['data']['image_directory']
        for value in (str(self.variant / 'multigrader_labels.xlsx'), '../outside.xlsx'):
            self.settings['data']['dataset'] = value
            self.save()
            with self.assertRaisesRegex(ValueError, 'relative'):
                load_training_config(self.config)

    def test_cli_overrides_yaml_then_defaults_without_argparse_defaults_winning(self):
        from auto_qc.CNN.training import argument_parser
        self.settings['training']['epochs'] = 100
        self.settings['optimizer']['lr'] = .0001
        self.save()
        args = argument_parser().parse_args(['--config', str(self.config)])
        overrides = {key: getattr(args, key) for key in ('epochs', 'learning_rate', 'seed', 'device')}
        baseline = load_training_config(self.config, overrides)
        self.assertEqual(baseline['training']['epochs'], 100)
        self.assertEqual(baseline['training']['optimizer_settings']['lr'], .0001)
        config = load_training_config(self.config, {'epochs': 7, 'learning_rate': .002, 'seed': 9})
        self.assertEqual(config['training']['epochs'], 7)
        self.assertEqual(config['training']['seed'], 9)
        self.assertEqual(config['training']['optimizer_settings']['lr'], .002)
        del self.settings['training']['batch_size']
        self.save()
        self.assertEqual(load_training_config(self.config)['training']['batch_size'], 16)

    def test_incompatible_head_loss_and_weights_are_rejected(self):
        for section, key, value in [('model', 'head', 'five_logits'),
                                    ('loss', 'name', 'hard_ce'),
                                    ('loss', 'class_weights', {'mode': 'explicit', 'values': [1] * 5})]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.settings)
                self.settings[section][key] = value
                self.save()
                with self.assertRaises(ValueError):
                    load_training_config(self.config)
                self.settings = original

    def test_output_config_preserves_input_text_and_comments(self):
        self.config.write_text('# My readable config\n' + self.config.read_text(encoding='utf-8'), encoding='utf-8')
        config = load_training_config(self.config)
        config['effective_runtime'] = {'device': 'cpu', 'class_weights': [1, 2, 3, 4, 5]}
        output = self.base / 'synthetic experiment'
        output.mkdir()
        saved = write_model_config(output, config)
        self.assertEqual(saved.name, 'or_soft.yaml')
        self.assertEqual(saved.read_text(encoding='utf-8'), self.config.read_text(encoding='utf-8'))
        self.assertEqual(read_config_yaml(saved), self.settings)

    def test_output_writer_rejects_expanded_metadata(self):
        config = load_training_config(self.config)
        output = self.base / 'rejected output'
        output.mkdir()
        for extra in ({'cohort': {'training_image_ids': ['internal']}},
                      {'training': {'epochs': 1, 'optimizer_settings': {'lr': .001}}}):
            with self.subTest(extra=extra):
                config['input_config_text'] = yaml.safe_dump({**self.settings, **extra})
                with self.assertRaises(ValueError):
                    write_model_config(output, config)
                self.assertEqual(list(output.iterdir()), [])

    def test_fallback_writer_only_exports_input_sections_for_all_configs(self):
        config = load_training_config(self.config)
        del config['input_config_text']
        for local in (True, False):
            with self.subTest(local=local):
                if not local:
                    del config['local_root']
                output = self.base / f'fallback-{local}'
                output.mkdir()
                saved = read_config_yaml(write_model_config(output, config))
                self.assertEqual(saved['training']['epochs'], config['training']['epochs'])
                self.assertNotIn('schema', saved)
                self.assertNotIn('cohort', saved)
                self.assertNotIn('preprocessing', saved)
                self.assertNotIn('optimizer_settings', saved['training'])

    def test_synthetic_pipeline_uses_root_outputs_and_saves_effective_defaults(self):
        import torch
        from auto_qc.CNN.training import train
        from auto_qc.CNN.utils import load_checkpoint
        original = self.config.read_bytes()

        def tiny_model(*args, **kwargs):
            return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(1, 4))

        with patch('auto_qc.CNN.training.code_version', return_value={}), \
                patch('auto_qc.CNN.training.source_files', return_value=[]), \
                patch('auto_qc.CNN.training.make_model', side_effect=tiny_model), \
                patch('auto_qc.CNN.model.make_model', side_effect=tiny_model), \
                patch('auto_qc.CNN.training.epoch_pass', return_value=[1., 0., 1.]):
            run = train(self.config, overrides={'epochs': 1})
            self.assertEqual(run.parent, self.root / 'models')
            saved = read_config_yaml(run / 'or_soft.yaml')
            expected = copy.deepcopy(self.settings)
            expected['training']['epochs'] = 1
            self.assertEqual(saved, expected)
            self.assertEqual(saved['training']['epochs'], 1)
            self.assertTrue((run / 'training_history.png').is_file())
            self.assertEqual({path.name for path in run.iterdir()}, {
                'checkpoint.pth', 'or_soft.yaml', 'training_history.png',
                'dataset.xlsx', 'training_history.csv',
                'validation_predictions.csv', 'validation_metrics.json'})
            history = pd.read_csv(run / 'training_history.csv')
            self.assertEqual(history.epoch.tolist(), [1])
            self.assertEqual(history.train_primary_loss.tolist(), [1.])
            self.assertEqual(history.val_primary_loss.tolist(), [1.])
            used_dataset, _ = load_workbook(run / 'dataset.xlsx', check_files=True)
            self.assertEqual(used_dataset.loc[used_dataset.training_eligible, 'image_id'].tolist(),
                             self.records.loc[self.records.split.eq('train'), 'image_id'].tolist())
            _, restored, checkpoint = load_checkpoint(run)
            self.assertEqual(restored['model']['head'], 'or_soft')
            self.assertEqual(restored['cohort']['training_image_ids'],
                             self.records.loc[self.records.split.eq('train'), 'image_id'].tolist())
            self.assertIn('dataset_snapshot', checkpoint)
            self.assertEqual((run / 'dataset.xlsx').read_bytes(), checkpoint['dataset_snapshot'])
            second = train(self.config, output_root='override models', overrides={'epochs': 1})
            self.assertEqual(second.parent, self.config.parent / 'override models')
            self.assertNotEqual(second.name, run.name)
            self.assertEqual(self.config.read_bytes(), original)

    def test_whole_image_resnet_trains_reloads_and_logs_input_mode(self):
        import torch
        from auto_qc.CNN.training import train
        from auto_qc.CNN.utils import load_checkpoint
        from openpyxl import load_workbook as read_excel
        self.settings['model'].pop('batch_normalization', None)
        self.settings['model'].update(backbone='resnet18', pretrained=False, head='or_soft_binary')
        self.settings['loss'].update(name='joint_bce', binary_loss_weight=1.)
        self.settings['data']['input_mode'] = 'whole_image'
        self.save()
        with patch('auto_qc.CNN.training.code_version', return_value={}):
            run = train(self.config, overrides={'epochs': 1})
        self.assertEqual(len(list(run.iterdir())), 7)
        model, config, checkpoint = load_checkpoint(run)
        self.assertEqual(config['preprocessing']['input_mode'], 'whole_image')
        with torch.inference_mode():
            ordinal, binary = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(ordinal.shape, (1, 4))
        self.assertEqual(binary.shape, (1,))
        metrics = json.loads((run / 'validation_metrics.json').read_text())
        self.assertEqual(metrics['inputs_per_image'], 1)
        self.assertEqual(metrics['input_mode'], 'whole_image')
        book = read_excel(self.root / 'experiment_log.xlsx')
        values = list(book.active.values)
        row = dict(zip(values[0], values[1]))
        self.assertEqual(row['Input mode'], 'whole_image')
        book.close()

    def test_four_convolution_iqanet_keeps_patch_training_and_seven_outputs(self):
        from auto_qc.CNN.training import train
        from auto_qc.CNN.utils import load_checkpoint
        from openpyxl import load_workbook as read_excel
        self.settings['model'].update(backbone='iqanet4', pretrained=False, head='or_soft_binary')
        self.settings['loss'].update(name='joint_bce', binary_loss_weight=1.)
        self.save()
        with patch('auto_qc.CNN.training.code_version', return_value={}):
            run = train(self.config, overrides={'epochs':1})
        self.assertEqual(len(list(run.iterdir())),7)
        self.assertTrue((run/'iqanet4_or_soft_binary.yaml').exists())
        model, config, _ = load_checkpoint(run)
        self.assertEqual(model.conv_layers,4)
        self.assertEqual(config['model']['backbone'],'iqanet4')
        metrics=json.loads((run/'validation_metrics.json').read_text())
        self.assertEqual(metrics['inputs_per_image'],9)
        book=read_excel(self.root/'experiment_log.xlsx')
        values=list(book.active.values)
        self.assertEqual(dict(zip(values[0],values[1]))['Architecture'],'iqanet4')
        book.close()

    def prepare_build_recipe(self):
        """Write the three source files used by dataset-builder integration tests."""
        labels = self.root / 'labels'
        labels.mkdir()
        ganaav = labels / SOURCE_FILENAMES['ganaav']
        pd.DataFrame({'image_filename': [value + '.png' for value in self.records.image_id],
                      'image_grading': [2, 3, 4]}).to_csv(ganaav, index=False)
        ratings = labels / SOURCE_FILENAMES['ratings']
        pd.read_excel(self.variant / 'multigrader_labels.xlsx').to_excel(ratings, sheet_name='Grades', index=False)
        specs = {'ganaav': {'path': 'old/ganaav.csv', 'sha256': sha256(ganaav)},
                 'ratings': {'path': 'old/ratings.xlsx', 'sha256': sha256(ratings), 'sheet': 'Grades'},
                 'split': self.manifest['sources']['split']}
        self.manifest['sources'] = specs
        self.manifest['source_exclusions'] = []
        write_json(self.variant / 'manifest.json', self.manifest)
        build = {'local_root': self.settings['local_root'], 'ganaav': specs['ganaav'],
                 'split': specs['split'], 'allowed_unmatched_ids': [],
                 'profiles': [{'name': 'reviewed_v2', 'ratings': specs['ratings'], 'graders': self.graders}],
                 'shane_authority': 'synthetic fixture current ratings'}
        path = self.config.parent / 'build.yaml'
        write_json(path, build)
        return path

    def test_build_verification_uses_root_label_names_and_preserves_old_metadata(self):
        from data.tools.generate_datasets import build_datasets
        path = self.prepare_build_recipe()
        before = (self.variant / 'manifest.json').read_bytes()
        self.assertEqual(len(build_datasets(path, verify_existing=True)), 5)
        self.assertEqual((self.variant / 'manifest.json').read_bytes(), before)

    def test_new_build_matches_existing_labels_and_preserves_split(self):
        from data.tools.generate_datasets import build_datasets
        from data.labels import logical_hash
        path = self.prepare_build_recipe()
        output = self.base / 'new labels'
        split = self.root / 'datasets' / SOURCE_FILENAMES['split']
        before = split.read_bytes()
        self.assertEqual(len(build_datasets(path, output_dir=output)), 5)
        generated = output / 'reviewed_v2'
        self.assertEqual({p.name for p in generated.iterdir()}, {*VARIANTS, 'manifest.json'})
        for name in VARIANTS:
            expected, _ = load_workbook(self.variant / name, local_root=self.root)
            actual, _ = load_workbook(generated / name, local_root=self.root)
            self.assertEqual(logical_hash(actual), logical_hash(expected))
        hashes = {p.name: sha256(p) for p in generated.iterdir()}
        build_datasets(path, verify_existing=True, output_dir=output)
        self.assertEqual(hashes, {p.name: sha256(p) for p in generated.iterdir()})
        self.assertEqual(split.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            build_datasets(path, output_dir=output)

    def test_historical_builder_still_writes_loadable_target_workbooks(self):
        from auto_qc.CNN.rerun import build_historical_datasets
        path = self.prepare_build_recipe()
        build = json.loads(path.read_text(encoding='utf-8'))
        del build['local_root']
        build['ganaav']['path'] = str(self.root / 'labels' / SOURCE_FILENAMES['ganaav'])
        build['split']['path'] = str(self.root / 'datasets' / SOURCE_FILENAMES['split'])
        split = pd.read_csv(build['split']['path'])
        split['image_path'] = split.image_id.map(lambda image_id: str(self.root / 'images' / (image_id + '.png')))
        split.to_csv(build['split']['path'], index=False)
        build['split']['sha256'] = sha256(build['split']['path'])
        build['profiles'][0]['ratings']['path'] = str(self.root / 'labels' / SOURCE_FILENAMES['ratings'])
        build['output_dir'] = str(self.base / 'historical labels')
        write_json(path, build)
        results = build_historical_datasets(path)
        self.assertEqual(len(results), 8)
        for result in results:
            frame, _ = load_workbook(result['path'], check_files=True)
            self.assertEqual(set(frame.image_id), set(self.records.image_id))


def read_config_yaml(path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))


class LearningCurveLabels(unittest.TestCase):
    def test_single_and_multi_head_titles_legends_and_values(self):
        from auto_qc.CNN.training import history_figure
        from matplotlib.figure import Figure
        history = pd.DataFrame({'epoch': [1, 2], 'train_primary_loss': [3., 2.], 'val_primary_loss': [4., 3.],
                                'train_binary_loss': [.4, .3], 'val_binary_loss': [.6, .5],
                                'train_total_loss': [3.4, 2.3], 'val_total_loss': [4.6, 3.5]})
        for loss, panels in [('ordinal_bce', 1), ('hard_ce', 1), ('binary_bce', 1), ('joint_bce', 3)]:
            with self.subTest(loss=loss), tempfile.TemporaryDirectory() as directory:
                figures = []
                def inspect(figure, *args, **kwargs):
                    figures.append({'title': figure._suptitle.get_text(),
                                    'axes': [{'title': axis.get_title(),
                                              'labels': axis.get_legend_handles_labels()[1],
                                              'values': [line.get_ydata().copy() for line in axis.lines]}
                                             for axis in figure.axes]})
                with patch.object(Figure, 'savefig', inspect):
                    # savefig is inspected rather than rendered in this label/value test.
                    with patch.object(Path, 'replace'):
                        history_figure(history, Path(directory) / 'curve.png',
                                       {'experiment_id': 'my_model', 'loss': {'name': loss}}, 2)
                figure = figures[-1]
                self.assertEqual(figure['title'], 'Train/Validation loss for model my_model')
                self.assertEqual(len(figure['axes']), panels)
                for axis in figure['axes']:
                    labels = axis['labels']
                    self.assertEqual(labels[:2], ['Training loss', 'Validation loss'])
                np.testing.assert_array_equal(figure['axes'][0]['values'][0], history.train_primary_loss)
                if panels == 1:
                    self.assertEqual(figure['axes'][0]['title'], '')
                else:
                    self.assertEqual([axis['title'] for axis in figure['axes']], ['Ordinal head', 'Binary head', 'Total loss'])
                    np.testing.assert_array_equal(figure['axes'][2]['values'][1], history.val_total_loss)


if __name__ == '__main__':
    unittest.main()
