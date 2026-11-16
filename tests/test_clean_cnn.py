"""Focused regressions for the consolidated original-layout pipeline."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from auto_qc.CNN.dataset import VOTES, ORDINAL, target_tensor
from auto_qc.CNN.dataset import (ImageQualityDataset, ImagePatchDataset, image_transform,
                                extract_patches, patch_dataset, subsample_subimages)
from data.labels import (generate_targets, load_workbook, save_workbook,
                                          logical_hash, validate_records)
from auto_qc.CNN.losses import grade_probabilities, isotonic_projection, loss_components, ordering_diagnostics
from auto_qc.CNN.model import make_model, FrozenBinaryIQAnet
from auto_qc.CNN.metrics import calculate_combined_metrics
from auto_qc.CNN.utils import class_weights, fingerprint, seed_everything, sha256, state_copy, write_json


def frame():
    ids = ['DARC-P2_SITE_001_OD_V1_2020_1.1_0', 'DARC-P2_SITE_002_OD_V1_2020_1.0_0', 'VISUS-NHP_SITE_003_OD_D07_2020_5.1_0']
    return pd.DataFrame({'image_id': ids, 'image_path': [p+'.png' for p in ids],
                         'patient_id': ['DARC-P2_SITE_001', 'DARC-P2_SITE_002', 'VISUS-NHP_SITE_003'],
                         'project': ['DARC-P2', 'DARC-P2', 'VISUS-NHP'], 'project_site': ['SITE']*3,
                         'split': ['train', 'validation', 'test'], 'mode': ['train', 'val', 'test'],
                         'ganaav_grade': [2, 1, 3], 'shane_grade': [3, 5, 3]})


class Targets(unittest.TestCase):
    def test_only_current_shane_ratings_are_read(self):
        from data.tools import generate_datasets as generation
        source = frame()
        ganaav = source[['image_id', 'patient_id', 'project', 'project_site']].copy()
        ganaav['image_grading'] = source.ganaav_grade
        current = source[['image_id', 'ganaav_grade', 'shane_grade']].copy()
        current['shane_grade'] = [5, 3, 1]
        tables = {'split': source, 'ganaav': ganaav, 'current': current}
        build = {'split': {'path': 'split'}, 'ganaav': {'path': 'ganaav'}, 'allowed_unmatched_ids': []}
        profile = {'ratings': {'path': 'current'}, 'graders': ['ganaav_grade', 'shane_grade']}
        with mock.patch.object(generation, 'pinned_table', side_effect=lambda spec, **kwargs: tables[spec['path']].copy()) as read:
            with mock.patch.object(generation, 'validate_records') as validate:
                records, exclusions = generation.join_sources(build, profile)
        self.assertEqual(records.shane_grade.tolist(), [5, 3, 1])
        self.assertFalse(any('_original_grade' in name for name in records))
        self.assertEqual(exclusions, [])
        self.assertEqual([call.args[0]['path'] for call in read.call_args_list], ['split', 'ganaav', 'current'])
        validate.assert_called_once()
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'build.json'
            write_json(config, {'profiles': [profile, profile]})
            with self.assertRaisesRegex(ValueError, 'one current ratings source'):
                generation.build_datasets(config)

    def test_plain_labels_and_shared_split(self):
        from data.tools.generate_datasets import VARIANTS, write_labels
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = frame()
            split = directory / 'split.csv'
            source.to_csv(split, index=False)
            graders = ['ganaav_grade', 'shane_grade']
            manifest = {'schema': 'autoqc_labels_v1', 'sources': {'split': {'path': str(split), 'sha256': sha256(split)}},
                        'graders': graders, 'ratings_file': 'multigrader_labels.xlsx',
                        'held_out_project': 'VISUS-NHP', 'variants': {}}
            for filename, (kind, selected) in VARIANTS.items():
                labels = pd.DataFrame({'image_filename': source.image_path.map(lambda p: Path(p).name)})
                if selected is None:
                    labels[graders] = source[graders]
                else:
                    values = source[selected].mean(axis=1)
                    labels['label'] = np.ceil(values) if kind == 'ceiling_mean' else values
                write_labels(directory / filename, labels)
                manifest['variants'][filename] = {'target_kind': kind, 'graders': selected or graders,
                    'mean_missing_policy': 'require_all' if selected else 'available',
                    'label_rule': kind, 'sha256': sha256(directory / filename)}
            write_json(directory / 'manifest.json', manifest)
            for filename in VARIANTS:
                loaded, _ = load_workbook(directory / filename)
                self.assertEqual(loaded.image_id.tolist(), source.image_id.tolist())
                self.assertEqual(loaded.split.tolist(), source.split.tolist())
            mean, _ = load_workbook(directory / 'shane_ganaav_average_labels.xlsx')
            rounded, _ = load_workbook(directory / 'shane_ganaav_average_rounded_labels.xlsx')
            self.assertEqual(mean.target_mean_grade.iloc[0], 2.5)
            self.assertEqual(rounded.target_grade.iloc[0], 3)
            multi = directory / 'multigrader_labels.xlsx'
            ordinal, meta = load_workbook(multi)
            np.testing.assert_array_equal(ordinal[ORDINAL].iloc[0].to_numpy(), [1, .5, 0, 0])
            soft, _ = load_workbook(multi, target_kind='soft_categorical')
            np.testing.assert_array_equal(soft[VOTES].iloc[0].to_numpy(), [0, .5, .5, 0, 0])
            snapshot = directory / 'snapshot.xlsx'
            save_workbook(snapshot, ordinal, meta)
            restored, _ = load_workbook(snapshot)
            torch.testing.assert_close(target_tensor(ordinal, 'ordinal'), target_tensor(restored, 'ordinal'))
            with self.assertRaisesRegex(ValueError, 'encoding/graders'):
                load_workbook(directory / 'shane_labels.xlsx', target_kind='exact_mean')
            split.write_text(split.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'split hash'):
                load_workbook(multi)
            manifest['sources']['split']['sha256'] = sha256(split)
            write_json(directory / 'manifest.json', manifest)
            with multi.open('ab') as handle:
                handle.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'hash/schema'):
                load_workbook(multi)

    def test_hand_worked_and_all_pairs(self):
        for g in range(1, 6):
            for s in range(1, 6):
                f = frame().iloc[:1].copy()
                f['ganaav_grade'], f['shane_grade'] = g, s
                mean = generate_targets(f, 'exact_mean', ['ganaav_grade', 'shane_grade'])
                ceiling = generate_targets(f, 'ceiling_mean', ['ganaav_grade', 'shane_grade'])
                ordinal = generate_targets(f, 'ordinal', ['ganaav_grade', 'shane_grade'])
                self.assertEqual(mean.target_mean_grade.iloc[0], (g+s)/2)
                self.assertEqual(ceiling.target_grade.iloc[0], np.ceil((g+s)/2))
                np.testing.assert_array_equal(ordinal[ORDINAL].iloc[0], [(int(g>k)+int(s>k))/2 for k in range(1, 5)])
                self.assertEqual(ordinal.target_binary_unacceptable.iloc[0], (int(g>=3)+int(s>=3))/2)
                self.assertEqual(target_tensor(mean, 'exact_mean').dtype, torch.float32)
                self.assertNotIn('target_class_index', mean)
        f = generate_targets(frame(), 'ordinal', ['ganaav_grade', 'shane_grade'])
        np.testing.assert_array_equal(f[ORDINAL].iloc[0], [1, .5, 0, 0])
        self.assertFalse(np.array_equal(f[VOTES].iloc[1], f[VOTES].iloc[2]))

    def test_missing_invalid_and_alias(self):
        f = frame().rename(columns={'ganaav_grade': 'previousStudent_grade'})
        f.loc[0, 'shane_grade'] = np.nan
        soft = generate_targets(f, 'soft_categorical', ['previousStudent_grade', 'shane_grade'])
        self.assertEqual(soft.rating_count.iloc[0], 1)
        self.assertEqual(soft.target_grade_2.iloc[0], 1)
        exact = generate_targets(f, 'exact_mean', ['previousStudent_grade', 'shane_grade'])
        self.assertFalse(exact.eligible.iloc[0])
        self.assertTrue(pd.isna(exact.target_mean_grade.iloc[0]))
        available = generate_targets(f, 'exact_mean', ['previousStudent_grade', 'shane_grade'], 'available')
        self.assertEqual(available.target_mean_grade.iloc[0], 2)
        for value in (0, 6, 2.5, True, 'bad'):
            bad = frame().astype({'shane_grade': object})
            bad.loc[0, 'shane_grade'] = value
            with self.assertRaises(ValueError):
                generate_targets(bad, 'ordinal', ['ganaav_grade', 'shane_grade'])

    def test_workbook_consumed_targets_and_determinism(self):
        f = generate_targets(frame(), 'ordinal', ['ganaav_grade', 'shane_grade'])
        meta = {'target_kind': 'ordinal', 'graders': ['ganaav_grade', 'shane_grade'], 'held_out_project': 'VISUS-NHP'}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'dataset.xlsx'
            save_workbook(path, f, meta)
            loaded, _ = load_workbook(path)
            self.assertEqual(logical_hash(f), logical_hash(loaded.fillna({'exclusion_reason': ''})))
            torch.testing.assert_close(target_tensor(f, 'ordinal'), target_tensor(loaded, 'ordinal'))
            loaded['training_eligible'] = loaded.split.eq('train')
            snapshot = Path(temp) / 'snapshot.xlsx'
            save_workbook(snapshot, loaded, meta)
            restored, _ = load_workbook(snapshot)
            self.assertEqual(logical_hash(loaded), logical_hash(restored))
            with self.assertRaises(FileExistsError):
                save_workbook(path, f, meta)
            loaded.loc[0, ORDINAL[0]] = .2
            with pd.ExcelWriter(path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                loaded.to_excel(writer, sheet_name='dataset', index=False)
            with self.assertRaisesRegex(ValueError, 'target disagrees'):
                load_workbook(path)

    def test_split_integrity(self):
        f = frame()
        validate_records(f)
        for bad in (pd.concat([f, f.iloc[:1]]), f.assign(patient_id=['DARC-P2_SITE_001']*3), f.assign(split=['train']*3)):
            with self.assertRaises(ValueError):
                validate_records(bad)


class ImagesAndPatches(unittest.TestCase):
    def test_whole_image_preserves_full_frame_flips_and_evaluation_pixels(self):
        from PIL import Image
        from auto_qc.CNN.dataset import extract_inputs
        with tempfile.TemporaryDirectory() as temp:
            pixels = np.zeros((64, 128), dtype=np.uint8)
            pixels[:32, :64], pixels[:32, 64:] = 40, 90
            pixels[32:, :64], pixels[32:, 64:] = 160, 230
            path = Path(temp) / 'whole.png'
            Image.fromarray(pixels).save(path)
            records = generate_targets(frame().iloc[:1].copy(), 'ordinal', ['ganaav_grade', 'shane_grade'])
            records['image_path'] = str(path)
            data = ImagePatchDataset(records, 'ordinal', augment=True, input_mode='whole_image')
            self.assertEqual(len(data), 2)
            self.assertEqual(data.cache_nbytes, 256 * 256)
            expected = torch.from_numpy(np.array(Image.fromarray(pixels).resize(
                (256, 256), Image.Resampling.BILINEAR))).unsqueeze(0).float() / 255
            torch.testing.assert_close(data[0][0], expected, rtol=0, atol=0)
            torch.testing.assert_close(data[1][0], expected.flip(-1), rtol=0, atol=0)
            torch.testing.assert_close(extract_inputs(path, 'whole_image')[0], expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(data[0][1], data[1][1]))

    def test_prepared_cache_survives_shuffle_without_reopening_images(self):
        from PIL import Image
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as temp:
            records = pd.concat([frame().iloc[:1]]*33, ignore_index=True)
            paths = []
            for i in range(len(records)):
                path = Path(temp) / f'image_{i}.png'
                # Cover all 256 pixel values, including the quantisation boundaries.
                pixels = np.tile(np.arange(256, dtype=np.uint8), (768, 3)) if i == 0 else (
                    (np.arange(17*19).reshape(17, 19)+i) % 256).astype(np.uint8)
                Image.fromarray(pixels).save(path)
                paths.append(str(path))
            records['image_path'] = paths
            records = generate_targets(records, 'ordinal', ['ganaav_grade', 'shane_grade'])
            expected = [extract_patches(path) for path in paths]
            with mock.patch('PIL.Image.open', wraps=Image.open) as opened:
                data = ImagePatchDataset(records, 'ordinal', augment=True)
                self.assertEqual(opened.call_count, len(records))
            self.assertEqual(data.cache_nbytes, len(records)*9*256*256)
            original = [(expected[i][p], data.targets[i]) for i in range(len(records)) for p in range(9)]
            original += [(image.flip(-1), target) for image, target in original]
            # Same sample order, targets and exact float pixels over two shuffled epochs.
            old_loader = DataLoader(original, batch_size=16, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))
            new_loader = DataLoader(data, batch_size=16, shuffle=True,
                                    generator=torch.Generator().manual_seed(42))
            with mock.patch('PIL.Image.open', side_effect=AssertionError('Image decoded during epoch')):
                for _ in range(2):
                    for (old_x, old_y), (new_x, new_y) in zip(old_loader, new_loader):
                        self.assertTrue(torch.equal(old_x, new_x))
                        self.assertTrue(torch.equal(old_y, new_y))

    def test_original_loader_and_current_patch_adapter(self):
        from PIL import Image
        from torchvision import transforms
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'image.png'
            pixels = (np.arange(317*509*3).reshape(317, 509, 3) % 256).astype(np.uint8)
            Image.fromarray(pixels).save(path)
            table = pd.DataFrame({'Image': [str(path)], 'Quality': [3]}, index=[17])
            csv = Path(temp) / 'labels.csv'
            table.to_csv(csv, index=False)
            original = ImageQualityDataset(csv, image_transform(), label_mapping={3: 2})
            self.assertEqual(len(original), 1)
            self.assertEqual(original[0][1], 2)
            image = transforms.ToPILImage()(original[0][0])
            expected = torch.stack([transforms.ToTensor()(image.crop((c*256, r*256, (c+1)*256, (r+1)*256)))
                                    for r in range(3) for c in range(3)])
            torch.testing.assert_close(extract_patches(path), expected, rtol=0, atol=0)
            records = generate_targets(frame().iloc[:1].assign(image_path=str(path)),
                                       'ordinal', ['ganaav_grade', 'shane_grade'])
            data = ImagePatchDataset(records, 'ordinal', augment=True)
            self.assertEqual(len(data), 18)
            for i in range(9):
                torch.testing.assert_close(data[i][0], expected[i], rtol=0, atol=0)
                torch.testing.assert_close(data[i+9][0], expected[i].flip(-1), rtol=0, atol=0)
                torch.testing.assert_close(data[i][1], torch.tensor([1., .5, 0., 0.]), rtol=0, atol=0)
            alias = table.reset_index(drop=True).rename(columns={'Image': 'image_filename', 'Quality': 'image_grading'})
            self.assertEqual(ImageQualityDataset(alias, image_transform(), {3: 2})[0][1], 2)

    def test_original_patch_helpers(self):
        image = torch.arange(3*6*8, dtype=torch.float32).reshape(3, 6, 8)
        quadrants = patch_dataset([(image, 4)])
        self.assertEqual(len(quadrants), 4)
        for index, (patch, label) in enumerate(quadrants):
            row, column = divmod(index, 2)
            torch.testing.assert_close(patch, image[:, row*3:(row+1)*3, column*4:(column+1)*4])
            self.assertEqual(label, 4)
        a = torch.zeros(1, 6, 6)
        b = torch.ones(1, 6, 6)
        patches, positions = subsample_subimages([a, b], method='all', patch_size=2,
                                                class_labels=False, verbose=False)
        self.assertEqual(len(patches), 18)
        self.assertEqual(positions.subimage_position.tolist(), list(range(1, 10))*2)
        self.assertTrue(all(torch.equal(patch, torch.zeros(1, 2, 2)) for patch in patches[:9]))
        self.assertTrue(all(torch.equal(patch, torch.ones(1, 2, 2)) for patch in patches[9:]))
        with self.assertRaises(ValueError):
            subsample_subimages([a], method='bad', verbose=False)


class ModelsAndLosses(unittest.TestCase):
    def test_projection_identity_invalid_and_probabilities(self):
        valid = torch.tensor([[1., .8, .4, 0.]])
        torch.testing.assert_close(valid, isotonic_projection(valid), rtol=0, atol=0)
        invalid = torch.tensor([[.9, .2, .8, .1]])
        torch.testing.assert_close(isotonic_projection(invalid), torch.tensor([[.9, .5, .5, .1]]))
        self.assertEqual(ordering_diagnostics(invalid)['violating_patch_count'], 1)
        logits = torch.logit(invalid)
        p = grade_probabilities(logits, 'or_soft')
        self.assertTrue((p >= 0).all())
        torch.testing.assert_close(p.sum(1), torch.ones(1))

    def test_gradients_seeding_and_independent_heads(self):
        target = torch.tensor([[1., .5, 0., 0.], [1., 1., .5, 0.]])
        for head, name in [('five_logits', 'hard_ce'), ('five_logits', 'soft_ce'), ('five_logits', 'continuous_cdw_ce'),
                           ('coral', 'ordinal_bce'), ('or_soft', 'ordinal_bce'), ('or_soft_binary', 'joint_bce')]:
            seed_everything(42)
            settings = {'head': head, 'batch_normalization': False, 'output_sigmoid': False}
            model = make_model(settings)
            initial = fingerprint(model)
            seed_everything(42)
            self.assertEqual(initial, fingerprint(make_model(settings)))
            x = torch.randn(2, 1, 32, 32)
            t = torch.tensor([1, 2]) if name == 'hard_ce' else torch.tensor([2.5, 3.]) if name == 'continuous_cdw_ce' else target
            if name == 'soft_ce':
                t = torch.tensor([[0., .5, .5, 0., 0.], [0., 0., 1., 0., 0.]])
            values = loss_components(model(x), t, {'name': name, 'alpha': 1., 'binary_loss_weight': 1.})
            values[2].backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            if head == 'or_soft':
                self.assertEqual(tuple(model.fc3_d.weight.shape), (4, 400))

    def test_frozen_modes_gradients_and_ordinal_identity(self):
        seed_everything(42)
        source = make_model({'head': 'or_soft', 'batch_normalization': True}).eval()
        model = FrozenBinaryIQAnet(source, 42)
        snapshot = fingerprint(source)
        x = torch.randn(2, 1, 32, 32)
        before = model(x)[0].detach().clone()
        optimizer = torch.optim.AdamW(model.binary_head.parameters(), lr=.0001, weight_decay=.05)
        model.train()
        self.assertFalse(any(m.training for m in model.source.modules()))
        loss_components(model(x), torch.tensor([[1., .5, 0., 0.]]*2), {'name': 'binary_bce'})[2].backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in source.parameters()))
        self.assertEqual(snapshot, fingerprint(source))
        self.assertTrue(torch.equal(before, model(x)[0]))
        restored = copy.deepcopy(model)
        restored.load_state_dict(state_copy(model))
        model.eval(); restored.eval()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model(x), restored(x))))

    def test_best_state_and_class_order(self):
        model = torch.nn.Linear(2, 1)
        best = state_copy(model)
        with torch.no_grad():
            model.weight.add_(1)
        self.assertFalse(torch.equal(best['weight'], model.weight))
        model.load_state_dict(best)
        self.assertTrue(torch.equal(best['weight'], model.weight))
        f = pd.DataFrame({'target_class_index': [4, 0, 0, 1, 2, 3, 4, 4]})
        expected = torch.tensor([8/10, 8/5, 8/5, 8/5, 8/15])
        a = class_weights(f, {'mode': 'inverse_frequency'})
        torch.testing.assert_close(a, expected)
        torch.testing.assert_close(a, class_weights(f.iloc[::-1], {'mode': 'inverse_frequency'}))

    def test_patch_aggregation_and_binary_mapping(self):
        patches = torch.tensor([[.1, .5, .2, .1, .1], [.1, .1, .2, .3, .3]])
        p = patches.mean(0).numpy()[None, :]
        f = generate_targets(frame().iloc[:1], 'soft_categorical', ['ganaav_grade', 'shane_grade'])
        metrics, details = calculate_combined_metrics(f, p, ['ganaav_grade', 'shane_grade'], .5, np.array([.5]))
        self.assertEqual(details.predicted_grade.iloc[0], int(p.argmax()+1))
        self.assertEqual(details.binary_reference.iloc[0], 'acceptable')
        self.assertEqual(details.binary_head_prediction.iloc[0], 'unacceptable')
        self.assertEqual(metrics['binary_dedicated_head']['FP'], 1)
        self.assertEqual(metrics['binary_grade_argmax']['TN'], 1)


if __name__ == '__main__':
    unittest.main()
