"""Focused regressions for the consolidated original-layout pipeline."""
import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from auto_qc.CNN.dataset import (VOTES, ORDINAL, generate_targets, load_workbook, save_workbook,
                                logical_hash, target_tensor, validate_records)
from auto_qc.CNN.losses import grade_probabilities, isotonic_projection, loss_components, ordering_diagnostics
from auto_qc.CNN.model import make_model, FrozenBinaryIQAnet
from auto_qc.CNN.metrics import calculate_combined_metrics
from auto_qc.CNN.utils import class_weights, fingerprint, seed_everything, state_copy


def frame():
    ids = ['DARC-P2_SITE_001_OD_V1_2020_1.1_0', 'DARC-P2_SITE_002_OD_V1_2020_1.0_0', 'VISUS-NHP_SITE_003_OD_D07_2020_5.1_0']
    return pd.DataFrame({'image_id': ids, 'image_path': [p+'.png' for p in ids],
                         'patient_id': ['DARC-P2_SITE_001', 'DARC-P2_SITE_002', 'VISUS-NHP_SITE_003'],
                         'project': ['DARC-P2', 'DARC-P2', 'VISUS-NHP'], 'project_site': ['SITE']*3,
                         'split': ['train', 'validation', 'test'], 'mode': ['train', 'val', 'test'],
                         'ganaav_grade': [2, 1, 3], 'shane_grade': [3, 5, 3]})


class Targets(unittest.TestCase):
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
