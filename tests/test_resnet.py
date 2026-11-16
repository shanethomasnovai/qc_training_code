"""ResNet input adaptation, both head gradients and offline checkpoint reload."""
import unittest
from unittest.mock import patch

import torch
from torchvision.models import resnet18

from auto_qc.CNN.losses import loss_components
from auto_qc.CNN.model import make_model
from auto_qc.CNN.utils import state_copy


class ResNetQuality(unittest.TestCase):
    def test_two_heads_train_and_restore_without_downloading_weights(self):
        settings = {'backbone': 'resnet18', 'pretrained': True,
                    'head': 'or_soft_binary', 'dropout_p': .5}
        with patch('torchvision.models.resnet18', wraps=resnet18) as builder:
            model = make_model(settings, load_source=False)
        self.assertIsNone(builder.call_args.kwargs['weights'])
        images = torch.rand(2, 1, 64, 64)
        adapted = []
        hook = model.backbone.register_forward_pre_hook(lambda module, args: adapted.append(args[0].detach()))
        ordinal, binary = model(images)
        hook.remove()
        self.assertEqual(ordinal.shape, (2, 4))
        self.assertEqual(binary.shape, (2,))
        expected = (images.expand(-1, 3, -1, -1) - model.mean) / model.std
        torch.testing.assert_close(adapted[0], expected)
        targets = torch.tensor([[1., .5, .25, 0.], [1., 1., .5, .25]])
        loss = loss_components((ordinal, binary), targets, {'name': 'joint_bce', 'binary_loss_weight': 1.})[2]
        loss.backward()
        for parameter in (model.backbone.conv1.weight, model.fc3_d.weight, model.binary_head.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        restored = make_model(settings, load_source=False)
        restored.load_state_dict(state_copy(model), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            for before, after in zip(model(images), restored(images)):
                self.assertTrue(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main()
