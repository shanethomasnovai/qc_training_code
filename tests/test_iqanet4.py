"""Four-convolution depth experiment, preserving original heads and initialization."""
import unittest

import torch

from auto_qc.CNN.model import make_model
from auto_qc.CNN.losses import loss_components


class FourConvolutionIQAnet(unittest.TestCase):
    def model(self, backbone):
        torch.manual_seed(42)
        return make_model({'backbone':backbone, 'head':'or_soft_binary', 'dropout_p':0.})

    def test_original_layers_and_rng_are_preserved(self):
        original = self.model('iqanet')
        original_rng = torch.get_rng_state().clone()
        deeper = self.model('iqanet4')
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, deeper.state_dict()[name]), name)
        self.assertEqual(sum(isinstance(m,torch.nn.Conv2d) for m in original.modules()), 2)
        self.assertEqual(sum(isinstance(m,torch.nn.Conv2d) for m in deeper.modules()), 4)
        self.assertEqual(deeper.conv3.stride, (1,1))
        self.assertEqual(deeper.conv4.stride, (1,1))
        self.assertEqual(deeper.conv4.out_channels, original.conv2.out_channels)

    def test_all_four_convolutions_and_heads_train_and_reload(self):
        model = self.model('iqanet4')
        images = torch.rand(2,1,64,64)
        targets = torch.tensor([[1.,.5,0.,0.],[1.,1.,.5,0.]])
        outputs = model(images)
        self.assertEqual(outputs[0].shape, (2,4))
        self.assertEqual(outputs[1].shape, (2,))
        loss = loss_components(outputs,targets,{'name':'joint_bce','binary_loss_weight':1.})[2]
        loss.backward()
        for layer in (model.conv1,model.conv2,model.conv3,model.conv4,model.fc3_d,model.binary_head):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()),0.)
        restored = self.model('iqanet4')
        restored.load_state_dict(model.state_dict(),strict=True)
        model.eval();restored.eval()
        with torch.no_grad():
            for before,after in zip(model(images),restored(images)):
                self.assertTrue(torch.equal(before,after))


if __name__=='__main__':
    unittest.main()
