import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspired by:
#       Kang, L., Ye, P., Li, Y. and Doermann, D., 2014. Convolutional neural
#       networks for no-reference image quality assessment. In Proceedings of
#       the IEEE conference on computer vision and pattern recognition (pp.
#       1733-1740).


class IQAnet(nn.Module):
    def __init__(
        self,
        ker_size=5,
        n_kers=64,
        n1_nodes=600,
        n2_nodes=400,
        num_classes=2,
        batch_normalization=False,
        output_sigmoid=False,
        dropout_p: float = 0.5,
        conv_layers: int = 2
    ):
        super(IQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, n_kers, ker_size, stride=3, padding=0)
        self.bn1 = nn.BatchNorm2d(n_kers)  #   batch norm to original
        
        self.conv2 = nn.Conv2d(n_kers, n_kers // 2, 3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(n_kers // 2)

        self.fc1 = nn.Linear(n_kers, n1_nodes)
        self.fc2 = nn.Linear(n1_nodes, n2_nodes)
        # output for single-class or multiclass quality levels
        self.fc3_d = nn.Linear(n2_nodes, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        self.batch_normalization = batch_normalization

        self.dropout_p = dropout_p
        if conv_layers not in (2, 4):
            raise ValueError('IQAnet supports two or four convolutional layers')
        self.conv_layers = conv_layers
        if conv_layers == 4:
            # Add depth at the existing feature-map resolution. Keep the original
            # layers' initialization and subsequent dropout RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                self.conv3 = nn.Conv2d(n_kers // 2, n_kers // 2, 3, padding=1)
                self.conv4 = nn.Conv2d(n_kers // 2, n_kers // 2, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(n_kers // 2)
                self.bn4 = nn.BatchNorm2d(n_kers // 2)

    def forward_features(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

        # convolve
        h = self.conv1(x)
        if self.batch_normalization:
            h = self.bn1(h)
        h = F.relu(h)
        
        h = self.conv2(h)
        if self.batch_normalization:
            h = self.bn2(h)
        h = F.relu(h)

        if self.conv_layers == 4:
            h = self.conv3(h)
            if self.batch_normalization:
                h = self.bn3(h)
            h = F.relu(h)
            h = self.conv4(h)
            if self.batch_normalization:
                h = self.bn4(h)
            h = F.relu(h)

        # pool pos & neg
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h = torch.cat((h1, h2), 1)
        h = h.squeeze(3).squeeze(2)

        # fully connected
        h = F.relu(self.fc1(h))
        if self.dropout_p > 0:
            h = self.dropout(h)
        h = F.relu(self.fc2(h))
        if self.dropout_p > 0:
            h = self.dropout(h)

        return h

    def forward(self, x):
        h = self.forward_features(x)
        # classify
        d = self.fc3_d(h)
        if self.output_sigmoid:
            d = self.sigmoid(d)

        return d


class OrderedCoralHead(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.score = nn.Linear(features, 1, bias=False)
        self.first_bias = nn.Parameter(torch.tensor(1.0))
        self.raw_gaps = nn.Parameter(torch.full((3,), float(torch.log(torch.expm1(torch.tensor(.25))))))

    def forward(self, features):
        biases = torch.cat((self.first_bias.reshape(1), self.first_bias - F.softplus(self.raw_gaps).cumsum(0)))
        return self.score(features) + biases


class OrdinalIQAnet(IQAnet):
    def __init__(self, head='or_soft', **kwargs):
        super().__init__(num_classes=5, **kwargs)
        if head not in ('or_soft', 'coral', 'or_soft_binary'):
            raise ValueError(f'Unknown ordinal head: {head}')
        self.fc3_d = OrderedCoralHead(self.fc2.out_features) if head == 'coral' else nn.Linear(self.fc2.out_features, 4)
        self.head = head
        if head == 'or_soft_binary':
            # Preserve shared initialisation and subsequent dropout RNG of the control.
            with torch.random.fork_rng(devices=[]):
                self.binary_head = nn.Linear(self.fc2.out_features, 1)

    def forward(self, x):
        features = self.forward_features(x)
        ordinal = self.fc3_d(features)
        return (ordinal, self.binary_head(features).squeeze(1)) if self.head == 'or_soft_binary' else ordinal


class FrozenBinaryIQAnet(nn.Module):
    def __init__(self, source, seed):
        super().__init__()
        self.source = source.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.binary_head = nn.Linear(source.fc2.out_features, 1)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.source.forward_features(x)
            ordinal = self.source.fc3_d(features)
        return ordinal, self.binary_head(features).squeeze(1)


class ResNet18Quality(nn.Module):
    """Torchvision ResNet-18 features with the existing quality-head definitions.

    https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
    Keep the original 256px grayscale patches; repeat channels and apply ImageNet
    normalization inside the model so training and evaluation use the same input.
    """
    def __init__(self, head, dropout_p=.5, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout_p)
        self.head = head
        self.fc3_d = (OrderedCoralHead(features) if head == 'coral'
                      else nn.Linear(features, 5 if head == 'five_logits' else 4))
        if head == 'or_soft_binary':
            self.binary_head = nn.Linear(features, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        features = self.dropout(self.backbone((x - self.mean) / self.std))
        grades = self.fc3_d(features)
        return (grades, self.binary_head(features).squeeze(1)) if self.head == 'or_soft_binary' else grades


def make_model(settings, seed=42, load_source=True):
    settings = dict(settings)
    head = settings.pop('head')
    if settings.pop('num_grades', 5) != 5:
        raise ValueError('These methods require exactly five grades')
    source = settings.pop('source', None)
    if settings.get('output_sigmoid', False):
        raise ValueError('All supported objectives require raw logits')
    backbone = settings.pop('backbone', 'iqanet')
    pretrained = settings.pop('pretrained', False)
    if backbone == 'resnet18':
        if head not in ('five_logits', 'or_soft', 'coral', 'or_soft_binary'):
            raise ValueError('ResNet-18 does not support frozen_binary')
        return ResNet18Quality(head, settings.get('dropout_p', .5), pretrained and load_source)
    if backbone not in ('iqanet', 'iqanet4'):
        raise ValueError(f'Unknown backbone: {backbone}')
    if backbone == 'iqanet4':
        settings['conv_layers'] = 4
    if head == 'five_logits':
        return IQAnet(num_classes=5, **settings)
    if head == 'frozen_binary':
        if not source or source['architecture']['head'] != 'or_soft':
            raise ValueError('Frozen binary requires an explicit single-head OR-Soft source')
        model = make_model(source['architecture'], seed)
        if load_source:
            from .utils import sha256
            if sha256(source['checkpoint']) != source['sha256']:
                raise ValueError('Frozen source checkpoint hash mismatch')
            state = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
            model.load_state_dict(state.get('model_state', state), strict=True)
        return FrozenBinaryIQAnet(model, seed)
    return OrdinalIQAnet(head=head, **settings)
