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
        dropout_p: float = 0.5
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


def make_model(settings, seed=42, load_source=True):
    settings = dict(settings)
    head = settings.pop('head')
    if settings.pop('num_grades', 5) != 5:
        raise ValueError('These methods require exactly five grades')
    source = settings.pop('source', None)
    if settings.get('output_sigmoid', False):
        raise ValueError('All supported objectives require raw logits')
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
