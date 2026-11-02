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

    def forward(self, x):
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

        # classify
        d = self.fc3_d(h)
        if self.output_sigmoid:
            d = self.sigmoid(d)

        return d
