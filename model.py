import torch
from torch import nn
from torch.nn import functional as F


class IQAnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 50, 5, stride=3)
        self.conv2 = nn.Conv2d(50, 25, 3, stride=2)
        self.fc1 = nn.Linear(50, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3_d = nn.Linear(400, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = F.relu(self.conv1(images))
        features = F.relu(self.conv2(features))
        maximum = F.adaptive_max_pool2d(features, 1)
        minimum = -F.adaptive_max_pool2d(-features, 1)
        features = torch.cat((maximum, minimum), dim=1).flatten(1)
        features = self.dropout(self.fc1(features))
        features = self.dropout(self.fc2(features))
        return self.fc3_d(features)
