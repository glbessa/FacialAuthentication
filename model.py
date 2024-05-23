import torch
from torch import nn

import torchvision

class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.resnet = torchvision.models.resnet50(weights=None)
    self.fc = nn.Sequential(
        nn.Linear(self.resnet.fc.in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    self.resnet.fc = nn.Flatten()

  def forward(self, a, b):
    output1 = self.resnet(a)
    output2 = self.resnet(b)

    outputs = torch.subtract(output1, output2)
    outputs = self.fc(outputs)

    return outputs