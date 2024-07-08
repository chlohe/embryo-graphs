import torch
import torch.nn as nn

from torchvision.models.resnet import resnet50, ResNet50_Weights

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.preprocess = self.weights.transforms()
        self.net = resnet50(weights=self.weights)
        self.net.fc = nn.Linear(2048, 2)
        # self.net.conv1 = nn.Conv2d(11, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        # x = x[:, 4:7, :, :]
        # x = self.preprocess(x)
        return self.net(x)