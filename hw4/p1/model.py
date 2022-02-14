from torch import nn
import torch

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.conv = Convnet(in_channels, hid_channels, out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(1600, 800),
            nn.RReLU(),
            nn.Dropout(),
            nn.Linear(800, 400)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x)
        return x
