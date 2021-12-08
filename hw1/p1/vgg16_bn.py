
import torch.nn as nn
from torchvision import models

model_name = 'vgg16_bn'

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.name = 'vgg16_bn'
        self.model = getattr(models, self.name)(pretrained = True)
        self.model.classifier[6] = nn.Linear(4096, 50)
    def forward(self, x):
        x = self.model(x)
        return x

