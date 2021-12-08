
import torch.nn as nn
from torchvision import models

model_name = 'resnet101'

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.name = 'resnet101'
        self.model = getattr(models, self.name)(pretrained = True)
        self.model.fc = nn.Linear(2048, 50)
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    m = Classifier()
    print(m)

