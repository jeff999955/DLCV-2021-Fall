import torch
import torch.nn as nn
from torchvision import models

model_name = 'resnet101'


class Classifier(nn.Module):
    def __init__(self, n_classes = 1000, dropout = False):
        super(Classifier, self).__init__()
        self.name = 'resnet50'
        self.model = getattr(models, self.name)(pretrained=False)
        if dropout:
            self.model.fc = nn.Sequential(
                        nn.Linear(2048, n_classes),
                        nn.Dropout(),
                    )
        else:
            self.model.fc = nn.Linear(2048, n_classes)
    
    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def load_pretrained_weights(self, path, fc = False):
        psd = torch.load(path, map_location = 'cpu')
        if not fc:
            rm = []
            for name, param in psd.items():
                if 'fc' in name:
                    rm.append(name)
            for name in rm:
                del psd[name]
        else:
            for name, param in psd.items():
                if 'fc' in name:
                    print(name, param)

        self.model.load_state_dict(psd, strict = False)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    m = Classifier(65)
    # m.load_pretrained_weights('../hw4_data/pretrain_model_SL.pt')
    print(m)
