import os
import argparse
import glob
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable, Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DSN(nn.Module):
    def __init__(self, code_size=512, n_class=10):
        super(DSN, self).__init__()
        self.code_size = code_size

        ##########################################
        # private source encoder
        ##########################################

        self.source_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.source_encoder_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True),
        )

        #########################################
        # private target encoder
        #########################################

        self.target_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.target_encoder_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True),
        )

        ################################
        # shared encoder (dann_mnist)
        ################################

        self.shared_encoder_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.shared_encoder_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True)
        )

        # classify 10 numbers
        self.shared_encoder_pred_class = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, n_class)
        )

        # classify two domain
        self.shared_encoder_pred_domain = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2)
        )

        ######################################
        # shared decoder (small decoder)
        ######################################

        self.shared_decoder_fc = nn.Sequential(
            nn.Linear(code_size, 50 * 4 * 4),
            nn.ReLU(True)
        )

        self.shared_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(50, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, bias=False),
            nn.Tanh()
        )
        

    def encode(self, input_data):
        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 50 * 4 * 4)
        shared_code = self.shared_encoder_fc(shared_feat)
        return shared_code
    

    def forward(self, input_data, mode, rec_scheme='all', p=0.0):
        if mode == 'source':
            private_feat = self.source_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 50 * 4 * 4)
            private_code = self.source_encoder_fc(private_feat)

        elif mode == 'target':
            private_feat = self.target_encoder_conv(input_data)
            private_feat = private_feat.view(-1, 50 * 4 * 4)
            private_code = self.target_encoder_fc(private_feat)

        shared_feat = self.shared_encoder_conv(input_data)
        shared_feat = shared_feat.view(-1, 50 * 4 * 4)
        shared_code = self.shared_encoder_fc(shared_feat)

        class_label = self.shared_encoder_pred_class(shared_code)

        reversed_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)

        if rec_scheme == 'share':
            union_code = shared_code
        elif rec_scheme == 'all':
            union_code = private_code + shared_code
        elif rec_scheme == 'private':
            union_code = private_code

        recon = self.shared_decoder_fc(union_code)
        recon = recon.view(-1, 50, 4, 4)
        recon = self.shared_decoder_conv(recon)
        return class_label, domain_label, private_code, shared_code, recon

def main(config, model_path):
    print('loading from', model_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    model = DSN().cuda()

    state = torch.load(model_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.data_dir, '*.png'))
    filenames = sorted(filenames)

    out_filename = config.out_csv
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_name,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn).convert('RGB')
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.cuda()
                output, _, _, _, _ = model(data, mode= 'target')
                pred = output.max(1, keepdim=True)[1] 
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='../hw3_data/digits/svhn/test')
    parser.add_argument('--out_csv', type=str, default='test.csv')
    parser.add_argument('--target', default='target', type=str)
    config = parser.parse_args()

    model_name = { "usps":"mnistm-usps.pth", "mnistm":"svhn-mnistm.pth", "svhn": "usps-svhn.pth"}

    main(config, model_name[config.target])