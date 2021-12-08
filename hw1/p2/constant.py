from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

n_classes = 50
chkpt_dir = '/tmp2/b08902134/hw1_checkpoint/'
train_dir = '/tmp2/b08902134/testdata/hw1_data/p2_data/train/'
valid_dir = '/tmp2/b08902134/testdata/hw1_data/p2_data/validation/'
