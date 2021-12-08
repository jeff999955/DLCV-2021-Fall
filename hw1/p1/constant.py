from torchvision import transforms

train_tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

test_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

n_classes = 50
chkpt_dir = '/tmp2/b08902134/hw1_sackpt/'
train_dir = '/tmp2/b08902134/testdata/hw1_data/p1_data/train_50/'
valid_dir = '/tmp2/b08902134/testdata/hw1_data/p1_data/val_50/'

