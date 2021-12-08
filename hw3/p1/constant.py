from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(), 
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


train_dir = "/tmp2/b08902134/hw3-neverloses87/hw3_data/p1_data/train"
valid_dir = "/tmp2/b08902134/hw3-neverloses87/hw3_data/p1_data/val"
