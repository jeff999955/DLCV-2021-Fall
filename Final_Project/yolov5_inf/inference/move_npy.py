import os
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, help='')
parser.add_argument('--target_path', type=str, help='')
config = parser.parse_args()

images_path = config.images_path # '../skull/test'
target_path = config.target_path # '../skull/test_all_images'

folders = os.listdir(images_path)
print(folders)

for folder in folders:
    # os.mkdir(folder)
    for file in os.listdir(os.path.join(images_path, folder)):
        copyfile(os.path.join(images_path, folder, file), 
                 os.path.join(target_path, file))
