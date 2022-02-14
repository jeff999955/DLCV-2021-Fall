import os
from shutil import copyfile

images_path = '../skull/test'
target_path = '../skull/test_all_images'

folders = os.listdir(images_path)
# print(folders)

for folder in folders:
    for file in os.listdir(os.path.join(images_path, folder)):
        copyfile(os.path.join(images_path, folder, file), 
                 os.path.join(target_path, file))