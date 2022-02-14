# import torch

# # Model
# model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or yolov3-spp, yolov3-tiny, custom

# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# results.show()
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import os
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# a = np.load('./skull/train/H1_00000000_00000181/H1_00000000_00000181_00000001.npy')

# print(a.shape)
# plt.imshow(a)
# plt.show()
# H1_00000569_00000187_00000028.png
# filepath = '../skull/images/H1_00000569_00000187/'
# filenames = os.listdir(filepath)
# path = Path(filepath)
# files = list(path.rglob('*.*'))

fn = './skull/images/H1_00000607_00000221/H1_00000607_00000221_00000008.npy'

td = np.load(os.path.join(fn))
data = np.stack([td, td, td])
data = np.transpose(data, (1, 2, 0))
mmax = np.amax(data)
mmin = np.amin(data)
data = ((data - mmin) / (mmax - mmin) * 255).astype(np.uint8)
# data[356][83] = 255
# data[349][89] = 255
plt.axis("off")
plt.imshow(data, cmap=cm.Greys_r,animated=True)
plt.show()
#plt.savefig("H1_00000607_00000221_00000008.png",bbox_inches='tight')


# im = Image.fromarray(data, 'RGB')
# im.save("H1_00000569_00000187_00000028.png")

# img = ax.imshow(data)
# ax.imshow(np.array(attentions[i]), cmap='gray', alpha=0.7, extent=img.get_extent())
# plt.savefig(os.path.join(output_path, fn.split('.')[0]))

print("")
