import glob
import random
import os
from PIL import Image

import torch.utils.data
from utils.utils import *
from data.transform import *


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode="train", args=None):
        self.unaligned = unaligned
        self.args = args

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned: # unpaired #todo 나중에 조금 다시 조작
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        modified_opt = copyconf(self.args, load_size=self.args.crop_size) # load size =258
        transform = get_transform(modified_opt)

        image_A = transform(image_A)
        image_B = transform(image_B)

        return {"A": image_A, "B": image_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image