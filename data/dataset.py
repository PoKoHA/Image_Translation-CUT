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

        # grayscle 에서 rgb로 전환 (아니면 .convert('RGB')도 가능)
        if image_A.mode != "RGB": # 즉 image_A가 흑백이라면
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)
        modified_opt = copyconf(self.args, load_size=self.args.crop_size) # load size =258
        transform = get_transform(modified_opt)

        image_A = transform(image_A)
        image_B = transform(image_B)

        return {"A": image_A, "B": image_B}

    def __len__(self): # len 실행 시 실제로 __len__메서드 호출
        return max(len(self.files_A), len(self.files_B))

    # A와B 파일을 따로따로 return
def to_rgb(image):
    # Image.new(mode, szie, color) : 새로운 이미지 생성
    rgb_image = Image.new("RGB", image.size)
    # .paste(추가할 이미지, 붙일 위치(가로, 세로)): 이미지붙이기
    rgb_image.paste(image)
    return rgb_image