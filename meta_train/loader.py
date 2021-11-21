from PIL import ImageFilter, Image
from torch.utils.data import Dataset
import torchvision
import sys
import cv2
import random
import os
import numpy as np

class OfflineLoader(Dataset):
    def __init__(self, args):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if args.aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            transform = [
                torchvision.transforms.RandomResizedCrop(224, scale = (0.2, 1.)),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p = 0.8),
                torchvision.transforms.RandomGrayscale(p = 0.2),
                torchvision.transforms.RandomApply([GaussianBlur([.1, 2.])], p = 0.5),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            transform = [
                torchvision.transforms.RandomResizedCrop(224, scale = (0.2, 1.)),
                torchvision.transforms.RandomGrayscale(p = 0.2),
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
        self.transform = TwoCropsTransform(torchvision.transforms.Compose(transform))

        self.img_file = []
        for cate_file in os.listdir(args.data_path):
            if os.path.isdir(os.path.join(args.data_path, cate_file)) is False:
                continue
            for img_name in os.listdir(os.path.join(args.data_path, cate_file)):
                self.img_file.append(os.path.join(args.data_path, cate_file, img_name))
        random.shuffle(self.img_file)
        data_len = int(len(self.img_file) * args.data_rate)
        self.img_file = self.img_file[0 : data_len]

    def __len__(self):
        return len(self.img_file)

    def __getitem__(self, item):
        img_path = self.img_file[item]
        img = Image.open(img_path)
        img = img.convert('RGB')
        q, k = self.transform(img)
        return q, k
    

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def check_img(path):
    img = cv2.imread(path)
    if img is None:
        return False
    else:
        return True

