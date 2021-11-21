import os
import sys
import pdb
import csv
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Only support meta test mode.
class MetaOmniglot(Dataset):
    def __init__(self, args, partition = 'train', fix_seed = True):
        super().__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.n_test_runs = args.n_test_runs
        self.fix_seed = fix_seed

        self.category_list = []
        if partition == 'train':
            parent1_list = os.listdir(os.path.join(self.data_root, 'images_background'))
            for parent1_file in sorted(parent1_list):
                parent2_list = os.listdir(os.path.join(self.data_root, 'images_background', parent1_file))
                for parent2_file in sorted(parent2_list):
                    self.category_list.append(os.path.join(self.data_root, 'images_background', parent1_file, parent2_file))
        elif partition == 'test':
            parent1_list = os.listdir(os.path.join(self.data_root, 'images_evaluation'))
            for parent1_file in sorted(parent1_list):
                parent2_list = os.listdir(os.path.join(self.data_root, 'images_evaluation', parent1_file))
                for parent2_file in sorted(parent2_list):
                    self.category_list.append(os.path.join(self.data_root, 'images_evaluation', parent1_file, parent2_file))
        else:
            raise Exception("partition must be 'train' or 'test'.")

        self.classes = len(self.category_list)

    def __len__(self):
        return self.n_test_runs

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)

        cls_sampled = np.random.choice(self.category_list, self.n_ways, False)  # Sample the queired categories.
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []

        for idx, cls in enumerate(cls_sampled):
            img_name_list = sorted(os.listdir(cls))

            support_index = np.random.choice(range(len(img_name_list)), self.n_shots, False)  # Sample support images in a specific category.
            support_xs.append(self.load_imgs(cls, img_name_list, support_index))
            support_ys = support_ys + self.n_shots * [idx]

            query_index = np.setxor1d(range(len(img_name_list)), support_index)
            query_index = np.random.choice(query_index, self.n_queries, False)
            query_xs.append(self.load_imgs(cls, img_name_list, query_index))
            query_ys = query_ys + self.n_queries * [idx]

        support_xs = np.concatenate(support_xs, axis = 0).astype(np.float32)
        support_ys = np.array(support_ys).astype(np.int64)
        query_xs = np.concatenate(query_xs, axis = 0).astype(np.float32)
        query_ys = np.array(query_ys).astype(np.int64)
        
        support_xs = support_xs.transpose(0, 3, 1, 2)   # -> (B, C, H, W)
        query_xs = query_xs.transpose(0, 3, 1, 2)   # -> (B, C, H, W)
        support_xs = torch.tensor(support_xs)
        query_xs = torch.tensor(query_xs)
       
        return support_xs, support_ys, query_xs, query_ys
        

    def load_imgs(self, file_path, name_list, index_list):
        img_list = []
        for index in index_list:
            img = Image.open(os.path.join(file_path, name_list[index]))
            img = np.array(img)
            img = img.reshape(1, img.shape[0], img.shape[1], 1)
            img = np.tile(img, (1, 1, 1, 3))
            img_list.append(img)
        img_batch = np.concatenate(img_list, axis = 0)
        return img_batch
        

if __name__ == '__main__':
    class args():
        def __init__(self):
            self.data_root = '/opt/data/private/zli/data/rfs/Omniglot'
            self.n_ways = 5
            self.n_shots = 5
            self.n_queries = 15
            self.n_test_runs = 10

    para = args()
    a = MetaOmniglot(para, 'test')
    support_xs, support_ys, query_xs, query_ys = a.__getitem__(0)
    pdb.set_trace()

        

