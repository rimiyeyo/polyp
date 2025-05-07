# 필요 라이브러리 로딩
from __future__ import absolute_import, division, print_function

import os

import matplotlib as mpl
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from loader import SegDataset

DATASET_PATH = os.path.join("/mnt/d/proj/vivapoly/data")

img_dir = os.path.join(DATASET_PATH, "train")
label_dir = os.path.join(DATASET_PATH, "train_label")

x_train_filenames = [
    os.path.join(img_dir, filename) for filename in os.listdir(img_dir)
]
x_train_filenames.sort()
y_train_filenames = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]
y_train_filenames.sort()

x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames = \
                    train_test_split(x_train_filenames, y_train_filenames, test_size=0.2)

num_train_examples = len(x_train_filenames)
num_test_examples = len(x_test_filenames)

print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))

train_dataset = SegDataset(x_train_filenames, y_train_filenames)
test_dataset = SegDataset(x_test_filenames, y_test_filenames, is_train=False)
train_dataloader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              pin_memory=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1,
                             pin_memory=True)
