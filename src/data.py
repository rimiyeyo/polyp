import os
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(PROJECT_ROOT) # /mnt/d/proj/polyp

DATASET_PATH = os.path.join(f"{PROJECT_ROOT}/data")
checkpoint_dir =os.path.join(DATASET_PATH,"train_ckpt")

img_dir = os.path.join(DATASET_PATH, "train")
label_dir = os.path.join(DATASET_PATH, "train_label")

checkpoint_dir =os.path.join(DATASET_PATH,"train_ckpt")

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

x_train_filenames = [
    os.path.join(img_dir, filename) for filename in os.listdir(img_dir)
]
x_train_filenames.sort()
y_train_filenames = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]
y_train_filenames.sort()

x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames = \
                    train_test_split(x_train_filenames, y_train_filenames, test_size=0.2)
