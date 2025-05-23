import random

import albumentations as A
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import functional as F

image_size = 256
img_shape = (image_size, image_size, 3)
batch_size = 1
max_epochs = 50

def _process_pathnames(fname, label_path):

    img = Image.open(fname).convert("RGB")  # PNG를 RGB로 불러오기
    label_img = Image.open(label_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    img = transform(img)  # 3. 256, 256
    label_img = transform(label_img)
    return img, label_img


def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """Perform horizontal or vertical shift on both image and label (PIL format)."""
    if width_shift_range or height_shift_range:
        w, h = output_img.size  # (width, height)

        dx = 0
        dy = 0
        if width_shift_range:
            max_dx = int(width_shift_range * w)
            dx = random.randint(-max_dx, max_dx)
        if height_shift_range:
            max_dy = int(height_shift_range * h)
            dy = random.randint(-max_dy, max_dy)

        output_img = F.affine(output_img, angle=0, translate=(dx, dy), scale=1.0, shear=0)
        label_img = F.affine(label_img, angle=0, translate=(dx, dy), scale=1.0, shear=0)

    return output_img, label_img


def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        transform = A.Compose([
          A.HorizontalFlip(p=0.5)
        ])
        aug = transform(image=tr_img, mask=label_img)
    return aug['image'], aug['mask']


def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0.,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=True,  # Random left right flip,
             width_shift_range=0.05,  # Randomly translate the image horizontally
             height_shift_range=0.05):
    if resize is not None:
        resize_tf = transforms.Resize(resize)
        img = resize_tf(img)
        label_img = resize_tf(label_img)
    if hue_delta:
        hue_factor = random.uniform(-hue_delta, hue_delta)
        img = F.adjust_hue(img, hue_factor)
        
        img, label_img = flip_img(horizontal_flip, img, label_img)
        img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
    return img, label_img

def get_cosine_scheduler(optimizer, max_epochs, min_lr=0, last_epoch=-1):
    # Cosine annealing from initial lr to min_lr over max_epochs
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=min_lr,
        last_epoch=last_epoch
    )
