import os

import matplotlib.pyplot as plt
import torch

from data import PROJECT_ROOT
from loader import test_dataloader
from models import UNet

batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = os.path.join(f"{PROJECT_ROOT}/data")
checkpoint_dir =os.path.join(DATASET_PATH,"train_ckpt")

model = UNet()
checkpoint = torch.load(f"{checkpoint_dir}/best_unet.pth", map_location=device)
model.load_state_dict(checkpoint)

if __name__ == '__main__':    
    model.eval()
    with torch.no_grad():
        for test_img, test_label in test_dataloader:
            outputs = model(test_img)
            outputs = outputs.cpu().numpy()  
            test_img = test_img.permute(0, 2, 3, 1)
            plt.subplot(1, 3, 1)
            plt.imshow(test_img[0, :, :, :])
            plt.title("Original Image")
            
            plt.subplot(1, 3, 2)
            plt.imshow(test_label[0, 0, :, :])
            plt.title("Ground Truth")
            
            plt.subplot(1, 3, 3)
            plt.imshow(outputs[0, 0, :, :])
            plt.title("Predicted Image")
            plt.show()
            break
