import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PROJECT_ROOT
from loader import test_dataloader, train_dataloader
from losses import bce_dice_loss
from models import UNet
from utils import get_cosine_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                model: nn.Module,
                optimizer: torch.optim.Optimizer,
                max_epochs: int = 50):
    # Initialize scheduler
    scheduler = get_cosine_scheduler(optimizer, max_epochs)
    best_val_loss = float('inf')
    model = model.to(device)
    model.train()

    for epoch in tqdm(range(1, max_epochs + 1)):
        train_loss = 0.0
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks  = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = bce_dice_loss(masks, outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = bce_dice_loss(masks, outputs)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        # Step the scheduler after each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{max_epochs}]  "
              f"Train Loss: {train_loss:.4f}  "
              f"Val Loss: {val_loss:.4f}  "
              f"LR: {current_lr:.6f}")

        # Save best model on validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{PROJECT_ROOT}/data/train_ckpt/best_unet.pth")

if __name__ == '__main__':   
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(train_dataloader, test_dataloader, model, optimizer, max_epochs=50)
