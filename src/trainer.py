from tqdm import tqdm

device = 'cuda'

def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                model: nn.Module,
                optimizer: torch.optim.Optimizer,
                max_epochs: int = 50):
    # Initialize scheduler
    scheduler = get_cosine_scheduler(optimizer, max_epochs)
    best_val_loss = float('inf')

    for epoch in tqdm(range(1, max_epochs + 1)):
        model.train()
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
            torch.save(model.state_dict(), 'best_unet.pth')
            
train_model(train_dataloader, test_dataloader, model, optimizer, max_epochs=50)
