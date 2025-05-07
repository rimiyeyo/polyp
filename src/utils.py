import torch


def get_cosine_scheduler(optimizer, max_epochs, min_lr=0, last_epoch=-1):
    # Cosine annealing from initial lr to min_lr over max_epochs
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=min_lr,
        last_epoch=last_epoch
    )
