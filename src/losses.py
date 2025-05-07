import torch
import torch.nn.functional as F


def dice_coeff(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-10) -> torch.Tensor:
    """
    Compute the Dice coefficient between two tensors.

    Args:
        y_true (torch.Tensor): Ground truth tensor, any shape.
        y_pred (torch.Tensor): Prediction tensor, same shape as y_true.
        smooth (float): Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: Dice coefficient (scalar).
    """
    # Flatten tensors
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)

    # Intersection and sums
    intersection = (y_true_f * y_pred_f).sum()
    sum_true_sq = (y_true_f * y_true_f).sum()
    sum_pred_sq = (y_pred_f * y_pred_f).sum()

    # Dice calculation
    dice = (2.0 * intersection + smooth) / (sum_true_sq + sum_pred_sq + smooth)
    return dice

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
  
  
def bce_dice_loss(y_true: torch.Tensor,
                  y_pred: torch.Tensor,
                  smooth: float = 1e-10) -> torch.Tensor:
    """
    Combined Binary Cross-Entropy and Dice loss.

    Args:
        y_true (torch.Tensor): Ground truth tensor, same shape as y_pred.
        y_pred (torch.Tensor): Predicted tensor with values in [0,1].
        smooth (float): Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Binary Cross-Entropy (mean reduction by default)
    bce = F.binary_cross_entropy(y_pred, y_true)

    # Dice Loss (assumes dice_loss is defined as above)
    dice = dice_loss(y_true, y_pred)

    return bce + dice
