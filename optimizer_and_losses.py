import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn as nn

def get_loss(name: str):
    """
    지정된 loss 이름에 해당하는 PyTorch loss 함수를 반환합니다.

    Parameters:
        name (str): Loss 함수의 이름. "cross_entropy", "dice", "focal" 등이 될 수 있습니다.

    Returns:
        torch.nn.modules.loss._Loss: 지정된 loss 함수 객체.
    """
    name = name.lower()

    if name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif name == "dice":
        criterion = DiceLoss()
    elif name == "focal":
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unsupported criterion for semantic segmentation: {name}")

    return criterion

# Dice Loss 구현
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = (logits * targets).sum()
        union = logits.sum() + targets.sum() + self.smooth

        dice_score = (2.0 * intersection + self.smooth) / union
        dice_loss = 1.0 - dice_score

        return dice_loss

# Focal Loss 구현
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        cross_entropy = nn.CrossEntropyLoss()(logits, targets)
        pt = torch.exp(-cross_entropy)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy

        return focal_loss



def get_optim(name: str, params, lr):
    """
    지정된 optimizer 이름에 해당하는 PyTorch optimizer를 반환합니다.

    Parameters:
        name (str): Optimizer 이름. "SGD", "Adam" 등이 될 수 있습니다.
        parameters (iterable): 최적화 대상인 매개변수(iterable of dict 또는 torch.Tensor).
        lr (float, optional): 학습률. 기본값은 0.001입니다.

    Returns:
        torch.optim.Optimizer: 지정된 optimizer 객체.
    """
    name = name.lower()

    if name == "sgd":
        optimizer = optim.SGD(params, lr=lr)
    elif name == "adam":
        optimizer = optim.Adam(params, lr=lr)
    elif name == "rmsprop":
        optimizer = optim.RMSprop(params, lr=lr)
    elif name == "adagrad":
        optimizer = optim.Adagrad(params, lr=lr)
    elif name == "adadelta":
        optimizer = optim.Adadelta(params, lr=lr)
    elif name == "adamw":
        optimizer = optim.AdamW(params, lr=lr)
    elif name == "adamax":
        optimizer = optim.Adamax(params, lr=lr)
    elif name == "adagrad":
        optimizer = optim.Adagrad(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    return optimizer