import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler

from lr_scheduler import *

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
        #criterion = DiceLoss()
        pass
    elif name == "focal":
        criterion = FocalLoss()
    elif name == "mixed":
        criterion = MixedLoss()
    else:
        raise ValueError(f"Unsupported criterion for semantic segmentation: {name}")

    return criterion

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

def one_hot_encode(target, num_classes):
    target_one_hot = torch.zeros(target.size(0), num_classes, target.size(1), target.size(2))
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    return target_one_hot

def dice_loss(input, target):
    input = torch.softmax(input, dim=1)
    smooth = 1.0
    iflat = input.view(-1)
    
    target_one_hot = one_hot_encode(target, num_classes=input.size(1))
    tflat = target_one_hot.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

class MixedLoss(nn.Module):
    def __init__(self, alpha=10):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss()

    def forward(self, input, target):
        """

        Args:
            input (_type_): logit [# of batch, class, 512, 512]
            target (_type_): [# of batch, 512, 512]

        Returns:
            _type_: loss
        """
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


def get_scheduler(name:str, optimizer):
    
    name = name.lower()
    
    if name == "steplr":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    elif name == "reducelronplateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    elif name == "sgdr":
        """
        먼저 warm up을 위하여 optimizer에 입력되는 learning rate = 0 또는 0에 가까운 아주 작은 값을 입력합니다.
        위 코드의 스케쥴러에서는 T_0, T_mult, eta_max 외에 T_up, gamma 값을 가집니다.
        T_0, T_mult의 사용법은 pytorch 공식 CosineAnnealingWarmUpRestarts와 동일합니다. 
        T_0는 최초 주기값 입니다. T_mult는 주기가 반복되면서 최초 주기값에 비해 얼만큼 주기를 늘려나갈 것인지 스케일 값에 해당합니다
        eta_max는 learning rate의 최댓값을 뜻합니다. 
        T_up은 Warm up 시 필요한 epoch 수를 지정하며 일반적으로 짧은 epoch 수를 지정합니다.
        gamma는 주기가 반복될수록 eta_max 곱해지는 스케일값 입니다
        """
        scheduler = CosineAnnealingWarmUpRestarts(optimizer,
                                                  T_0=10,
                                                  T_mult=2,
                                                  eta_max=0.1, 
                                                  T_up=10, 
                                                  gamma=0.5)
        
    return scheduler

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