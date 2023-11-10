import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

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

def get_loss(name: str):
    pass