import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader

from models import get_model
from dataloader import get_loaders


def main(configs):

    train_loader , val_loader, test_loader = get_loaders(configs)

    model = get_model(configs)


if __name__=="__main__":
    
    configs = {
        'DATA_PATH' : '/content/datasets',
        'VALI_SIZE' : 0.2,
        "SEED" : 42,
        "RESIZE" : (512,512),
        "NUM_WORKERS" : 2,
        
        "batch_size" : 8,
        "train_transform" : "base_transform"
    }
    
    main(configs)