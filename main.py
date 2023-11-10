import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
import logging
import sys

from models import get_model
from dataloader import get_loaders
from trainer import train, test

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

def main(configs):

    logger.info("*** data loading ***")
    
    train_loader , val_loader, test_loader = get_loaders(configs)
    
    logger.info(f"Train dataloader size: {len(train_loader)}")
    logger.info(f"Validation datalodaer size: {len(val_loader)}")
    logger.info(f"Test datalodaer size: {len(test_loader)}")


    logger.info("*** model loading***")
    
    init_params = {
        "encoder_name": configs.encoder_name, 
        "encoder_weights": configs.encoder_weights, 
        "classes": configs.classes, 
        "activation": configs.activation
    } 
    model = get_model(configs, architecture=configs.architecture, init_params=init_params)    
    model.to(DEVICE)
    
    if configs.IS_TRAIN:
        logger.info("*** train start ***")
        train(configs, model, train_loader, val_loader)
    
    if configs.IS_TEST:
        logger.info("*** inference start ***")
        test(configs, model, test_loader)
    

if __name__=="__main__":

    configs = { 
        "IS_TRAIN" : True,
        "IS_TEST" : False,
        "SAVE_MODEL" : True,
        
        'SAVE_DIR' : './predicted_masks',
        'DATA_PATH' : 'segmentation_basis/data',
        'VALI_SIZE' : 0.2,
        "SEED" : 42,
        "RESIZE" : (224,224),
        "NUM_WORKERS" : 1,
        
        "epoch" : 5,
        "batch_size" : 2,
        "accumulation_step" : 2,
        "train_transform" : "base_transform",

        "optimizer" : "adamw", #(optimizer in torch.optim.*)
        "loss" : "focal",
        "lr" : 0.001,
        
        "encoder_name": 'resnet50', 
        "encoder_weights": 'imagenet', 
        "classes": 10, 
        "activation": None,
        "architecture": 'Unet'
    }
    name = f"{configs['encoder_name']}-{configs['architecture']}-{configs['train_transform']}"

    wandb.init(
            project="UNITON_segmentation",
            name=name,
            config=configs
    )

    class CONFIGS:
        """
        DICT 변수를 configs.LR 등의 방법으로 접근하기 위함
        """
        def __init__(self):
            self.configs = configs
        def __getattr__(self, name):
            if name in self.configs:
                return self.configs[name]
            else:
                raise AttributeError(f"'CONFIGS' object has no attribute '{name}'")
    cfg = CONFIGS()
    
    main(cfg)