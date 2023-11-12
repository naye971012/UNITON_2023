import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .augmentation import transformation_class

class SegDataset(Dataset):
    def __init__(self, image_paths,
                 mask_paths=None,
                 resize=(512, 512), 
                 mode='train',
                 train_transform_name="base_transform"):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize = resize
        self.mode = mode
        self.train_transform , self.test_transform = self.get_transform(train_transform_name)

    def get_transform(self, train_transform_name:str):
        """
        get transformation from augmentation.py
        """
        transform_class = transformation_class(resize=self.resize)
        
        train_transform = transform_class(train_transform_name)
        test_transform = transform_class("test_transform")
        
        return train_transform, test_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode=='train':
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            augmented = self.train_transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']

        elif self.mode=='val': #validation은 test와 유사 환경이어야 함. test_transform적용
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            augmented = self.test_transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']

        else:
            return self.test_transform(image=image)['image']