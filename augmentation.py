import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

class transformation_class:
    def __init__(self, resize:Tuple[int,int]) -> None:
        self.resize = resize
        
        self.init_transforms()
        
        self.str2transform = {
        "test_transform" : self.test_transform,
        "base_transform" : self.base_transform
        }
            
    def __call__(self, name:str):
        if name in self.str2transform.keys():
            return self.str2transform[name]
        else:
            print(f"wrong transform name: {name}, return base_transform")
            return self.str2transform["base_transform"]
        
    def init_transforms(self):
        
        self.test_transform = A.Compose([
                    A.Resize(*self.resize),
                    A.Normalize(),
                    ToTensorV2()
        ])

        self.base_transform = A.Compose([
                    A.Resize(*self.resize),
                    A.Normalize(),
                    ToTensorV2()
        ])