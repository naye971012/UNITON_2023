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
        "base_transform" : self.base_transform,
        "hard_transform" : self.hard_transform,
        "mask_transform" : self.mask_transform,
        "mixed_transform" : self.mixed_transform,
        "hard_transform_plus" : self.hard_transform_plus
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
        
        self.hard_transform = A.Compose([

            A.OneOf([
                A.HorizontalFlip(p=0.5),  # 수평 뒤집기
                A.Rotate(limit=10, p=0.5),                   # -10도에서 10도 사이 랜덤 회전
            ], p=0.75),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            A.RandomBrightnessContrast(p=0.15),  # 랜덤한 밝기와 대비 조절
            A.RandomBrightness(p=0.15),                       # 랜덤 밝기 조절
            A.HueSaturationValue(p=0.15),                    # 색조, 채도, 명도 조절
            A.RandomGamma(p=0.15),                           # 랜덤 감마 조절
            A.GaussNoise(var_limit=(0.0, 25.0), p=0.15),     # 가우시안 노이즈 추가
            A.Sharpen(p=0.15),                               # 이미지 선명도 조절

            A.OneOf([
                A.RandomCrop(height=334, width=334, p=0.7),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=384, width=384, p=0.8),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=448, width=448, p=0.9),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=496, width=496, p=1),  # 무작위 크롭 (정사각형으로 crop)
            ], p=0.3),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            
            A.Resize(*self.resize),
            A.Normalize(),
            ToTensorV2()
        ])

        self.hard_transform_plus = A.Compose([
            A.Resize(*self.resize),
            
            A.OneOf([
                A.HorizontalFlip(p=0.5),  # 수평 뒤집기
                A.Rotate(limit=10, p=0.5),                   # -10도에서 10도 사이 랜덤 회전
            ], p=0.75),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            A.RandomBrightnessContrast(p=0.15),  # 랜덤한 밝기와 대비 조절
            A.RandomBrightness(p=0.15),                       # 랜덤 밝기 조절
            A.HueSaturationValue(p=0.15),                    # 색조, 채도, 명도 조절
            A.RandomGamma(p=0.15),                           # 랜덤 감마 조절
            A.GaussNoise(var_limit=(0.0, 25.0), p=0.15),     # 가우시안 노이즈 추가
            A.Sharpen(p=0.15),                               # 이미지 선명도 조절
            
            A.OneOf([
                A.RandomCrop(height=334, width=334, p=0.7),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=384, width=384, p=0.8),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=448, width=448, p=0.9),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=496, width=496, p=1),  # 무작위 크롭 (정사각형으로 crop)
            ], p=0.3),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=384,width=384, ignore_values=[0,1,2,3,4,7,8,9],p=0.3),
            
            A.Normalize(),
            ToTensorV2()
        ])


        self.mask_transform = A.Compose([

            A.OneOf([
                A.HorizontalFlip(p=0.5),  # 수평 뒤집기
                A.Rotate(limit=10, p=0.5),                   # -10도에서 10도 사이 랜덤 회전
            ], p=0.75),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            A.GridDropout(holes_number_x=6, holes_number_y=6, p=0.75),

            A.OneOf([
                A.RandomCrop(height=334, width=334, p=0.7),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=384, width=384, p=0.8),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=448, width=448, p=0.9),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=496, width=496, p=1),  # 무작위 크롭 (정사각형으로 crop)
            ], p=0.3),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            
            A.Resize(*self.resize),
            A.Normalize(),
            ToTensorV2()
        ])
        
        
        self.mixed_transform = A.Compose([

            A.OneOf([
                A.HorizontalFlip(p=0.5),  # 수평 뒤집기
                A.Rotate(limit=10, p=0.5),                   # -10도에서 10도 사이 랜덤 회전
            ], p=0.75),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            A.RandomBrightnessContrast(p=0.15),  # 랜덤한 밝기와 대비 조절
            A.RandomBrightness(p=0.15),                       # 랜덤 밝기 조절
            A.HueSaturationValue(p=0.15),                    # 색조, 채도, 명도 조절
            A.RandomGamma(p=0.15),                           # 랜덤 감마 조절
            A.GaussNoise(var_limit=(0.0, 10.0), p=0.15),     # 가우시안 노이즈 추가

            A.GridDropout(holes_number_x=6, holes_number_y=6, p=0.75),

            A.OneOf([
                A.RandomCrop(height=334, width=334, p=0.7),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=384, width=384, p=0.8),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=448, width=448, p=0.9),  # 무작위 크롭 (정사각형으로 crop)
                A.RandomCrop(height=496, width=496, p=1),  # 무작위 크롭 (정사각형으로 crop)
            ], p=0.3),  # OneOf로 랜덤 선택, p=1.0은 항상 적용하도록 함
            
            
            A.Resize(*self.resize),
            A.Normalize(),
            ToTensorV2()
        ])