## Uni-DTHON (유니드톤) 2023 데이터톤 트랙 16팀
- Semantic Segmentation Task
- 특별상 수상 (Public Learboard 1st, Final Leaderboard 6st)

## Dataset Description
- Semantic Semgentation of indoor images
- distinguish 12 different classes
- dataset has class imblance problem & small object problem

![dataset_example](https://github.com/naye971012/UNITON_2023/assets/74105909/fc2c48ec-1a72-4212-a4df-ad352a76ac90)

## Used Method

### Baseline Model with SMP library 
- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- testing various models including Unet, DeepLab, MaNet, etc...
- select baseline model as DeepLabv3 + ResNet101 Backbone

### ABL(Active Boundary Loss) method 
- [active_boundary_loss](https://github.com/wangchi95/active-boundary-loss)
- dataset has very small object including light bulb
- using ABL method to detect small object's boundary correctly

 ![abl_loss_example2](https://github.com/naye971012/UNITON_2023/assets/74105909/62be0041-45a7-4f97-b714-ecbaba456611)

### Hard Augmentation Method
- hard augmentation including following method
- 1. cutout augmentation
- 2. CropNonEmptyMaskIfExits
- 3. Flip, Dropout, Color, etc...

 ![transform_example](https://github.com/naye971012/UNITON_2023/assets/74105909/77c982b1-1f75-4326-8529-a632d11c5836)

### Hyperparameter Tuning using WandB/Bayesian-Search
- hyperparameter tuning using Wandb sweep method

 ![wandb_sweep](https://github.com/naye971012/UNITON_2023/assets/74105909/7c52ee89-b96a-4921-ba12-a66ecd8b0115)

### Apply Test-Time-Augmentation(TTA) Method
- increase public IOU 0.003


## Final Prediction

 ![predict_example](https://github.com/naye971012/UNITON_2023/assets/74105909/71780ed2-0ade-4da7-990b-03ae433ad614)


### Public/Final Leaderboard
- public score 0.5986

![public_leaderboard](https://github.com/naye971012/UNITON_2023/assets/74105909/bf80c5fb-cd91-4ee1-9929-a709d79e1ef2)

- final score 0.6173

![final_leaderboard](https://github.com/naye971012/UNITON_2023/assets/74105909/586711a2-010f-4f91-b4d7-11d917a9a82e)