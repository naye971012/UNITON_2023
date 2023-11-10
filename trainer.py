from PIL import Image
import re
from tqdm import tqdm
import torch
import os
import numpy as np
from glob import glob
import zipfile
import wandb

from utils import calculate_iou, calculate_miou
from optimizer_and_losses import get_loss, get_optim
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(configs, model, train_lodaer, vali_loader):

    accumulation_step = configs.accumulation_step
    optimizer = get_optim(configs.optimizer, model.parameters(), configs.lr)
    criterion = get_loss(configs.loss)
    
    for epoch in range(configs.epoch):
        
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        
        for i, batch in enumerate(tqdm(train_lodaer, position=0, desc=f"epoch {epoch}")):
            images, masks = batch
            
            images = images.float().to(DEVICE)
            masks = masks.long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss = loss / accumulation_step
            loss.backward()
            if (i+1) % accumulation_step == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step    
                model.zero_grad()
            
            epoch_loss += loss.item()

            #save temp iou, acc
            iou_per_class, acc_per_class = calculate_iou(pred_masks=outputs,
                                                         true_masks=masks,
                                                         num_classes=configs.classes)
            mIOU = calculate_miou(iou_per_class)
            epoch_iou += mIOU
            
            tqdm.set_postfix({'Loss': epoch_loss / (i+1),
                              'mIOU': epoch_iou / (i+1) })

        validation(configs, model, vali_loader)


def validation(configs,model, vali_lodaer):

    criterion = get_loss(configs.loss)
     
    with torch.no_grad():
        
        model.eval()
        epoch_loss = 0.0
        epoch_iou = 0.0
        class_iou_values = []
        class_acc_values = []
        
        for i, batch in enumerate(tqdm(vali_lodaer, position=0, desc=f"validation epoch")):
            images, masks = batch
            
            images = images.float().to(DEVICE)
            masks = masks.long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            
            epoch_loss += loss.item()

            #save temp iou, acc
            iou_per_class, acc_per_class = calculate_iou(pred_masks=outputs,
                                                         true_masks=masks,
                                                         num_classes=configs.classes)
            class_iou_values.append(iou_per_class)
            class_acc_values.append(acc_per_class)
            mIOU = calculate_miou(iou_per_class)
            epoch_iou += mIOU
            
            tqdm.set_postfix({'Loss': epoch_loss / (i+1),
                              'mIOU': epoch_iou / (i+1) })

    # 전체 validation 데이터에 대한 클래스별 IoU를 평균냄
    mean_class_iou = torch.mean(torch.tensor(class_iou_values), dim=0).tolist()
    mean_class_acc = torch.mean(torch.tensor(class_acc_values), dim=0).tolist()
    # 평균 IoU 출력
    for class_idx, mean_iou in enumerate(mean_class_iou):
        print(f'Mean IoU for Class {class_idx}: {mean_iou}')
    # 평균 acc 출력
    for class_idx, mean_acc in enumerate(mean_class_acc):
        print(f'Mean ACC for Class {class_idx}: {mean_acc}')
    print("\n")


def test(configs, model, test_loader):

    # 예측 결과를 저장할 경로를 생성합니다.
    save_directory = configs.SAVE_DIR
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 테스트를 수행합니다.
    model.eval()
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader, position=0, leave=True, desc='Prediction')):

            # 가져온 데이터를 장치에 할당합니다.
            images = images.float().to(DEVICE)

            # 모델의 출력값을 계산합니다.
            pred_masks = model(images)

            # argmax 연산을 통해 확률이 가장 높은 클래스를 예측값으로 선택합니다.
            pred_masks = torch.argmax(pred_masks, dim=1)

            for _, a_pred_mask in enumerate(pred_masks):
                # pred_mask를 PIL image로 변환합니다.
                pred_mask_image = Image.fromarray(np.uint8(a_pred_mask.cpu().numpy()))

                # 파일 이름에서 인덱스를 추출합니다.

                # 이미지를 저장합니다. 파일 이름을 추출된 인덱스로 설정합니다.
                pred_mask_image.save(os.path.join(save_directory, f"test_{str(i).zfill(4)}.png"))

    pred_files = sorted(glob(f'{save_directory}/*.png'))  # 파일을 숫자 순서대로 정렬합니다.

    # 압축을 수행합니다.
    with zipfile.ZipFile('sample_submission.zip', 'w') as zipf:
        for pred_file in pred_files:
            zipf.write(pred_file, os.path.basename(pred_file))