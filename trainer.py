from PIL import Image
import re
from tqdm import tqdm
import torch
import os
import numpy as np
from glob import glob
import zipfile

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(configs, model, train_lodaer, vali_loader):
    pass

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