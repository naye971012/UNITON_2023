import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler

from lr_scheduler import *
from lovasz_losses import lovasz_softmax

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    elif name == "abl_ce_iou":
        criterion = ABL_CE_IOU()
    elif name == "abl_ce_iou_weight": #weight per class ratio
        criterion = ABL_CE_IOU(weight=torch.Tensor([1, 1.5, 1, 5.0, 2.5, 20.0, 20.0, 4.0, 4.0, 1.0]).to(DEVICE))
    else:
        raise ValueError(f"Unsupported criterion for semantic segmentation: {name}")

    return criterion


class ABL_CE_IOU(nn.Module):
    def __init__(self,weight=None, label_smooth=0.2):
        super(ABL_CE_IOU,self).__init__()
        if weight!=None:
            label_smooth = 0
        self.abl_loss = ABL(weight=weight,label_smoothing=label_smooth)
        self.focal_loss = FocalLoss()
    def forward(self,logits, targets):
        x =self.abl_loss.forward(logits, targets) + self.focal_loss.forward(logits, targets) + lovasz_softmax(F.softmax(logits,dim=1),targets)
        return x

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
    target_one_hot = torch.zeros(target.size(0), num_classes, target.size(1), target.size(2)).to(DEVICE)
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


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
from label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from torchvision import transforms
from functools import partial
from operator import itemgetter

# Tools
def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

# Active Boundary Loss
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_ud = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_lr = gt[:,:,1:]-gt[:,:,:-1]
        gt_ud = torch.nn.functional.pad(gt_ud, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_lr = torch.nn.functional.pad(gt_lr, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).cuda() # 8k
        kl_maps = torch.zeros((0,len(x))).cuda() # 8k

        kl_center = logits[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, logits, target):
        eps = 1e-10
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt) # careful
        
        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss



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