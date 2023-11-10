import torch

def calculate_iou(pred_masks, true_masks, num_classes):
    iou_values = []
    accuracy_values = []

    for class_idx in range(num_classes):
        pred_class = (pred_masks == class_idx)
        true_class = (true_masks == class_idx)

        intersection = torch.logical_and(pred_class, true_class).sum().item()
        union = torch.logical_or(pred_class, true_class).sum().item()

        iou = intersection / (union + 1e-8)  # 0으로 나누는 것을 방지하기 위해 작은 값을 더합니다.
        accuracy = (pred_class == true_class).sum().item() / true_class.numel()

        iou_values.append(iou)
        accuracy_values.append(accuracy)

    return iou_values, accuracy_values

def calculate_miou(iou_values):
    return sum(iou_values) / len(iou_values)