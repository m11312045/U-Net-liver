import numpy as np
import torch
import cv2
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from torchvision.utils import save_image
from scipy.spatial.distance import directed_hausdorff
from skimage import img_as_ubyte
from unet import Unet
from dataset import LiverDataset
from stackimage import stackImages
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast
import csv

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

y_transforms = transforms.ToTensor()

# Dice Coefficient
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    if union == 0:
        return 1.0
    return (2.0 * intersection) / union

# IoU Score
def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:
        return 1.0
    return intersection / union

# Hausdorff Distance 95 (HD95)
def hausdorff_distance_95(y_true, y_pred):
    y_true_points = np.argwhere(y_true > 0.5)
    y_pred_points = np.argwhere(y_pred > 0.5)
    
    if len(y_true_points) == 0 or len(y_pred_points) == 0:
        return np.inf  # 空集情况

    # Calculate directed Hausdorff distances
    d_true_to_pred = directed_hausdorff(y_true_points, y_pred_points)[0]
    d_pred_to_true = directed_hausdorff(y_pred_points, y_true_points)[0]
    return max(d_true_to_pred, d_pred_to_true)

# Test Function
def test_1():
    model = Unet(3, 1).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location=device))
    model.eval()

    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    
    total_dice, total_iou, total_hd95 = 0, 0, 0
    count = 0
    
    # 打開 CSV 檔案以寫入
    with open('test_results.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Index", "Dice", "IoU", "HD95"])  # 標題
        
        with torch.no_grad():
            for i, (x, y_true) in enumerate(dataloaders):
                x = x.to(device)
                y_true = y_true.to(device)

                with autocast():  # 使用混合精度
                    y_pred = model(x)
                    y_pred = torch.sigmoid(y_pred)
                    y_pred_bin = (y_pred > 0.5).float()

                y_true_np = y_true.squeeze().cpu().numpy()
                y_pred_bin_np = y_pred_bin.squeeze().cpu().numpy()

                # 計算指標
                dice = dice_coefficient(y_true_np, y_pred_bin_np)
                iou = iou_score(y_true_np, y_pred_bin_np)
                hd95 = hausdorff_distance_95(y_true_np, y_pred_bin_np)

                total_dice += dice
                total_iou += iou
                total_hd95 += hd95
                count += 1

                # 寫入 CSV
                writer.writerow([i, f"{dice:.4f}", f"{iou:.4f}", f"{hd95:.2f}"])
                print(f"Image {i}: Dice={dice:.4f}, IoU={iou:.4f}, HD95={hd95:.2f}")

        writer.writerow(["Overall", f"{total_dice/count:.4f}", f"{total_iou/count:.4f}", f"{total_hd95/count:.2f}"])
        print(f"Overall : Dice={total_dice/count:.4f}, IoU={total_iou/count:.4f}, HD95={total_hd95/count:.2f}")
    print("Results have been saved to 'test_results.csv'.")

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=1)  # 可根據GPU記憶體調整
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()
    
    args.ckp = "weights_49.pth"  # 指定預訓練模型權重路徑
    test_1()
