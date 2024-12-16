import numpy as np
import torch
import cv2
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.utils import save_image
from stackimage import stackImages
from unet import Unet
from skimage import img_as_ubyte
from dataset import LiverDataset
import matplotlib.pyplot as plt
import seaborn as sns

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料增強
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.ToTensor()

def compute_pixel_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    計算像素級混淆矩陣
    :param y_true: numpy array, shape = [H, W]
    :param y_pred: numpy array, shape = [H, W]
    :param num_classes: int, 類別數
    :return: 混淆矩陣
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    return cm

def plot_pixel_confusion_matrix(cm, labels , title="Pixel-Level Confusion Matrix", save_path=None):
    """
    繪製像素級混淆矩陣
    :param cm: 混淆矩陣
    :param classes: 類別標籤
    :param title: 圖表標題
    :param save_path: 儲存路徑
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def test_with_confusion_matrix(args):
    model = Unet(3, 1).to(device)
    model.load_state_dict(torch.load(args.ckp, map_location=device))
    model.eval()
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)

    # Ensure the directory for confusion matrices exists
    output_dir = "confusion_matrices"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global_conf_matrix = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloaders):
            inputs = x.to(device)
            targets = y_true.to(device)
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            # Convert to numpy arrays
            targets_np = targets.squeeze().cpu().numpy()
            preds_np = preds.squeeze().cpu().numpy()

            # Compute confusion matrix
            conf_matrix = compute_pixel_confusion_matrix(targets_np, preds_np, num_classes=2)
            global_conf_matrix += conf_matrix

            # Save confusion matrix for each image
            plot_pixel_confusion_matrix(
                conf_matrix,
                labels=["Background", "Liver"],
                title=f"Image {idx} Confusion Matrix",
                save_path=f"{output_dir}/image_{idx}_cm.png"
            )

    # Save global confusion matrix
    plot_pixel_confusion_matrix(
        global_conf_matrix,
        labels=["Background", "Liver"],
        title="Global Confusion Matrix",
        save_path=f"{output_dir}/global_cm.png"
    )

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    # 設定權重路徑
    args.ckp = "weights_49.pth"

    # 測試並產生混淆矩陣
    test_with_confusion_matrix(args)
