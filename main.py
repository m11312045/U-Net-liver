import numpy as np
import torch
import cv2
import argparse
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from torchvision.utils import save_image
from stackimage import stackImages
from unet import Unet
from skimage import img_as_ubyte
from dataset import LiverDataset
from sklearn.metrics import accuracy_score

# 是否使用cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把多個步驟整合到一起, channel=（channel-mean）/std, 因為是分別對三個通道處理
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

# mask只需要轉換為tensor
y_transforms = transforms.ToTensor()

# 參數解析器,用來解析從終端機讀取的命令
parse = argparse.ArgumentParser()


def train_model(model, criterion, optimizer, dataload, num_epochs=50):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    if union == 0:  # 避免全零導致誤判
        return 0.0
    return (2.0 * intersection) / union

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    if union == 0:  # 避免全零導致誤判
        return 0.0
    return intersection / union

# 訓練模型
def train():
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


# 顯示模型的輸出結果
def test_1():
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    imgs = []
    root = "data/val"
    n = len(os.listdir(root)) // 2
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        # mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append(img)
    i = 0
    total_accuracy = 0
    total_dice = 0
    total_iou = 0
    count = 0
    with torch.no_grad():
        # for x, _ in dataloaders:
        for x, y_true in dataloaders:
            y_pred = model(x)
            y_pred = torch.sigmoid(y_pred)  # 對模型輸出應用 sigmoid 激活
            y_pred_bin = (y_pred > 0.5).float()  # 閾值分割

            # 轉換為 numpy 格式
            y_true_np = y_true.squeeze().cpu().numpy()
            y_pred_bin_np = y_pred_bin.squeeze().cpu().numpy()

            y = model(x)
            img_x = torch.squeeze(y_true).numpy()
            img_y = torch.squeeze(y).numpy()
            img_input = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
            im_color = cv2.applyColorMap(img_input, cv2.COLORMAP_JET)
            img_x = img_as_ubyte(img_x)
            img_y = img_as_ubyte(img_y)
            imgStack = stackImages(0.8, [[img_input, img_x, img_y]])
            # 轉為偽彩色，視情況可加上
            # imgStack = cv2.applyColorMap(imgStack, cv2.COLORMAP_JET)
            cv2.imwrite(f'train_img/{i}.png',imgStack)
            plt.imshow(imgStack)
            i = i + 1
            plt.pause(0.1)
        plt.show()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()
    
    # train
    # train()

    # test()
    args.ckp = "weights_49.pth"
    test_1()

