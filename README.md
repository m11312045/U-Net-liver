# U-Net-liver
基於UNet的肝CT分割
### 環境的配置：
pytorch==1.10.2
python==3.6.12
skimage==0.17.2
opencv-python==4.4.0
其他庫依需求安裝

### 訓練
大約50分鐘，先執行train()再執行test_1()函數
> python main.py
#### 如果報錯，可以註解或取消註解第85行，或是批次檔檔名錯誤
#### train_img
當中保存測試集資料的效果，圖片由左至右分別為：原肝臟CT切片，人工標註的mask，訓練出的模型預測效果

### 產生各指標
> python main_SegmentationIndex.py
> 
> python main_ConfusionMatrix.py
