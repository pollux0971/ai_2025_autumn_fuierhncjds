# DN-DETR Windows 部署和訓練指南

## 📋 系統需求

- **Windows 10/11**
- **Python 3.8-3.11** (建議 3.9 或 3.10)
- **CUDA 11.8** (如果使用 GPU 訓練)
- **至少 8GB RAM**
- **GPU**: 建議 4GB+ VRAM (如 GTX 1650, RTX 2060 或更好)

## 🚀 快速開始

### 方法 1: 一鍵部署 (推薦)

1. **下載並解壓 DN-DETR 專案**
2. **準備你的資料集**：
   ```
   coco2017_augmented/
   ├── train2017/
   ├── val2017/
   └── annotations/
       ├── instances_train2017.json
       └── instances_val2017.json
   ```

3. **執行一鍵部署腳本**：
   ```cmd
   setup_and_train_windows.bat
   ```

這個腳本會自動：
- 創建虛擬環境
- 安裝所有依賴項
- 檢查 CUDA 可用性
- 開始訓練

### 方法 2: 手動安裝

1. **創建虛擬環境**：
   ```cmd
   python -m venv dn_detr_env
   dn_detr_env\Scripts\activate.bat
   ```

2. **安裝 PyTorch (CUDA 版本)**：
   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **安裝其他依賴**：
   ```cmd
   pip install -r requirements_windows.txt
   pip install pycocotools
   pip install "git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi"
   ```

4. **開始訓練**：
   ```cmd
   train_windows.bat
   ```

## 📁 文件說明

| 文件名 | 說明 |
|--------|------|
| `setup_and_train_windows.bat` | 一鍵部署腳本 (環境設置+訓練) |
| `train_windows.bat` | 純訓練腳本 (需要已設置環境) |
| `train_custom.py` | 訓練主程式 (含 GPU 監控) |
| `requirements_windows.txt` | Windows 依賴清單 |

## ⚙️ 訓練配置

### 預設參數
- **模型**: DN-DAB-DETR
- **Epochs**: 200 (最多)
- **Early Stopping**: patience=3
- **Batch Size**: 2 (適合 4GB GPU)
- **學習率調整**: 第 150 epoch

### 自定義參數
執行 `train_windows.bat` 時會提示輸入：
- 訓練輪數
- Early stopping patience
- Batch size (根據你的 GPU 記憶體調整)
- 資料集路徑

## 🖥️ GPU 監控功能

訓練過程中會即時顯示：
- **GPU 記憶體使用量** (已用/總容量)
- **GPU 利用率百分比**
- **系統資源監控** (CPU, RAM)
- **每個 epoch 的 AP 分數**

範例輸出：
```
============================================================
System Stats:
CPU Usage: 15.2%
RAM Usage: 6.79GB / 16.00GB (42.4%)
GPU Memory:
GPU 0: 2.15GB / 4.00GB allocated, 2.25GB cached
GPU 0 (NVIDIA GTX 1650): 87.5% memory, 95.2% utilization
============================================================

[Epoch 1/200] Starting...
GPU 0: 2.15GB / 4.00GB allocated, 2.25GB cached
```

## 🔧 故障排除

### 常見問題

**1. CUDA 記憶體不足**
```
解決方案：
- 減少 batch_size (從 2 改為 1)
- 關閉其他使用 GPU 的程式
```

**2. PyTorch 無法找到 CUDA**
```cmd
# 檢查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果回傳 False，安裝 CPU 版本
pip install torch torchvision torchaudio
```

**3. 資料集路徑錯誤**
```
確保資料集結構正確：
coco2017_augmented/
├── annotations/instances_train2017.json
├── annotations/instances_val2017.json
├── train2017/ (包含圖片)
└── val2017/ (包含圖片)
```

**4. 依賴安裝失敗**
```cmd
# 升級 pip
python -m pip install --upgrade pip

# 重新安裝依賴
pip install --upgrade --force-reinstall -r requirements_windows.txt
```

## 📊 訓練結果

訓練完成後，結果保存在 `logs/dn_dab_detr/custom_training/`:
- `checkpoint.pth` - 最新模型權重
- `log.txt` - 訓練日誌
- `eval/` - 評估結果

## 🎯 性能預期

使用你的自製資料集，DN-DETR 通常可以達到：
- **mAP**: 35-50% (取決於資料集質量)
- **訓練時間**: GTX 1650 約 1-2 小時/epoch

## 💡 優化建議

1. **資料擴增**: 確保資料集有足夠的多樣性
2. **Early Stopping**: 使用 patience=3 避免過擬合
3. **學習率調整**: 預設在第 150 epoch 降低學習率
4. **GPU 監控**: 觀察記憶體使用，適時調整 batch_size

## 📞 需要幫助？

如果遇到問題：
1. 檢查 GPU 記憶體使用情況
2. 確認資料集格式正確
3. 查看 `log.txt` 中的錯誤訊息
4. 嘗試減少 batch_size 或使用 CPU 訓練