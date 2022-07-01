# crowNER


## 套件安裝
- Step 1: PyTorch (GPU 版) [參考連結](https://pytorch.org/get-started/previous-versions/#linux-and-windows-13)
```python
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

- Step 2: SimpleTransformers [參考連結](https://simpletransformers.ai/docs/installation/#installation-steps)
```python
pip install simpletransformers
```

- Step 3: matplotlib
```python
pip install matplotlib
```

- Optional: opencc
```python
pip install opencc
```

## 程式檔案簡介
- s2t.py: 簡轉繁體 (透過 OpenCC)
- train.py: 訓練模型
- predict.py: 人工評估成效
- checkGPU.py: 確認電腦環境是否擁有 GPU
- plot.py: 基本 plot 輸出/檢視
- example.py: 基本的訓練與評估語法
- cals.py: 計算 confusion matrix (尚未完工)

## 預訓練模型選擇
- [HuggingFace](https://huggingface.co/) 首頁
  - [NER + Chinese](https://huggingface.co/models?sort=downloads&search=ner+chinese)

## 顯示卡型號
GPU: 1080 Gaming 8G

## 訓練花費時間
- bert-base-chinese
  - batch size: 64
  - epochs: 30
  - 16268.5659 秒 => 約略 4.519 小時
- chinese-macbert-base
  - batch size: 64
  - epochs: 30
  - 16363.2973 秒 => 約略 4.5454 小時