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

## 預訓練模型選擇
- [HuggingFace](https://huggingface.co/) 首頁
  - [NER + Chinese](https://huggingface.co/models?sort=downloads&search=ner+chinese)

# 訓練花費時間
- bert-base-chinese
  - 16268.5659 秒 => 約略 4.519 小時
- chinese-macbert-base
  -  秒 => 約略  小時