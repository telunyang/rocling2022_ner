# crowNER
團隊成員: 張智星教授、戚得郁、張秋霞、楊德倫

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

- Optional: opencc 與 flask
```python
pip install opencc flask
```

## 程式檔案簡介
- app.py: Flask 的 app.py
- s2t.py: 簡轉繁體 (透過 OpenCC)
- train.py: 訓練模型
- predict.py: 人工評估成效
- checkGPU.py: 確認電腦環境是否擁有 GPU
- plot.py: 基本 plot 輸出/檢視
- example.py: 測試語法用的程式檔
- check.py: 檢查關鍵字的數量
- convert.py: 將夥伴的整理好的資料，轉換成訓練資料
- web_scraper_kingnet.py: 取得 KingNet 網站的衛教資訊

## NER 的 tokens
```python
labels = [
    "O",
    "B-BODY","I-BODY",
    "B-CHEM","I-CHEM",
    "B-DISE","I-DISE",
    "B-DRUG","I-DRUG",
    "B-EXAM","I-EXAM",
    "B-INST","I-INST",
    "B-SUPP","I-SUPP",
    "B-SYMP","I-SYMP",
    "B-TIME","I-TIME",
    "B-TREAT","I-TREAT"
]
```

## 預訓練模型選擇
- [HuggingFace](https://huggingface.co/)
  - [bert-base-chinese](https://huggingface.co/bert-base-chinese)
  - [hfl/chinese-macbert-base](https://huggingface.co/hfl/chinese-macbert-base)
  - [ckiplab/albert-base-chinese-ner](https://huggingface.co/ckiplab/albert-base-chinese-ner)
  - [hfl/rbt6](https://huggingface.co/hfl/rbt6)

## 顯示卡型號
- 桌機 (主要訓練用)
  - GPU: 1080 Gaming 8G
- 筆電 (測試訓練程式用)
  - GPU: 2070 Max-Q Design

## 語料
- [train.json、test.json](https://github.com/NCUEE-NLPLab/Chinese-HealthNER-Corpus)
- ccks2017.json、ccks2018.json (由 秋霞、得郁 提供)

## 訓練設定
- batch size: 64
- epochs: 30

## 成果
- [ACL Anthology: CrowNER at Rocling 2022 Shared Task: NER using MacBERT and Adversarial Training](https://aclanthology.org/2022.rocling-1.40/)