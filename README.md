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
- s2t.py: 簡轉繁體 (透過 OpenCC)
- train.py: 訓練模型
- predict.py: 人工評估成效
- checkGPU.py: 確認電腦環境是否擁有 GPU
- plot.py: 基本 plot 輸出/檢視
- example.py: 基本的訓練與評估語法

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

## 訓練 / 預測 結果
- 2022-07-07 (實驗編號: T01)
  - 訓練資料: train.json 取 70%
  - 驗證資料: 訓練資料的另外 30%

|           | bert-base-chinese  | chinese-macbert-base |
|:----------|:-------------------|:---------------------|
| eval_loss | 0.0010151309992032164 |  |
| f1_score  | 0.994490291846113 |  |
| precision | 0.9928658932045831 |  |
| recall    | 0.9961200144595734 |  |

- 2022-07-07 (實驗編號: T02)
  - 訓練資料: train.json + test.json，取 70%
  - 驗證資料: 訓練資料的另外 30%

|           | bert-base-chinese  | chinese-macbert-base |
|:----------|:-------------------|:---------------------|
| eval_loss | 0.0009665362070964399 |  |
| f1_score  | 0.9950256191289496 |  |
| precision | 0.9934772877454011 |  |
| recall    | 0.9965787841854299 |  |

## 訓練花費時間
| 實驗編號 | bert-base-chinese       | chinese-macbert-base     |
|:-------:|------------------------:|-------------------------:|
|   T01   | 2.7964 小時               |  小時               |
|   T02   | 3.1944 小時             |   小時               |
|   T03   |  小時             |  小時                |
|   T04   |  小時             |  小時               |