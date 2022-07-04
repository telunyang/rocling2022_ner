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
  - bert-base-chinese
  - chinese-macbert-base

## 顯示卡型號
- 桌機 (主要訓練用)
  - GPU: 1080 Gaming 8G
- 筆電 (測試訓練程式用)
  - GPU: 2070 Max-Q Design

## 訓練設定
- batch size: 64
- epochs: 30

## 訓練 / 預測 結果
- 2022-07-03 (實驗編號: T01)
  - 訓練資料: train.json 取 70%
  - 驗證資料: 訓練資料的另外 30%

|           | bert-base-chinese  | chinese-macbert-base |
|:----------|:-------------------|:---------------------|
| eval_loss | 0.2883742261107994 | 0.277212801452418 |
| f1_score  | 0.7530979347101933 | 0.7566796717341585 |
| precision | 0.7236505172590393 | 0.7305933700509196 |
| recall    | 0.7850436135340852 | 0.7846978067972543 |

- 2022-07-04 (實驗編號: T02)
  - 訓練資料: train.json + ccks2018_m.json(簡轉繁)，取 70%
  - 驗證資料: 訓練資料的另外 30%

|           | bert-base-chinese  | chinese-macbert-base |
|:----------|:-------------------|:---------------------|
| eval_loss | 0.28548791684486247 | 0.27646474805143145 |
| f1_score  | 0.7667149556805066 | 0.7636156682255709 |
| precision | 0.7374710689150252 | 0.7360938841713577 |
| recall    | 0.7983739005931684 | 0.793275410004627 |

- 2022-07-04 (實驗編號: T03)
  - 訓練資料: train.json + ccks2018_m.json(簡轉繁) + **test.json (validation data)**，取 70%
  - 驗證資料: 訓練資料的另外 30%

|           | bert-base-chinese  | chinese-macbert-base |
|:----------|:-------------------|:---------------------|
| eval_loss | 0.28819293192484313 | 0.28106662206765676 |
| f1_score  | 0.7655794338014267 | 0.7634140419802681 |
| precision | 0.7433998431645901 | 0.7344148122982843 |
| recall    | 0.7891231964483907 | 0.7947975550346983 |

- 2022-07-05 (實驗編號: T04)
  - 訓練資料: train.json + ccks2018_m.json(簡轉繁) + test.json (validation data) +  **ccks2017_m.json(簡轉繁)**，取 70%
  - 驗證資料: 訓練資料的另外 30%

|           | bert-base-chinese  | chinese-macbert-base |
|:----------|:-------------------|:---------------------|
| eval_loss |  |  |
| f1_score  |  |  |
| precision |  |  |
| recall    |  |  |

## 訓練花費時間
| 實驗編號 | bert-base-chinese       | chinese-macbert-base     |
|:-------:|------------------------:|-------------------------:|
|   T01   | 3.11 小時               | 3.1244 小時               |
|   T02   | 3.2153 小時             | 3.229  小時               |
|   T03   | 3.3468 小時             | 3.344 小時                |
|   T04   |  小時             |  小時                |