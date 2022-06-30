# 匯入套件
import logging, sys
import pandas as pd
from time import time
from pprint import pprint
from simpletransformers.ner import NERModel, NERArgs


# 模型設定
model = NERModel(
    model_type, 
    output_dir,
    use_cuda = True, 
    cuda_device = 0
)

# 評估模型
# list_sentences = [
#     '需要看醫生嗎？',
#     '以上建議希望對您有幫助。'
# ]
list_sentences = [
    '身體若有發炎較常導致該指數偏高，但若是病毒感染會合併發燒。',
]

# 使用模型預測 (split_on_space 預設為 True，將英文單字間的空白當作分隔符號，但中文不需要，所以要改成 False)
predictions, raw_outputs = model.predict(list_sentences, split_on_space=False)
print("=== 預測結果 ===")
print(predictions)