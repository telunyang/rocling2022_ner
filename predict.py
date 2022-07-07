# 匯入套件
import pandas as pd
from time import time
from pprint import pprint
from simpletransformers.ner import NERModel

# 模型路徑
model_type = 'bert'
output_dir = 'model_bert-base-chinese_T01/'

if __name__ == "__main__":
    # 模型設定
    model = NERModel(
        model_type, 
        output_dir,
        use_cuda = True, 
        cuda_device = 0
    )

    # 測試句子(可以有很多筆，都 append 在 list_sentences 當中)
    list_sentences = [
        # '身體若有發炎較常導致該指數偏高，但若是病毒感染會合併發燒。',
        # '修復肌肉與骨骼最重要的便是熱量、蛋白質與鈣質。',
        '如何治療胃食道逆流症？'
    ]

    # 使用模型預測 (split_on_space 預設為 True，將英文單字間的空白當作分隔符號，但中文不需要，所以要改成 False)
    predictions, raw_outputs = model.predict(list_sentences, split_on_space=False)
    print("=== 預測結果 ===")
    print(predictions)