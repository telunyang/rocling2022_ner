# 匯入套件
from simpletransformers.ner import NERModel
from time import time
from pprint import pprint
import json

# 模型路徑
model_type = 'bert'

'''
./model_bert-base-chinese_train_test_standalone/
./model_train-test-labelled-custom250sent_standalone
'''
output_dir = './model_train-test-labelled-custom250sent_standalone'

# 官方提供 eval 資料 (被轉換過，非原始資料)
path_eval_data = './dataset/ROCLING22_CHNER_test.json'

if __name__ == "__main__":
    # 計算執行時間
    time_s = time()

    # 模型設定
    time_model = time()
    model = NERModel(
        model_type, 
        output_dir,
        use_cuda = True, 
        cuda_device = 0
    )
    print(f"模型讀取時間: {time() - time_model} 秒 => {(time() - time_model) / 60} 分鐘")

    # 測試句子(可以有很多筆，都 append 在 list_sentences 當中)
    time_read_eval_data = time()
    with open(path_eval_data, "r", encoding="utf-8") as file:
        list_sentences = json.loads( file.read() )
    print(f"資料讀取時間: {time() - time_read_eval_data} 秒 => {(time() - time_read_eval_data) / 60} 分鐘")

    # 使用模型預測 (split_on_space 預設為 True，將英文單字間的空白當作分隔符號，但中文不需要，所以要改成 False)
    time_predict = time()
    predictions, raw_outputs = model.predict(list_sentences[:2], split_on_space=False)
    print(f"預測執行時間: {time() - time_read_eval_data} 秒 => {(time() - time_read_eval_data) / 60} 分鐘")

    pprint(predictions)


    print(f"整體執行時間: {time() - time_s} 秒 => {(time() - time_s) / 60} 分鐘")