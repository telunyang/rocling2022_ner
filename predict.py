# 匯入套件
from simpletransformers.ner import NERModel
from time import time
from pprint import pprint
import json

# 模型路徑
model_type = 'bert'
# output_dir = './model_train-test_standalone'
# output_dir = './model_train-test-custom250sent_standalone'
output_dir = './model_train-test-labelled-custom250sent_standalone'

# 官方提供 eval 資料 (被轉換過，非原始資料)
path_eval_data = './dataset/ROCLING22_CHNER_test.json'

# eval 結果儲存路徑
# path_save_to = './crowNER_Run1_train-test.txt'
# path_save_to = './crowNER_Run1_train-test-custom250sent.txt'
path_save_to = './crowNER_Run1_train-test-labelled-custom250sent.txt'

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
    print(f"模型讀取時間: {time() - time_model} 秒")

    # 測試句子(可以有很多筆，都 append 在 list_sentences 當中)
    time_read_eval_data = time()
    with open(path_eval_data, "r", encoding="utf-8") as file:
        list_sentences = json.loads( file.read() )
    print(f"比賽 test data 讀取時間: {time() - time_read_eval_data} 秒")

    # 使用模型預測 (split_on_space 預設為 True，將英文單字間的空白當作分隔符號，但中文不需要，所以要改成 False)
    time_predict = time()
    predictions, raw_outputs = model.predict(list_sentences, split_on_space=False)
    print(f"預測執行時間: {time() - time_predict} 秒")

    # pprint(predictions)



    # 放置評估結果的字串變數
    str_txt = ''

    # 將預測結果寫入 eval (test submission) 檔案
    time_save_to = time()
    with open(path_save_to, "w", encoding="utf-8") as file:
        # list_result 裡面有許多 dict
        for list_result in predictions:
            # 每個 dict_char 的 key 都是 character，其 label 為 BIO tag 格式 
            for dict_char in list_result:
                '''
                因為不知道 dict_char 會有哪些 key，所以直接把所有 key: value 組合找出來
                (雖然每個 dict_char 只會有一組 key:value)
                '''
                for char in dict_char:
                    str_txt += f"{char} {dict_char[char]}\n"

            # 每整理一個句子的結果，中間用兩個斷行間隔
            str_txt += "\n"

        # 寫入檔案
        file.write(str_txt)

    print(f"建立 test submission 花費時間: {time() - time_save_to} 秒")

        


    print(f"整體執行時間: {time() - time_s} 秒")