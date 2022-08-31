'''
標註檔案來源: Seven、秋霞
'''
import uuid
import json, time, re

# 將隊友標註好的資料，轉成訓練格式
def label_to_train_data():
    # 讀取 label 結束後的 json 檔
    with open("dataset/rocling_after_process.json", "r", encoding="gbk") as file:
        content = file.read()
        content = content.encode('utf-8').decode('utf-8')

    # 建立訓練語料 (原先的 keys: id, sentence, character_label)
    list_json = json.loads(content)
    str_lines = ''
    for obj in list_json:
        o = {
            "id": str(uuid.uuid4()),
            "genre": 'ner',
            "sentence": obj['sentence'],
            "word": [],
            "word_label": [],
            "character": [char for char in obj['sentence']],
            "character_label": obj['character_label']
        }
        str_obj = json.dumps(o, ensure_ascii=False)
        str_lines += str_obj + "\n"

    # 儲存訓練語料
    with open("dataset/labelling_train.json", "w", encoding="utf-8") as file:
        file.write(str_lines)

# 官方 test 資料轉成 predict 用的格式
def test_to_predict_data():
    # 儲存資料的變數初始化
    list_data = []
    
    # 將原始檔案轉成符合 predict 的格式
    with open('dataset/ROCLING22_CHNER_test.txt', 'r', encoding='utf-8') as file:
        content = file.read()

        # 將主要區隔切割成 list of strings
        list_raw_sentence = content.split("\n \n")
        
        # 將每個 list of strings 變成原始的 sentence，再一個一個放到儲存資料的變數
        for raw_data in list_raw_sentence:
            sentence = re.sub(r"\n", "", raw_data)
            list_data.append(sentence)
    
    # 將儲存資料的變數存成 json 檔 
    with open('dataset/ROCLING22_CHNER_test.json', 'w', encoding='utf-8') as file:
        file.write( json.dumps( list_data, ensure_ascii=False ) )

    # 將 predict 格式的 sentences，另外存成 txt，用於 web (5566) 產生資料
    with open('dataset/ROCLING22_CHNER_test_sentences.txt', 'w', encoding='utf-8') as file:
        str_sentences = ''
        for sentence in list_data:
            str_sentences += sentence + "\n"
        file.write(str_sentences)


'''主程式'''
if __name__ == "__main__":
    # 計算執行時間
    time1 = time.time()

    # 官方 test 資料轉成 predict 用的格式
    test_to_predict_data()

    # 顯示執行時間
    print(f"轉換時間: {time.time() - time1} 秒 => {(time.time() - time1) / 60} 分鐘")