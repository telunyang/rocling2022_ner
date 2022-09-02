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

# 將 test submissions (.txt 格式) 轉成 training data (.json 格式)
def txt_to_train_data():
    # predict 結束後，產生的 test submission 檔案
    list_path_txt = [
        './submissons/bert/crowNER_Run1_train-test.txt',
        './submissons/bert/crowNER_Run2_train-test-custom250sent.txt',
        './submissons/bert/crowNER_Run3_train-test-labelled-custom250sent.txt'
    ]

    # 將預測結果轉換成比對成效用的 json lines 格式
    list_path_save_to = [
        './submissons/bert/crowNER_Run1_train-test.json',
        './submissons/bert/crowNER_Run2_train-test-custom250sent.json',
        './submissons/bert/crowNER_Run3_train-test-labelled-custom250sent.json'
    ]

    # 走訪每一個 txt 檔
    for index in range( len(list_path_txt) ):
        # 開啟 txt 檔
        with open(list_path_txt[index], "r", encoding="utf-8") as file:
            content = file.read()

        # 將 txt 的內容經過處理後，存在 json 檔當中
        with open(list_path_save_to[index], "w", encoding="utf-8") as file:
            # 將主要區隔切割成 list of strings
            list_raw_sentence = content.split("\n\n")
            '''
            例如 list_raw_sentence[1] 的 raw_data 可能是以下用 \n 區隔每一個 [char]space[BIOtag]:

            [raw_data 的內容]
            在 O
            一 O
            個 O
            個 O
            體 O
            當 O
            中 O
            所 O
            有 O
            的 O
            基 B-BODY
            因 I-BODY
            總 O
            和 O
            叫 O
            基 B-BODY
            因 I-BODY
            組 I-BODY
            。 O
            '''
            # 取得 raw data
            for raw_data in list_raw_sentence:
                # 基本 json line 儲存結構
                obj = {
                    "id": str(uuid.uuid4()),
                    "genre": 'txt',
                    "sentence": '',
                    "word": [],
                    "word_label": [],
                    "character": [],
                    "character_label": []
                }

                # 每個 raw data 用 \n 切割，其元素為字串格式的 [char]space[BIOtag]
                '''
                ['在 O', '一 O', '個 O', ...]
                '''
                list_char_space_tag = raw_data.split("\n")

                # 逐個 [char]space[BIGtag] 進行分割
                for str_char_space_tag in list_char_space_tag:
                    '''
                    list_char_label[0] 是 character
                    list_char_label[1] 是 label
                    '''
                    list_char_label = str_char_space_tag.split(" ")
                    
                    # 若是切割結果不是 character 和 label 的組合，通常是空行，則略過
                    if len(list_char_label) < 2:
                        continue

                    obj['character'].append( list_char_label[0] )
                    obj['character_label'].append( list_char_label[1] )

                # 將所有 characters 合併成字串
                obj['sentence'] = ''.join(obj['character'])

                # 如果 sentence 為空，代表空行，則不進行寫入
                if obj['sentence'] != '':
                    file.write( json.dumps(obj, ensure_ascii=False) + "\n" )


'''主程式'''
if __name__ == "__main__":
    # 計算執行時間
    time1 = time.time()

    # 將 test submissions 轉成 training data
    txt_to_train_data()

    # 顯示執行時間
    print(f"轉換時間: {time.time() - time1} 秒 => {(time.time() - time1) / 60} 分鐘")