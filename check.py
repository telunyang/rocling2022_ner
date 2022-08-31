import json
from pprint import pprint

# 搜尋 word 並取得對應 word_label 的 count
'''
例如

輸入關鍵字: 荷爾蒙
{'BODY': 19, 'CHEM': 109, 'DRUG': 1, 'O': 10}
'''
while 1:
    keyword = input("輸入關鍵字: ")

    dict_words = {}

    # 將原始檔案轉成符合 predict 的格式 (merge.json 是 train.json + test.json 合併資料)
    with open('dataset/merge.json', 'r', encoding='utf-8') as file:
        for obj in file:
            o = json.loads(obj)
            for index, word in enumerate(o['word']):
                if word == keyword:
                    # print(f"對應標註為: {o['word_label'][index]}")
                    if o['word_label'][index] not in dict_words:
                        dict_words[ o['word_label'][index] ] = 0
                    dict_words[ o['word_label'][index] ] += 1

        pprint(dict_words)

        print("\n")
