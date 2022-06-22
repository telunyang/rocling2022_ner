# 匯入套件
import logging
import pandas as pd
from pprint import pprint
from simpletransformers.ner import NERModel, NERArgs

# 日誌設定
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)





''' train '''

# 訓練資料檔案路徑
path_train_data = './dataset/train.json'

# 讀取 dataframe
df_train = pd.read_json(path_train_data, lines=True)

# 預覽前幾筆資料
# print(df_train.head())

# 將訓練資料轉換成 list of dict
list_train = df_train.values.tolist()

# 放置符合訓練格式的資料
train_data = []

# 儲存不重複的 labels
set_train_labels = set()
set_eval_labels = set()

'''
訓練資料 - 欄位
0: id
1: genre => "ft" (formal texts) and "sm" (social media)
2: sentence
3: word[]
4: word_label[]
5: character[]
6: character_label[]
'''
# 整理訓練資料
for index, line in enumerate(list_train):
    # word-based
    # for idx, word in enumerate(line[3]):
    #     train_data.append([
    #         line[0], word, line[4][idx]
    #     ])
    #     set_train_labels.add(line[4][idx]) # 整理出不重複的 labels

    # character-based
    for idx, char in enumerate(line[5]):
        train_data.append([
            line[0], char, line[6][idx]
        ])
        set_train_labels.add(line[6][idx]) # 整理出不重複的 labels

# 檢視訓練資料內容
print(train_data[:50])

# # # 整理 labels 資料，提供給 model args 使用
# labels_list = list(set_train_labels)
# pprint(labels_list)


# # 建立 dataframe 的 headers
# train_data = pd.DataFrame(
#     train_data, columns=["sentence_id", "words", "labels"]
# )





# ''' eval '''

# # 評估(測試)資料檔案路徑
# path_eval_data = './dataset/test.json'

# # 讀取 dataframe
# df_eval = pd.read_json(path_eval_data, lines=True)

# # 預覽前幾筆資料
# # print(df_eval.head())

# # 將評估(測試)資料轉換成 list of dict
# list_eval = df_eval.values.tolist()

# # 放置符合評估(測試)格式的資料
# eval_data = []

# # 整理訓練資料
# for index, line in enumerate(list_eval):
#     # word-based
#     # for idx, word in enumerate(line[3]):
#     #     eval_data.append([
#     #         line[0], word, line[4][idx]
#     #     ])

#     # character-based
#     for idx, char in enumerate(line[5]):
#         eval_data.append([
#             line[0], char, line[6][idx]
#         ])

# # 建立 dataframe 的 headers
# eval_data = pd.DataFrame(
#     eval_data, columns=["sentence_id", "words", "labels"]
# )





# ''' train '''

# # 自訂設定
# batch_size = 64
# epochs = 10
# output_dir = f'outputs/bert-base-chinese-ner-bs-{batch_size}-epo-{epochs}/' 

# # 自訂模型參數
# model_args = NERArgs()
# model_args.evaluate_during_training = True
# model_args.labels_list = labels_list
# model_args.train_batch_size = batch_size
# model_args.num_train_epochs = epochs
# model_args.output_dir = output_dir

# model = NERModel(
#     "bert", 
#     "bert-base-chinese", 
#     use_cuda = True, 
#     cuda_device = 0,
#     labels = labels_list, 
#     args = model_args # 帶入自訂參數
# )

# # 訓練模型
# model.train_model(train_data, eval_data=eval_data)





# ''' eval '''

# # 評估
# result, model_outputs, preds_list = model.eval_model(eval_data)





# ''' predict '''

# # 使用模型預測
# predictions, raw_outputs = model.predict([
#     '雞蛋含有多種維生素，包括Ｄ和Ｋ，是骨骼健康生長不可缺少的成分，又有豐富蛋白質。',
#     '１．重建餐桌時光，好好吃飯，就是好好地和大家一起吃飯，尊重每一個和自己吃飯的人，珍惜彼此溝通談天的機會，杜絕３Ｃ產品佔據餐桌的時光，「如果父母拿出手機，孩子也會跟著拿出手機，珍貴的交流時間就消失了，」張瑋琦說。',
#     '使用替代能源、加速總量管制、課徵空污健康捐、查緝烏賊車。'
# ])