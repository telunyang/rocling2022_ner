'''
嘗試過的 pre-trained model
https://huggingface.co/bert-base-chinese
https://huggingface.co/hfl/rbt6
https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large


尚未嘗試的 pre-trained model
https://huggingface.co/hfl/chinese-macbert-large
https://huggingface.co/hfl/chinese-macbert-base
https://huggingface.co/9pinus/macbert-base-chinese-medical-collation
'''

# 匯入套件
import logging, sys, os
import pandas as pd
from time import time
from pprint import pprint
from random import shuffle
from simpletransformers.ner import NERModel, NERArgs

# transformers 的 logging 設定
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# 自訂 logging

class crowNER:
    '''建構子'''
    def __init__(self):
        # 放置符合訓練格式的資料 與 符合評估(測試)格式的資料
        self.train_data = []
        self.eval_data = []

        # 儲存不重複的 labels
        self.set_labels = set()

        # 不重複的 labels
        self.labels_list = []

        # 訓練資料檔案路徑 與 評估(測試)資料檔案路徑
        self.path_train_data = './dataset/train.json'
        self.path_train_data_1 = './dataset/ccks2018_m.json'
        self.path_eval_data = './dataset/test.json'

        # 自訂設定
        self.batch_size = 8
        self.eval_batch_size = 8
        self.epochs = 10
        self.model_type = 'bert'
        self.model_name = 'luhua/chinese_pretrain_mrc_macbert_large'
        self.output_dir = f'outputs/models/'

        # 自訂參數
        self.model_args = NERArgs()
        self.model_args.evaluate_during_training = True
        self.model_args.n_gpu = 1
        # self.model_args.learning_rate = 4e-5   # 4e-5
        # self.model_args.weight_decay = 1e-2
        self.model_args.max_seq_length = 128
        self.model_args.eval_batch_size = self.eval_batch_size
        self.model_args.train_batch_size = self.batch_size
        self.model_args.num_train_epochs = self.epochs
        self.model_args.output_dir = self.output_dir
        self.model_args.overwrite_output_dir = True
        self.model_args.reprocess_input_data = True
        self.model_args.use_multiprocessing = False
        self.model_args.use_multiprocessing_for_evaluation = False
        self.model_args.save_steps = -1
        self.model_args.save_model_every_epoch = False
        self.model = None

    '''讀取資料集'''
    def read_data(self):
        try:
            # 讀取 dataframe (來自 train.json)
            self.df_train = pd.read_json(self.path_train_data, lines=True)

            # 將訓練資料轉換成 list of dict
            self.list_train = self.df_train.values.tolist()

            # 讀取 dataframe (來自 ccks2018_m.json)
            self.df_train_1 = pd.read_json(self.path_train_data_1, lines=True)

            # 將訓練資料轉換成 list of dict
            self.list_train_1 = self.df_train_1.values.tolist()

            # 讀取 dataframe (來自 test.json)
            self.df_eval = pd.read_json(self.path_eval_data, lines=True)

            # 將評估(測試)資料轉換成 list of dict
            self.list_eval = self.df_eval.values.tolist()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))

    '''轉換資料'''
    def convert_data(self):
        try:
            for line in self.list_train:
                # word-based
                # for idx, word in enumerate(line[3]):
                #     train_data.append([
                #         line[0], word, line[4][idx]
                #     ])
                #     set_train_labels.add(line[4][idx]) # 整理出不重複的 labels

                # character-based
                for idx, char in enumerate(line[5]):
                    self.train_data.append([
                        line[0], char, line[6][idx]
                    ])
                    self.set_labels.add(line[6][idx]) # 整理出不重複的 labels

            for line in self.list_train_1:
                # character-based
                for idx, char in enumerate(line[5]):
                    self.train_data.append([
                        line[0], char, line[6][idx]
                    ])

            # 整理 labels 資料，提供給 model args 使用
            self.labels_list = list(self.set_labels)
            self.model_args.labels_list = self.labels_list

            # 建立 dataframe 的 headers
            self.train_data = pd.DataFrame(
                self.train_data, columns = ["sentence_id", "words", "labels"]
            )

            # 整理評估資料
            for line in self.list_eval:
                # word-based
                # for idx, word in enumerate(line[3]):
                #     eval_data.append([
                #         line[0], word, line[4][idx]
                #     ])

                # character-based
                for idx, char in enumerate(line[5]):
                    self.eval_data.append([
                        line[0], char, line[6][idx]
                    ])
            # 建立 dataframe 的 headers
            self.eval_data = pd.DataFrame(
                self.eval_data, columns = ["sentence_id", "words", "labels"]
            )
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))

    '''預覽訓練或評估資料內容'''
    def get_head_data(self, type = 'train', num = 5):
        try:
            if type == 'train':
                return self.train_data.head(num)
            else:
                return self.eval_data.head(num)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))

    '''儲存成 csv'''
    def save_to_csv(self):
        try:
            self.train_data.to_csv("training_data.csv", index = False, encoding = "utf-8")
            self.eval_data.to_csv("eval_data.csv", index = False, encoding = "utf-8")
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))

    '''訓練模型'''
    def train(self):
        try:
            # 模型設定
            self.model = NERModel(
                self.model_type, 
                self.model_name,
                use_cuda = True, 
                cuda_device = 0,
                args = self.model_args # 帶入自訂參數
            )

            # 訓練模型
            self.model.train_model(self.train_data, eval_data = self.eval_data)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))

    '''評估模型'''
    def eval(self):
        try:
            # 評估
            result, model_outputs, preds_list = self.model.eval_model(self.eval_data)
            print("=== 評估結果 ===")
            pprint(result)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(str(e))

if __name__ == "__main__":
    # 計算整體執行時間
    time_s = time()
    try:
        # 實體化
        obj = crowNER()

        # 讀取資料集
        obj.read_data()

        # 轉換成訓練與評估資料
        obj.convert_data()

        # 預覽資料
        pprint(obj.get_head_data('train', 10))
        pprint(obj.get_head_data('eval', 10))

        # 將 dataframe 儲存成 csv
        obj.save_to_csv()

        # 訓練模型
        obj.train()

        # 評估模型
        obj.eval()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    print(f"整體執行時間: {time() - time_s} 秒")