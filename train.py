# 匯入套件
import logging, sys, os, traceback
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
        self.path_train_data_ccks2018 = './dataset/ccks2018_m.json'
        self.path_eval_data = './dataset/test.json'

        # 自訂設定
        self.batch_size = 64
        self.eval_batch_size = 64
        self.epochs = 30
        self.model_type = 'bert'
        self.model_name = 'bert-base-chinese' # hfl/chinese-macbert-base , bert-base-chinese
        self.output_dir = f'outputs-bert-base-chinese/' # outputs-chinese-macbert-base/, outputs-bert-base-chinese/

        # 自訂參數
        self.model_args = NERArgs()
        self.model_args.n_gpu = 1
        self.model_args.evaluate_during_training = True
        self.model_args.eval_batch_size = self.eval_batch_size
        self.model_args.train_batch_size = self.batch_size
        self.model_args.num_train_epochs = self.epochs
        self.model_args.output_dir = self.output_dir
        self.model_args.overwrite_output_dir = True
        self.model_args.reprocess_input_data = True
        self.model_args.use_multiprocessing = False
        self.model_args.use_multiprocessing_for_evaluation = False
        # self.model_args.save_steps = -1
        # self.model_args.save_model_every_epoch = False

        # model 變數初始化
        self.model = None

        # 不重複的 labels
        self.labels = [
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

    '''讀取資料集'''
    def read_data(self):
        try:
            # 將訓練資料轉換成 list of dict
            self.list_train = pd.read_json(self.path_train_data, lines=True).values.tolist()
            self.list_train += self.list_train + pd.read_json(self.path_train_data_ccks2018, lines=True).values.tolist()
            # self.list_train += self.list_train + pd.read_json(self.path_eval_data, lines=True).values.tolist()

            # 手動切分
            shuffle(self.list_train) # 洗牌
            len_train_data = len(self.list_train) # 資料總數
            middle = int(len_train_data * 0.7) # 訓練資料的總數 (70%)
            list_train = self.list_train[:middle] # 取得訓練資料 (70%)
            list_eval = self.list_train[middle:] # 取得評估資料 (30%)

            # 準備訓練與評估資料
            self.list_train = list_train
            self.list_eval = list_eval

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
                    # self.set_labels.add(line[6][idx]) # 整理出不重複的 labels

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
                labels = self.labels,
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
            eexc_type, exc_obj, exc_tb = sys.exc_info()
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

        # # 預覽資料
        # pprint(obj.get_head_data('train', 10))
        # pprint(obj.get_head_data('eval', 10))

        # # 將 dataframe 儲存成 csv
        # obj.save_to_csv()

        # 訓練模型
        obj.train()

        # 評估模型
        obj.eval()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(str(e))

    print(f"整體執行時間: {time() - time_s} 秒")