# 匯入套件
import logging, sys
import pandas as pd
from time import time
from pprint import pprint
from simpletransformers.ner import NERModel, NERArgs

# 日誌設定
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

class crowNER:
    '''建構子'''
    def __init__(self):
        # 自訂設定
        self.model_args = NERArgs()
        self.batch_size = 4
        self.epochs = 10
        # self.output_dir = f'outputs/bs-{self.batch_size}-epo-{self.epochs}/'
        self.output_dir = f'outputs/best_model/'
        self.model = None

        # 不重複的 labels
        self.labels_list = [
            "B-BODY","I-BODY",
            "B-CHEM","I-CHEM",
            "B-DISE","I-DISE", 
            "B-DRUG","I-DRUG",
            "B-EXAM","I-EXAM",
            "B-INST","I-INST",
            "B-SUPP","I-SUPP",
            "B-SYMP","I-SYMP",
            "B-TIME","I-TIME",
            "B-TREAT","I-TREAT",
            "O"
        ]

    '''預測結果'''
    def predict(self, list_sentences):
        try:
            if len(list_sentences) > 0:
                # 自訂參數
                self.model_args.evaluate_during_training = True
                # self.model_args.labels_list = self.labels_list
                self.model_args.train_batch_size = self.batch_size
                self.model_args.num_train_epochs = self.epochs
                self.model_args.output_dir = self.output_dir
                self.model_args.overwrite_output_dir = True
                self.model_args.reprocess_input_data = True
                self.model_args.use_multiprocessing = False
                self.model_args.save_model_every_epoch = False

                # 模型設定
                self.model = NERModel(
                    "bert", 
                    self.output_dir, # 這裡要放先前訓練好的模型路徑!!
                    use_cuda = True, 
                    cuda_device = 0,
                    labels = self.labels_list, 
                    args = self.model_args # 帶入自訂參數
                )

                # 使用模型預測 (split_on_space 預設為 True，將英文單字間的空白當作分隔符號，但中文不需要，所以要改成 False)
                predictions, raw_outputs = self.model.predict(list_sentences, split_on_space=False)
                print("=== 預測結果 ===")
                pprint(predictions)
            else:
                print("請提供預測文句 (list 格式，一句以上亦可)")
        except Exception as e:
            print("Unexpected error: ", sys.exc_info())

if __name__ == "__main__":
    # 計算整體執行時間
    time_s = time()
    try:
        # 實體化
        obj = crowNER()

        # 評估模型
        list_sentences = [
            '需要看醫生嗎？',
            '以上建議希望對您有幫助。'
        ]
        obj.predict(list_sentences)
    except Exception as e:
        print("Unexpected error: ", sys.exc_info())

    print(f"整體執行時間: {time() - time_s} 秒")