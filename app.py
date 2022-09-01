import sys, math, os, logging
from time import time
from pprint import pprint
from simpletransformers.ner import NERModel
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory

''' Flask 初始化 '''
app = Flask(__name__)

# 模型設定
model_type = 'bert'

'''
./model_bert-base-chinese_train_test_standalone/
./model_train-test-labelled-custom250sent_standalone
'''
output_dir = './model_train-test-labelled-custom250sent_standalone'

model = NERModel(
    model_type, 
    output_dir,
    use_cuda = True, 
    cuda_device = 0
)

'''
Templates
'''
# 套用網頁樣版(v1.0)
@app.route('/', methods=['GET'])
def index():
    return render_template('query.html')

'''Web API'''
# 使用向量查詢
@app.route('/predict', methods=['POST'])
def predict():
    global model

    # 預設錯誤訊息
    res_dict = {
        'success': False,
        'info': '請求失敗',
        'results': None
    }

    # 取得 POST 請求的 sentences 值
    if 'sentences' in request.values and request.values['sentences'] != '': 
        str_sentences = request.values['sentences']
    else:
        return jsonify(res_dict)

    # 斷句
    list_sentences = str_sentences.split("\n")

    # 預測結果
    predictions, raw_outputs = model.predict(list_sentences, split_on_space=False)

    # 整合資料
    res_dict['success'] = True
    res_dict['info'] = '預測完成'
    res_dict['results'] = predictions

    return jsonify(res_dict)
    

    

''' 檔案路徑 '''
# JS 資料夾
@app.route('/js/<path>')
def get_js_path(path):
    return send_from_directory('js', path)

# CSS 資料夾
@app.route('/css/<path>')
def get_css_path(path):
    return send_from_directory('css', path)

# 暫存檔案路徑
@app.route('/tmp/<path>')
def get_tmp_path(path):
    return send_from_directory('tmp', path)

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0', 
        port=5566)