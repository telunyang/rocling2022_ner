'''
套件匯入
'''
import requests as req
import os, json, sys
from time import time, sleep
from random import randint
from bs4 import BeautifulSoup as bs
from urllib import parse
from fake_useragent import UserAgent # 隨機取得 User-Agent
ua = UserAgent(cache=True) # cache=True 表示從已經儲存的列表中提取


'''
自訂變數
'''
# 取得基本資訊的網址
url = "https://www.kingnet.com.tw/ajax/selectLastNews?dateType=all&dataIndex=9400&dataCnt=10"

# 總頁數與分頁資訊
total_pages = -1
pagination = 30 # 最多 30

# 放置新聞相關的 metadata
list_data = []


'''
自訂函式 - 原始資料擷取
'''
# 初始化 (若用於 ipynb，可助於變數初始化，而不會持續增加 list_data 的多餘資料)
def init():
    global total_pages, pagination, list_data
    total_pages = -1
    pagination = 30 # 最多 30
    list_data = []

# 隨機取得 user-agent
def get_user_agent():
    return {'user-agent': ua.random}

# 取得總頁數 (total_pages)
def get_total_pages():
    global total_pages
    try:
        # 走訪 web api，取得分頁基本資訊
        res = req.get(url = url, headers = get_user_agent())

        # 若請求成功，則取得 json 的 total 值，作為總頁數
        if res.status_code == req.codes.ok: total_pages = res.json()['total']

        # 隨機等待 (politeness policy)
        sleep( randint(3, 5) )
    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(str(ex))

# 走訪每一個分頁，並取得主要資料
def get_page_data():
    global total_pages, list_data
    try:
        # 若總頁數被更新，則進行分頁走訪
        if total_pages != -1:
            # 走訪分頁型態的 web api
            for dataIndex in range(0, total_pages, pagination):
                # 整理網址
                url = f"https://www.kingnet.com.tw/ajax/selectLastNews?dateType=all&dataIndex={dataIndex}&dataCnt={pagination}"
                
                # 走訪 web api
                res = req.get(url = url, headers = get_user_agent())
                
                # 取得新聞列表
                list_data += res.json()['news']

                print(url)

                # 隨機等待 (politeness policy)
                sleep( randint(2, 8) )
        else:
            print("分頁資訊取得有誤")
    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(str(ex))

# 將 metadata 存成 json (原始資料)
def save_json():
    global list_data
    with open("kingnet_news_list.json", "w", encoding="utf-8") as file:
        file.write( json.dumps(list_data, ensure_ascii=False) )
        sleep(2)

'''
自訂函式 - 資料處理與文章擷取
'''

# 將 json 當中被編碼的資料，變成人類可讀的文字
def convert_encoded_text():
    # 讀取 json 檔
    with open("kingnet_news_list.json", "r", encoding="utf-8") as file:
        list_data = json.loads(file.read())

    # 資料處理
    for index, obj in enumerate(list_data):
        list_data[index]['newTitle'] = parse.unquote( list_data[index]['newTitle'] )
        list_data[index]['newContent'] = parse.unquote( list_data[index]['newContent'] )
        list_data[index]['newDescription'] = parse.unquote( list_data[index]['newDescription'] )
        list_data[index]['url'] = f"https://www.kingnet.com.tw/news/single?newId={ list_data[index]['newId'] }"

    # 另存 json
    with open("kingnet_news_list_conv.json", "w", encoding="utf-8") as file:
        file.write( json.dumps(list_data, ensure_ascii=False) )
    
# 取得新聞內文
def get_news_text():
    try:
        # 讀取 json 檔
        with open("kingnet_news_list_conv.json", "r", encoding="utf-8") as file:
            list_data = json.loads(file.read())

        # 走訪每一則新聞
        for index, obj in enumerate(list_data):
            # 已經有新聞資料，就不再走訪
            if 'content' in list_data[index]: continue

            # 請求頁面
            res = req.get(url = list_data[index]['url'], headers = get_user_agent())

            # 請求成功的處理流程
            if res.status_code == req.codes.ok:
                # 取得 html parser 物件
                soup = bs(res.text, "lxml")

                # 預先定義新聞內文的 css 選擇器
                list_css_selector = [
                    'td[name="text-area"]',
                    'div.post_details',
                ]

                # 只要 css 選擇器指定元素存在，則取得內文
                for str_css_selector in list_css_selector:
                    if len( soup.select(str_css_selector) ) > 0:
                        list_data[index]['content'] = soup.select_one(str_css_selector).get_text()
                        break

                # 隨機等待 (politeness policy)
                sleep( randint(2, 5) )
            else:
                print("網址請求失敗…")
                print(f"title: {list_data[index]['newTitle']}")
                print(f"url: {list_data[index]['url']}")
                print(f"newId: {list_data[index]['newId']}")
                print("結束執行")
                break

        # 另存 json
        with open("kingnet_news_list_conv.json", "w", encoding="utf-8") as file:
            file.write( json.dumps(list_data, ensure_ascii=False) )
    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(str(ex))

'''
主程式
'''
if __name__ == "__main__":
    time_b = time()


    # 變數初始化
    init()

    # 原始資料擷取
    get_total_pages()
    get_page_data()
    save_json()
    
    # 資料處理與文章擷取
    convert_encoded_text()
    # get_news_text()


    print(f"原始資料擷取時間: {time() - time_b} 秒")