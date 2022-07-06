// 送出查詢
$(document).on('click', 'button#btn_query', function (event) {
    event.preventDefault();

    //設定時間變數
    timeStart = timeEnd = 0

    //計算開始時間
    timeStart = Date.now() / 1000

    //取得文字並要求輸入字數
    let sentences = $('textarea#search_bar').val();
    if (sentences.length === 0) { alert('請輸入文字'); return false; }

    //取得 tbody 元素
    let tbody_result = $('tbody#result');

    //預設彈跳訊息
    let html_msg = `<h1 id="msg">請稍候</h1>`

    //浮現 blockUI
    $.blockUI({ 
        message: html_msg, 
        css: { 
            border: 'none', 
            padding: '15px', 
            backgroundColor: '#000', 
            '-webkit-border-radius': '10px', 
            '-moz-border-radius': '10px', 
            opacity: .5, 
            color: '#fff' 
        }
    });

    //預測句子
    $.post(`./predict`, { sentences: sentences }, (obj) => {
        //請求失敗的訊息
        if( !obj['success'] ){ alert(`${obj['info']}`); return false; }

        //清空 tbody 內文
        tbody_result.html('');

        //放置自動產生訓練資料用的字串變數
        let str_json_lines = '';
        
        //整合查詢結果
        let sn = 1
        for (let arr of obj['results']) {
            console.log(arr);

            //取得初始化 json，並對 id 給予 uuid
            let obj_json = get_init_json();
            obj_json.id = get_uuid();

            // 整合 html 資料
            let html = str_sentence = ``;
            for(let o of arr){
                for(let char in o){
                    if( o[char] == 'O' ){
                        html += `${char}`;
                    } else {
                        html += `<mark data-entity="${o[char]}">${char}</mark>`;
                    }

                    //合併 sentence
                    str_sentence += char;

                    //加入 sentence、character 與 character_based 資訊
                    obj_json.character.push(char);
                    obj_json.character_label.push(o[char]);
                }
            }

            //放置 sentence
            obj_json.sentence = str_sentence;
            
            //呈現 table 資訊
            tbody_result.append(`<tr>
                <th scope="row">${sn}</th>
                <td class="entities">${html}</td>
            </tr>`);

            //流水號遞增
            sn++;

            //整合 json 字串
            str_json_lines += JSON.stringify(obj_json) + "\n";
        }

        // 將 json 放入文字欄位
        $('textarea#export_json').val(str_json_lines);

        // 關閉 blockUI
        $.unblockUI();

        //計算結束時間
        timeEnd = Date.now() / 1000

        // //計算執行花費時間，顯示結果
        // let div_msg = document.getElementById('liveToast');
        // div_msg.querySelector('div.toast-body').innerText = `執行時間: ${timeEnd - timeStart} 秒`
        // let toast = new bootstrap.Toast( div_msg );
        // toast.show();

        //執行花費時間顯示在按鈕旁邊
        $('span#exec_secs').text(`${timeEnd - timeStart}`);
    });
});

//下載 json 檔案
$(document).on('click', 'button#btn_download_json', function(event){
    //取得 json 字串
    let str_json_lines = $('textarea#export_json').val();
    if (str_json_lines.length == 0) { alert(`請先進行預測`); return false; }

    //下載 json 檔案
    download('training_data.json', str_json_lines);
});

//下載 json 檔案
function download(filename, text) {
    let element = document.createElement('a');
    element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

//將初始化的 json 物件回傳
function get_init_json(){
    return {
        "id": "", 
        "genre": "custom", 
        "sentence": "", 
        "word": null, 
        "word_label": null, 
        "character": [], 
        "character_label": []
    };
}

//取得 uuid
function get_uuid() {
    let d = Date.now();
    if (typeof performance !== 'undefined' && typeof performance.now === 'function'){
        d += performance.now(); //use high-precision timer if available
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        let r = (d + Math.random() * 16) % 16 | 0;
        d = Math.floor(d / 16);
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}