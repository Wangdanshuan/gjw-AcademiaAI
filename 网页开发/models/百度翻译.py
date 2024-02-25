import pandas as pd
import requests
import hashlib
import random
import json
from retry import retry

# 定义重试装饰器
@retry(exceptions=requests.exceptions.ConnectionError, tries=3, delay=1)
def make_request(url, params):
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response

# 设置API密钥和APP ID
API_KEY =  ''
APP_ID = ''


def translate_text(text,  from_lang = 'zh', to_lang = 'en'):

    # 构造请求 URL 和参数
    print(text)
    url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    salt = str(random.randint(32768, 65536))
    sign = hashlib.md5(f"{APP_ID}{text}{salt}{API_KEY}".encode('utf-8')).hexdigest()
    params = {
        "q": text,
        "from": from_lang,
        "to": to_lang,
        "appid": APP_ID,
        "salt": salt,
        "sign": sign
    }

    # 发送请求并解析响应
    response = make_request(url, params)
    # response = requests.get(url, params=params)

    result = json.loads(response.content.decode())
    # 提取翻译结果
    try:
        translation = result['trans_result'][0]['dst']
    except Exception as e:
        print(e)
        print(result)
    return translation

if __name__ == "__main__":
    result = translate_text('你很骄傲啊')
    print(result)