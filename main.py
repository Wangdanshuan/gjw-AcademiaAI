# 现在你可以继续使用WebDriver来管理ChatGPT网页
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from numpy import random
from time import sleep
import subprocess
import os
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# 确定 chrome.exe 路径，给chrome_path

default_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

import numpy as np
def check_chrome_path(path):
    if os.path.isfile(path):
        print("Chrome.exe 路径:", path)
        return path
    else:
        print("默认路径中未找到 Chrome.exe")
        search_path = r"C:\Program Files"   # 一定要把谷歌安装到C:\Program Files路径下
        chrome_path = find_chrome_path(search_path)
        if chrome_path:
            print("Chrome.exe 路径:", chrome_path)
            return chrome_path
        else:
            error_message = "未安装谷歌浏览器"
            print(error_message)
            raise ValueError(error_message)


def find_chrome_path(folder_path):
    for root, dirs, files in os.walk(folder_path):
        if "chrome.exe" in files:
            return os.path.join(root, "chrome.exe")
    return None


chrome_path = check_chrome_path(default_path)

remote_debugging_port = 9222
user_data_dir = r"C:\temp\chrome_dev_test1"
# 构造命令行命令
command = f'"{chrome_path}" --remote-debugging-port={remote_debugging_port} --user-data-dir="{user_data_dir}"'
# 执行命令行命令
subprocess.Popen(command, shell=True)
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=chrome_options)

# 期刊页面的 URL
def download_file(vol, iss):
    url = f"https://www.sciencedirect.com/journal/european-journal-of-operational-research/vol/{vol}/issue/{iss}"
    driver.get(url)
    sleep(random.uniform(4, 8))  # 随机等待时间，减少被检测风险


    # 使用XPath定位按钮
    button = driver.find_element(By.XPATH, "//button[contains(@class, 'button-link') and contains(., 'Download full issue')]")

    # 点击按钮
    button.click()


import logging

# 配置日志记录器
logging.basicConfig(filename='download_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 用于保存成功的(vol, iss)的列表
success_list = []

# 读取已经成功的记录并保存到success_list中
with open('download_log.txt', 'r') as log_file:
    for line in log_file:
        if '下载成功' in line:
            # 提取(vol, iss)信息并添加到success_list中
            parts = line.split()
            if len(parts) >= 4:
                vol = int(parts[3])
                iss = int(parts[4])
                success_list.append((vol, iss))

for vol in range(314, 0, -1):
    for iss in (1, 4):
        # 检查(vol, iss)是否已经成功爬取过，如果是则跳过
        if (vol, iss) in success_list:
            continue
        try:
            download_file(vol, iss)
            logging.info(f'下载成功 {vol} {iss}')
        except Exception as e:
            logging.error(f'下载失败 {vol} {iss}: {str(e)}')
        sleep(5)
