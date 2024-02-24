# PDFMinerLoader会把整个pdf读取合并为1个大page，即，不分page；他的好处是能够分清人为分段，还是自动分段
# PyPDFLoader会按照page来读取；无法分清人为分段，还是自动分段
import os
import numpy as np
import time
from utils import load_docs, load_embeddingModel, delete_folder_if_empty_or_single_file, wait_for_memory
import logging
import traceback
import torch
import re
from langchain.vectorstores import Chroma
import gc
from angle_emb import AnglE, Prompts
import pickle
model_name ="./UAE-Large-V1"
angle = AnglE.from_pretrained(model_name, pooling_strategy='cls').cuda()
angle.set_prompt(prompt=Prompts.C)
# from langchain_community.vectorstores import Chroma
root_path = 'EJOR文献库'
# model_name ="BAAI/bge-large-en-v1.5"

logging.basicConfig(filename="./error_log.txt", level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# 同一个文献库同一个方法的都在同一个大文件夹 encoding_save_path
encoding_save_path = './EJOR_angleUAE_EachPdf'
if not os.path.exists(encoding_save_path):
    os.makedirs(encoding_save_path)
already_exists = os.listdir(encoding_save_path)
already_exists = [f.replace('.pkl', '') for f in already_exists]
time.sleep(3)
print(f'已经有的文献:{len(already_exists)}, \n', already_exists)
# 定义目录路径和所有文件列表（假设你已经有了这些）

if __name__ == "__main__":

    # 以子文件夹为单位
    dirs = os.listdir(root_path)
    for dir in dirs:
        directory_path = os.path.join(root_path, dir)
        all_docs = load_docs(directory_path, already_exists, cores=2, max_files=2)
        if len(all_docs)>1:
            print(directory_path, '正在处理的句子数:', len(all_docs))
            # 创建一个空字典来存储分组后的文档
            document_groups = {}
            # 遍历 all_docs 列表并将文档按照 source 分组
            for doc in all_docs:
                source = doc.metadata['source']
                if source not in document_groups:
                    document_groups[source] = []  # 创建一个新的列表
                document_groups[source].append(doc)

            # 现在，document_groups 字典包含了按照 source 分组的文档列表
            for source in document_groups.keys():
                pdf_name = source.split('/')[-1].split('.')[0]
                document_list = document_groups[source]
                # 在这里进行您的操作，例如输出文档列表的长度
                print(f"Source: {source}, Number of Documents: {len(document_list)}")
                persist_directory = encoding_save_path + f'/{pdf_name}'
                contents = [doc.page_content for doc in document_list]  # 字符串列表
                # --------------------
                batch_size = 128  # 你可能需要根据你的GPU内存调整这个数字
                anglevecs = []
                try:
                    for i in range(0, len(contents), batch_size):
                        batch_contents = contents[i:i + batch_size]
                        batch_anglevecs = angle.encode([{'text': cont} for cont in batch_contents], to_numpy=True)
                        if i == 0:
                            anglevecs = batch_anglevecs
                        else:
                            anglevecs = np.concatenate((anglevecs, batch_anglevecs), axis=0)

                        # 使用示例
                        wait_for_memory()


                except Exception as e:
                    # 记录错误信息到本地文件，包括堆栈跟踪
                    logging.error("发生错误: %s", str(e))
                    traceback_str = traceback.format_exc()
                    logging.error("Traceback:\n%s", traceback_str)
                    logging.info("异常处理完成")
                # ----------------------
                metadata = {
                    "embedding": anglevecs,
                    "text_list": contents,
                    "source": source,
                }
                # 将数据保存到文件

                with open(f'{persist_directory}.pkl', 'wb') as file:
                    pickle.dump(metadata, file)
                # Now we can load the persisted database from disk, and use it as normal.
                logging.info("该pdf完成编码: %s", pdf_name)
    print('程序执行完毕!!!!!!!!!!!!!!!!!!')
