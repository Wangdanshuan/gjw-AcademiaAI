# PDFMinerLoader会把整个pdf读取合并为1个大page，即，不分page；他的好处是能够分清人为分段，还是自动分段
# PyPDFLoader会按照page来读取；无法分清人为分段，还是自动分段
import os
import logging
import time

from utils import load_docs, load_embeddingModel, delete_folder_if_empty_or_single_file
from langchain.text_splitter import RecursiveCharacterTextSplitter
import copy
import re
from langchain.vectorstores import Chroma
import gc



# from langchain_community.vectorstores import Chroma
root_path = 'EJOR文献库'
# model_name ="BAAI/bge-large-en-v1.5"
model_name ="./UAE-Large-V1"
log_file = os.path.join(root_path, "log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 同一个文献库同一个方法的都在同一个大文件夹 encoding_save_path
encoding_save_path = './ejor_EachPdf_' + re.sub(r'[^a-zA-Z]+', '', model_name)
already_exists = os.listdir(encoding_save_path)
already_exists_ = copy.deepcopy(already_exists)
for file_name in already_exists_:
    flag = delete_folder_if_empty_or_single_file(encoding_save_path +'/'+ file_name)
    if flag:
        already_exists.remove(file_name)

time.sleep(3)
print(f'已经有的文献:{len(already_exists)}, \n', already_exists)
# 定义目录路径和所有文件列表（假设你已经有了这些）
embedding = load_embeddingModel(model_name)

if __name__ == "__main__":

    # 以子文件夹为单位
    dirs = os.listdir(root_path)
    for dir in dirs:
        directory_path = os.path.join(root_path, dir)
        all_docs = load_docs(directory_path, already_exists, cores=10, max_files=10000000)
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
                vectorstore_df = Chroma.from_documents(documents=document_list,  embedding=embedding,
                                                       persist_directory=persist_directory)  #  collection_name="huggingface_embed"
                vectorstore_df.persist()
                # 循环结束后
                vectorstore_df = None
                gc.collect()
                # Now we can load the persisted database from disk, and use it as normal.
                logging.info("该pdf完成编码: %s", pdf_name)
