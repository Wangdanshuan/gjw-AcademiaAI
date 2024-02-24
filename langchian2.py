# PDFMinerLoader会把整个pdf读取合并为1个大page，即，不分page；他的好处是能够分清人为分段，还是自动分段
# PyPDFLoader会按照page来读取；无法分清人为分段，还是自动分段
import os
import logging
from utils import load_docs, load_embeddingModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
root_path = 'EJOR文献库'
# model_name ="BAAI/bge-large-en-v1.5"
model_name ="./UAE-Large-V1"
log_file = os.path.join(root_path, "log.txt")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# 定义目录路径和所有文件列表（假设你已经有了这些）
embedding = load_embeddingModel(model_name)

if __name__ == "__main__":

    # 以子文件夹为单位
    dirs = os.listdir(root_path)
    for dir in dirs:
        directory_path = os.path.join(root_path, dir)
        all_docs = load_docs(directory_path, cores=10, max_files=10000)
        if len(all_docs)>1:
            print(directory_path, '正在处理的句子数:', len(all_docs))
            persist_directory = './ejor_' + re.sub(r'[^a-zA-Z]+', '', model_name) + f'/{dir}_encoding'
            vectorstore_df = Chroma.from_documents(documents=all_docs,  embedding=embedding,
                                                   persist_directory=persist_directory)  #  collection_name="huggingface_embed"
            vectorstore_df.persist()
            vectorstore_df = None
            # Now we can load the persisted database from disk, and use it as normal.
            logging.info("该路径完成编码: %s", directory_path)
