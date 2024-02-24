# PDFMinerLoader会把整个pdf读取合并为1个大page，即，不分page；他的好处是能够分清人为分段，还是自动分段
# PyPDFLoader会按照page来读取；无法分清人为分段，还是自动分段
from utils import load_docs, load_embeddingModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma

directory_path = './EJOR文献库/ScienceDirect_articles_23Jan2024_14-34-28.966'
# model_name ="BAAI/bge-large-en-v1.5"
model_name ="./UAE-Large-V1"
persist_directory = './ejor_' + re.sub(r'[^a-zA-Z]+', '', model_name) + '_2'

if __name__ == "__main__":
    # 定义目录路径和所有文件列表（假设你已经有了这些）
    embedding = load_embeddingModel(model_name)
    all_docs = load_docs(directory_path, cores=-1, max_files=2)
    print(len(all_docs))
    vectorstore_df = Chroma.from_documents(documents=all_docs,  embedding=embedding,
                                           persist_directory=persist_directory)  #  collection_name="huggingface_embed"
    vectorstore_df.persist()
    vectorstore_df = None
    # Now we can load the persisted database from disk, and use it as normal.
    vectorstore_df = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    while True:
        query = input("请输入问题: ")
        if query == "break":
            break
        # TODO 用 gpt3.5 api 来优化这个用户问题，形成新的 query
        # result = vectorstore_df.similarity_search(query, k=5)
        result = vectorstore_df.similarity_search_with_score(query)
        # TODO 根据这个结果，形成输出，比如
        #  tast（1） 搜索文献，概括文献：匹配到原文献，得出文献题目，根据这篇文献的其他切片信息，gpt总结概括
        #  tast (2) 搜索实证机制，总结所有文献中的，用到的机制。 以及输出文献题目
        #  tast (3)
        print(result)


