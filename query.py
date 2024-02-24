# PDFMinerLoader会把整个pdf读取合并为1个大page，即，不分page；他的好处是能够分清人为分段，还是自动分段
# PyPDFLoader会按照page来读取；无法分清人为分段，还是自动分段
import os
import logging
import pandas as pd
import numpy as np
from utils import load_docs, load_embeddingModel, promote_query, has_chinese, chatgpt_summary
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from collections import Counter
from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
root_path = 'ejor_EachPdf_UAELargeV'
# model_name ="BAAI/bge-large-en-v1.5"
model_name ="./UAE-Large-V1"
# 定义目录路径和所有文件列表（假设你已经有了这些）
embedding = load_embeddingModel(model_name)

if __name__ == "__main__":
    or_query = '气候风险可以分为两大类，物理风险和转型风险。气候风险会对公司经营产生影响，相关的实证研究有哪些？'
    query = promote_query(or_query)
    if has_chinese(query):
        raise Exception('提示词不准有汉语！')
    K = 10  # 取出前K篇文献进行细致分析
    # TODO 针对query进行拓展，优化
    # 以子文件夹为单位
    dirs = os.listdir(root_path)
    dist_pdf = []
    for pdf_name in dirs:
        # 现在是逐篇文献对比
        persist_directory = './ejor_EachPdf_' + re.sub(r'[^a-zA-Z]+', '', model_name) + f'/{pdf_name}'
        vectorstore_df = None
        vectorstore_df = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        result = vectorstore_df.similarity_search_with_score(query, k=5)  # k要与基础库的大小相匹配
        distances_mean = np.mean([r[-1] for r in result])
        dist_pdf.append((distances_mean, pdf_name))
    # %% 根据元组的第一个值排序，从小到大
    sorted_dist_pdf = sorted(dist_pdf, key=lambda x: x[0])
    # 记录排序结果
    sorted_dist_pdf_df = pd.DataFrame(sorted_dist_pdf, columns=['distance', 'pdf_name'])
    sorted_dist_pdf_df.to_excel(f'{query}.xlsx')
    # 根据排序结果取出前K篇进行二次检索，找出相关段落
    needed = sorted_dist_pdf[:K]
    needed = [s[1] for s in needed]
    title_content = {}
    for pdf_name in needed:
        persist_directory = './ejor_EachPdf_' + re.sub(r'[^a-zA-Z]+', '', model_name) + f'/{pdf_name}'
        vectorstore_df = None

        results = vectorstore_df.similarity_search_with_score(query, k=5)  # k要与基础库的大小相匹配
        vectorstore_df.reset()
        # 这里results是个列表，其中元素result是个元组(document, distance)，对于document，例如doc=results[0][0], doc.page_content是它的句子内容
        results_contents = [result[0].page_content for result in results]
        title_content.update({pdf_name: results_contents
                               })
    #  TODO 逐篇根据相关段落输入至LLM进行概括，总结。完成任务：文件检索--文献概括
    summarys = []
    for searched_pdf in title_content.keys():
        content = title_content[searched_pdf]
        resp = chatgpt_summary(searched_pdf, content)
        summarys.append({'title':searched_pdf,
                         'summary':resp,
                         })
    print(summarys)
    df = pd.DataFrame(summarys)
    df.to_excel(f'{or_query}.xlsx')




