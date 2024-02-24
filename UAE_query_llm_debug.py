# PDFMinerLoader会把整个pdf读取合并为1个大page，即，不分page；他的好处是能够分清人为分段，还是自动分段
# PyPDFLoader会按照page来读取；无法分清人为分段，还是自动分段
import os
from inputimeout import inputimeout, TimeoutOccurred
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
import pandas as pd
import numpy as np
from utils import load_docs, wait_for_memory, promote_query, has_chinese, chatgpt_summary, promote_query_notranslate
import re
from collections import Counter
from angle_emb import AnglE, Prompts
import pickle
model_name ="./UAE-Large-V1"
angle = AnglE.from_pretrained(model_name, pooling_strategy='cls').cuda()
angle.set_prompt(prompt=Prompts.C)
encoding_save_path = './EJOR_angleUAE_EachPdf'
enbedding_files = os.listdir(encoding_save_path)
# 定义目录路径和所有文件列表（假设你已经有了这些）

# ----------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = '/mnt/mydisk/xverse-XVERSE-7B'
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
# model = model.quantize(8).cuda()
# # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
# model.generation_config = GenerationConfig.from_pretrained(model_path)
# model = model.eval()
#
#

def myLLM(input_context):
    wait_for_memory(threshold_gb=10)
    history = [{"role": "user", "content": input_context}]
    while True:
        print(f'问题:{input_context}')
        response = model.chat(tokenizer, history)
        print(f'答案:{response}')
        try:
            check_ = inputimeout(prompt="是否对答案满意？请回答{是/否/手动输入答案}", timeout=5)
        except TimeoutOccurred:
            check_ = '是'

        if check_ == '是':
            break
        elif check_ == '否':
            pass
        else:
            response = check_
            break
    return response


# ----------------------------------------
if __name__ == "__main__":

    or_querys = ['气候风险可以分为两大类，物理风险和转型风险。气候风险会对公司经营产生影响，相关的实证研究有哪些？',
                 '基于文本挖掘或公司年报、新闻、媒体文本，来做金融或经济学的研究']
    query_nums = len(or_querys)
    querys = promote_query_notranslate(or_querys, llm_func=None)
    for query in querys:
        if has_chinese(query):
            raise Exception('提示词不准有汉语！')
    print('=================开始检索文献=================')
    query_vec = angle.encode([{'text': query} for query in querys], to_numpy=True)
    K = 10  # 取出前K篇文献进行细致分析
    # TODO 针对query进行拓展，优化
    # 对所有的pdf，遍历，计算距离
    dist_pdf = []
    for enbedding_file in enbedding_files:
        # 现在是逐篇文献对比
        # 打开文件并加载数据
        with open(f'{os.path.join(encoding_save_path, enbedding_file)}', 'rb') as file:
            metadata = pickle.load(file)
        # 打印加载的数据，确认内容
        anglevecs = metadata['embedding']
        cos_sim = cosine_similarity(anglevecs, query_vec)
        cos_sim = cos_sim.T
        cos_distance = 1 - cos_sim
        sorted_indices_row = np.argsort(cos_distance, axis=1)  # sorted_indices 中的第一个元素对应的是原始数组中的最小值的索引
        # 通过索引数组获取从小到大排序的数组
        cos_distance = np.array([row[idx] for row, idx in zip(cos_distance, sorted_indices_row)])
        # 余弦距离
        distances_mean = np.mean(cos_distance[:,:5], axis=1)  # 取其前5个最相关的片段的，距离平均值
        dist_pdf.append((distances_mean, enbedding_file, sorted_indices_row))
    # %% 根据元组的第一个值排序，从小到大
    for i in range(query_nums):
        sorted_dist_pdf = sorted(dist_pdf, key=lambda x: x[0][i])  # x[0]是距离向量，考虑第i个问题，进行排序
        # 只取每个元组的前两个元素
        a_filtered = [(x[0][i], x[1]) for x in sorted_dist_pdf]
        sorted_dist_pdf_df = pd.DataFrame(a_filtered, columns=['distance', 'pdf_name'])
        name_ = re.sub(r'[^\w\s\d]|_', '', or_querys[i])
        sorted_dist_pdf_df.to_excel(f'相关文献_{name_}.xlsx')
        # 根据排序结果取出前K篇进行二次检索，找出相关段落
        needed = sorted_dist_pdf[:K]
        needed_file_distIndices = [(s[1], s[2][i]) for s in needed]

        #
        querys_contents = []
        for enbedding_file, dist_indices in needed_file_distIndices:
            pdf_name = enbedding_file.split('.')[0]
            with open(f'{os.path.join(encoding_save_path, enbedding_file)}', 'rb') as file:
                metadata = pickle.load(file)
            results_contents = metadata['text_list']
            # 根据自定义顺序排序列表
            top = 5  # 每篇文章取前5个片段，进行后续总结
            assert dist_indices.ndim == 1   # 有几个问题，就有几行。
            sorted_row = [results_contents[j] for j in dist_indices]
            querys_contents.append({pdf_name: sorted_row[:top]
             })

        #  TODO 逐篇根据相关段落输入至LLM进行概括，总结。完成任务：文件检索--文献概括
        print('=================开始概括文献=================')
        summarys = []
        for querys_content in querys_contents:
            pdf_name = list(querys_content.keys())[0]
            resp = chatgpt_summary(title=pdf_name, content=querys_content[pdf_name], llm_func=None)
            summarys.append({'title':pdf_name,
                             'summary':resp,
                             })
        print(summarys)
        df = pd.DataFrame(summarys)
        df.to_excel(f'回答_{name_}.xlsx')




