import os
import pickle
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from inputimeout import inputimeout, TimeoutOccurred
from angle_emb import AnglE, Prompts
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from .utils import load_docs, wait_for_memory, promote_query, has_chinese, chatgpt_summary, promote_query_notranslate
import torch
import time
from flask_socketio import SocketIO
# embedding_model = "/mnt/mydisk/WhereIsAI-UAE-Large-V1"
# encoding_save_path = '/mnt/mydisk/EJOR_angleUAE_EachPdf'
# gpt_model = '/mnt/mydisk/xverse-XVERSE-7B'

class UAE_query:
    def __init__(self, embedding_model = "/mnt/mydisk/WhereIsAI-UAE-Large-V1", encoding_save_path='/mnt/mydisk/EJOR_angleUAE_EachPdf', gpt_model='/mnt/mydisk/xverse-XVERSE-7B', use_xv=False):
        self.use_xv = use_xv
        self.angle = AnglE.from_pretrained(embedding_model, pooling_strategy='cls').cuda()
        self.angle.set_prompt(prompt=Prompts.C)
        self.encoding_save_path = encoding_save_path
        self.enbedding_files = os.listdir(encoding_save_path)
        if self.use_xv:
            self.tokenizer = AutoTokenizer.from_pretrained(gpt_model)
            self.model = AutoModelForCausalLM.from_pretrained(gpt_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
            self.model = self.model.quantize(8).cuda()
            self.model.generation_config = GenerationConfig.from_pretrained(gpt_model)
            self.model = self.model.eval()


    def myLLM(self, input_context):
        wait_for_memory(threshold_gb=10)
        history = [{"role": "user", "content": input_context}]
        while True:
            print(f'问题:{input_context}')
            response = self.model.chat(self.tokenizer, history)
            self.output_queue(f'答案:{response}')
            try:
                check_ = inputimeout(prompt="是否对答案满意？请回答{是/否/手动输入答案}", timeout=0.1)
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

    def queries_vecs_and_compute_distance(self, or_querys):
        assert type(or_querys) == str
        self.uselogs = '/home/gang/Documents/AI_article/web_server/uselogs' + str(int(time.time()))
        if not os.path.exists(self.uselogs):
            os.makedirs(self.uselogs)
        # or_querys不同的问题以//分割
        # or_querys = '气候风险可以分为两大类，物理风险和转型风险。气候风险会对公司经营产生影响，相关的实证研究有哪些？'//'基于文本挖掘或公司年报、新闻、媒体文本，来做金融或经济学的研究'
        self.or_querys = or_querys.split('//')
        self.query_nums = len(self.or_querys)
        querys = promote_query_notranslate(self.or_querys, llm_func=None)
        for query in querys:
            if has_chinese(query):
                raise Exception('提示词不准有汉语！')
        print('=================开始检索文献=================')
        self.query_vec = self.angle.encode([{'text': query} for query in querys], to_numpy=True)
        # TODO 针对query进行拓展，优化
        # 对所有的pdf，遍历，计算距离
        self.dist_pdf = []
        # 创建 tqdm 进度条对象
        for enbedding_file in tqdm(self.enbedding_files, desc='Processing Embedding Files', unit='file'):
            # 现在是逐篇文献对比
            # 打开文件并加载数据
            with open(f'{os.path.join(self.encoding_save_path, enbedding_file)}', 'rb') as file:
                metadata = pickle.load(file)
            # 打印加载的数据，确认内容
            anglevecs = metadata['embedding']
            cos_sim = cosine_similarity(anglevecs, self.query_vec)
            cos_sim = cos_sim.T
            cos_distance = 1 - cos_sim
            sorted_indices_row = np.argsort(cos_distance, axis=1)  # sorted_indices 中的第一个元素对应的是原始数组中的最小值的索引
            # 通过索引数组获取从小到大排序的数组
            cos_distance = np.array([row[idx] for row, idx in zip(cos_distance, sorted_indices_row)])
            # 余弦距离
            distances_mean = np.mean(cos_distance[:, :5], axis=1)  # 取其前5个最相关的片段的，距离平均值
            self.dist_pdf.append((distances_mean, enbedding_file, sorted_indices_row))
    def sort_article_and_summary(self):
        K = 5  # 取出前K篇文献进行细致分析
        # %% 根据元组的第一个值排序，从小到大
        all_dfs = []
        for i in range(self.query_nums):
            sorted_dist_pdf = sorted(self.dist_pdf, key=lambda x: x[0][i])  # x[0]是距离向量，考虑第i个问题，进行排序
            # 只取每个元组的前两个元素
            a_filtered = [(x[0][i], x[1]) for x in sorted_dist_pdf]
            sorted_dist_pdf_df = pd.DataFrame(a_filtered, columns=['distance', 'pdf_name'])
            self.name_ = re.sub(r'[^\w\s\d]|_', '', self.or_querys[i])
            sorted_dist_pdf_df.to_excel(f'{self.uselogs}/相关文献_{self.name_}.xlsx')
            # 根据排序结果取出前K篇进行二次检索，找出相关段落
            needed = sorted_dist_pdf[:K]
            needed_file_distIndices = [(s[1], s[2][i]) for s in needed]

            #
            self.querys_contents = []
            for enbedding_file, dist_indices in needed_file_distIndices:
                pdf_name = enbedding_file.split('.')[0]
                with open(f'{os.path.join(self.encoding_save_path, enbedding_file)}', 'rb') as file:
                    metadata = pickle.load(file)
                results_contents = metadata['text_list']
                # 根据自定义顺序排序列表
                top = 5  # 每篇文章取前5个片段，进行后续总结
                assert dist_indices.ndim == 1  # 有几个问题，就有几行。
                sorted_row = [results_contents[j] for j in dist_indices]
                self.querys_contents.append({pdf_name: sorted_row[:top]
                                        })


            #  TODO 逐篇根据相关段落输入至LLM进行概括，总结。完成任务：文件检索--文献概括
            print('=================开始概括文献=================')
            summarys = []
            for querys_content in self.querys_contents:
                pdf_name = list(querys_content.keys())[0]
                resp = chatgpt_summary(title=pdf_name, content=querys_content[pdf_name], llm_func=None)
                summarys.append({'title': pdf_name,
                                 'summary': resp,
                                 })
                print({'title': pdf_name,
                                 'summary': resp,
                                 })
            self.df = pd.DataFrame(summarys)
            self.df.to_excel(f'{self.uselogs}/文献总结_{self.name_}.xlsx')


# # 使用方法

#
# my_query = UAE_query(embedding_model, encoding_save_path, gpt_model)
# while True:
#     my_query.queries_vecs()
#     my_query.process_queries()
#     my_query.搜索文献并概括()
#

