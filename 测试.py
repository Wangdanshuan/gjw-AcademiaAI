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
from angle_emb import AnglE, Prompts
# from langchain_community.vectorstores import Chroma
from angle_emb import AnglE, Prompts
angle = AnglE.from_pretrained('C:/Users/Administrator/.cache/torch/sentence_transformers/WhereIsAI_UAE-Large-V1', pooling_strategy='cls').cuda()
angle.set_prompt(prompt=Prompts.C)

root_path = 'ejor_EachPdf_UAELargeV'
# model_name ="BAAI/bge-large-en-v1.5"
model_name = "./UAE-Large-V1"
# 定义目录路径和所有文件列表（假设你已经有了这些）
embedding = load_embeddingModel(model_name)
persist_directory = 'ejor_UAELargeV'
vectorstore_df = Chroma(persist_directory=persist_directory, embedding_function=embedding)
query = 'appropriate  αk ∈ R m to accelerate  convergence.  In light of (18) , ev- \nery maximum  term can be rewritten  as \n⟨∇F i (x k ) , d ⟩ + αk \ni \n2 ∥ d ∥ 2 . (19) \nIt is well-known  by using αk \ni Ias an approximation  of ∇ 2 F i (x k ) so \nthat one possible  use for αk could be given by Barzilai-Borwein  \nMethod.  \n4.1. Barzilai-Borwein  descent  method  for MOPs \nFirstly, we provide  a brief introduction  of Barzilai-Borwein’s'

result = vectorstore_df.similarity_search_with_score(query, k=120)  # k要与基础库的大小相匹配
dists = [r[-1] for r in result]
results_contents = [r[0].page_content for r in result]




anglevecs = angle.encode([{'text': cont} for cont in results_contents] , to_numpy=True)
angl_query = angle.encode({'text': query}, to_numpy=True)

dist2 = np.sum((anglevecs - angl_query)**2, axis=1)

from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(anglevecs, angl_query)
# 余弦距离
cos_distance = 1 - cos_sim
print(np.argsort(cos_distance.reshape(-1)))
print(np.argsort(dist2))