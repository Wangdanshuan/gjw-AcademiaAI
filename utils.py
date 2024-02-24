import os
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from multiprocessing import Pool, cpu_count
from langchain.document_loaders import PDFMinerLoader as pdf_loader
# from langchain.document_loaders import PyPDFLoader as pdf_loader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from 枪版GPT import chatgpt as gpt
import shutil
import os
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 指定要读取的目录路径
# directory_path = './EJOR文献库'
def list_files(directory_path):
    # 使用os.walk()遍历目录及其子目录
    all_fils = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.pdf'):
                # 获取PDF文件的完整路径
                pdf_path = os.path.join(root, filename)
                all_fils.append(pdf_path.replace('\\', '/'))
    return all_fils
# all_fils = list_files(directory_path)
# 定义处理单个文件的函数
def process_file(file_name):
    loader = pdf_loader(file_name)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    docs = text_splitter.split_documents(pages)
    print('处理完毕:', file_name)
    return docs


import torch
import time

def wait_for_memory(threshold_gb=3, check_interval=5):
    """
    等待直到CUDA内存使用降至阈值以下。

    :param threshold_gb: 内存使用阈值（GB）
    :param check_interval: 检查内存使用情况的间隔时间（秒）
    """
    torch.cuda.empty_cache()
    threshold_bytes = threshold_gb * (1024 ** 3)  # 将GB转换为字节

    while True:
        # 获取当前CUDA内存使用情况
        allocated = torch.cuda.memory_allocated()
        max_allocated = torch.cuda.max_memory_allocated()

        print(f"当前内存使用: {allocated / (1024 ** 3):.2f} GB, 历史最高使用: {max_allocated / (1024 ** 3):.2f} GB")

        if allocated < threshold_bytes:
            print("内存使用已降至阈值以下，继续执行程序。")
            break

        time.sleep(check_interval)  # 等待一段时间后再次检查

def load_docs(directory_path, already_exists=None, cores=-1, max_files=10000000):
    all_fils = list_files(directory_path)[:max_files]
    all_fils = [f for f in all_fils if f.split('/')[-1].split('.')[0] not in already_exists]
    # 定义目录路径和所有文件列表（假设你已经有了这些）
    # 获取CPU核心数以确定要启动的进程数
    if cores == -1:
        num_processes = cpu_count()
    else:
        num_processes = cores

    # 初始化tqdm进度条
    with tqdm(total=len(all_fils), desc='Processing files', unit='file') as pbar:
        all_docs = []
        # 使用进程池并行处理文件
        with Pool(num_processes) as pool:
            results = list(pool.map(process_file, all_fils))

        # 汇总所有处理结果
        for docs in results:
            all_docs += docs
            pbar.update(1)

    # 进度条会自动完成，不需要额外的操作
    return all_docs

def load_embeddingModel(model_name):
    # UAE-Large-V1这个模型输入最大长度  512
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    print('===============加载完embedding模型============')
    return embedding



def delete_folder_if_empty_or_single_file(folder_path):
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        print('len(files):', len(files))
        if len(files) <= 1:
            shutil.rmtree(folder_path)
            print(f"文件夹 '{folder_path}' 及其内容已被删除")
            return True
        else:
            print(f"文件夹 '{folder_path}' 包含多个文件或目录，未删除")
            return False



def has_chinese(text):
    # 使用正则表达式来匹配中文字符
    pattern = re.compile(r'[\u4e00-\u9fa5]')

    # 使用re.search来查找匹配项
    if re.search(pattern, text):
        return True
    else:
        return False


from 百度翻译 import translate_text
def promote_query_notranslate(querys, llm_func=None, gtp_expand=False):
    if llm_func is not None:
        chatgpt = llm_func
    else:
        chatgpt = gpt

    t_q = []
    for query in querys:
        q_ = translate_text(query, 'zh', 'en')
        t_q.append(q_)



    if gtp_expand:
        finals = []
        for query in t_q:
        # 传入的查询语句
        # 输出提取的部分
            # 构造通用的instruction
            instruction = (
                "Please optimize and enrich the following query for a more comprehensive and accurate search "
                "within a knowledge base. The query is: '{}'. Refine the query to ensure it is "
                "clear, concise, and specifically tailored to retrieve relevant information from "
                "a knowledge base. "

            ).format(query)

            # 调用ChatGPT进行查询语句的优化
            resp = chatgpt(instruction)
            print('gpt的回复:', resp)
            filal = query + resp
            finals.append(filal)
    else:
        finals = t_q
    return finals

def promote_query(querys, llm_func=None, gtp_expand=False):
    if llm_func is not None:
        chatgpt = llm_func
    else:
        chatgpt = gpt
    finals = []
    for query in querys:
        # 传入的查询语句
        # query = '...' # 示例: '气候风险可以分为两大类，物理风险和转型风险。气候风险会对公司经营产生影响'
        Englishquery = chatgpt(f"将下文翻译为英文，并且将翻译的答案以符号“[]”包裹，需要翻译的文本如下：{query}")
        # 使用正则表达式提取以方括号包裹的部分
        Englishquerymatches = re.findall(r'\[([^\]]+)\]', Englishquery)
        # 输出提取的部分
        if type(Englishquerymatches) == list:
            Englishquerymatches = ', '.join(Englishquerymatches)
        print('翻译并提取后的原始提示词:', Englishquerymatches)
        optimized_query = ' '
        if gtp_expand:
            # 构造通用的instruction
            instruction = (
                "Please optimize and enrich the following query for a more comprehensive and accurate search "
                "within a knowledge base. The query is: '{}'. Refine the query to ensure it is "
                "clear, concise, and specifically tailored to retrieve relevant information from "
                "a knowledge base. Include key terms and concepts that are essential for an effective search. "
                "[To request that your response be enclosed in square brackets]."

            ).format(Englishquerymatches)

            # 调用ChatGPT进行查询语句的优化
            resp = chatgpt(instruction)
            print('gpt的回复:', resp)
            # 从ChatGPT的响应中提取优化后的查询语句
            optimized_query = re.findall(r'\[([^\]]+)\]', resp)
            # 输出提取的部分
            if type(optimized_query) == list:
                optimized_query = ', '.join(optimized_query)
            print('gpt拓展的提示词:', optimized_query)
        filal = Englishquerymatches + optimized_query
        finals.append(filal)
    return finals


def chatgpt_summary(title, content, llm_func=None):
    if llm_func is not None:
        chatgpt = llm_func
    else:
        chatgpt = gpt
    # 调用ChatGPT进行查询语句的优化
    instruction = (f"I have a list that contains multiple text fragments from the article titled '{title}'. "
                   f"Please help me summarize these fragments, distilling the main points and integrating them "
                   f"into a coherent paragraph. Here are the contents of the list: {content}.")
    resp = chatgpt(instruction)
    # Chinese = chatgpt(f"将下文翻译为中文：{resp}")
    Chinese = translate_text(resp, "en", "zh")
    return Chinese

