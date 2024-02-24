import os
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader as pdf_loader
# from langchain.document_loaders import PyPDFLoader as pdf_loader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from 枪版GPT import chatgpt
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





def promote_query(query, gtp_expand=False):
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
            "Enclose your response within a square bracket '[]'."

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

    return filal


def chatgpt_summary(title, content):
    # 调用ChatGPT进行查询语句的优化
    instruction = (f"I have a list that contains multiple texts about {title}. "
                   f"Please help me summarize these contents, distilling the main points and integrating them into a coherent paragraph. "
                   f"Here are the contents of the list: {content}.")
    resp = chatgpt(instruction)
    Chinese = chatgpt(f"将下文翻译为中文：{resp}")
    return Chinese

