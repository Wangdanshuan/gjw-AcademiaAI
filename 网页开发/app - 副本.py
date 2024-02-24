from flask import Flask, request, jsonify, render_template  # 引入 render_template
import subprocess
from UAE_query_llm_web import UAE_query  # 导入你的 MyLLM 类

app = Flask(__name__)
# 初始化 MyLLM 类的实例一次
model_name = "/mnt/mydisk/WhereIsAI-UAE-Large-V1"
encoding_save_path = '/mnt/mydisk/EJOR_angleUAE_EachPdf'
model_path = '/mnt/mydisk/xverse-XVERSE-7B'
my_query = UAE_query(model_name, encoding_save_path, model_path)
# 初始化你的 Python 应用程序一次

@app.route('/')
def index():
    return "Welcome to my Python application!"

@app.route('/chat')
def chat():
    return render_template('chat.html')  # 返回 chat.html 模板

@app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    input_text = data.get('input_text')

    # 使用 MyLLM 实例处理输入并生成回应
    try:
        # generated_response = myllm_instance.generate_response(input_text)
        my_query.queries_vecs_and_compute_distance(input_text)
        generated_response = my_query.sort_article_and_summary()


        response = {
            'result': generated_response,
            'error': None
        }
    except Exception as e:
        response = {
            'result': None,
            'error': str(e)
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
