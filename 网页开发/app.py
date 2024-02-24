from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from models.UAE_query_llm_web import UAE_query


my_query = UAE_query()
app = Flask(__name__)
# 配置一个文件夹，包含您的 Excel 文件


# 初始化你的 Python 应用程序一次

@app.route('/')
def index():
    return "Welcome to my Python application!"

@app.route('/chat')
def chat():
    return render_template('chat.html')  # 返回 chat.html 模板


@app.route('/execute', methods=['POST'])
def execute():
    try:


        data = request.json
        input_text = data.get('input_text')
        my_query.queries_vecs_and_compute_distance(input_text)
        my_query.sort_article_and_summary()
        # 处理执行结果
        result_text = f'分析完毕！'
        excel_folder = my_query.uselogs
        app.config['UPLOAD_FOLDER'] = excel_folder
        # 创建四个不同的 Excel 文件并将结果写入
        excel_files = os.listdir(excel_folder)
        # 返回文件下载链接
        download_links = [f'/download/{file}' for file in excel_files]

        # 返回包含结果和下载链接的响应
        return jsonify({'result': result_text, 'download_links': download_links})
    except Exception as e:
        # 发生异常时返回错误消息
        return jsonify({'error': str(e)})


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
