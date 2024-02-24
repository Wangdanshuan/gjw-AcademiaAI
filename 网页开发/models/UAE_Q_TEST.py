import time

class UAE_query:
    def __init__(self):
        pass
    def queries_vecs_and_compute_distance(self, input_text):
        # 模拟处理输入并计算距离的过程
        time.sleep(2)  # 模拟耗时操作
        print("Processing input:", input_text)
        self.input_text = input_text
        # 模拟距离计算结果
        distance_result = "Distance calculation result for: " + input_text
        yield distance_result

    def sort_article_and_summary(self):
        # 模拟排序和摘要生成的过程
        for i in range(5):
            time.sleep(3)  # 模拟耗时操作
            message = "Step " + str(i) + self.input_text
            yield message

        # 模拟排序和摘要生成结果
        summary_result = "Finish summary of sorted articles"
        yield summary_result
