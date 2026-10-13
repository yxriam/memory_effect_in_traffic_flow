import json
import re
from pre_process.markov_chain import FirstOrderMarkovChain, SecondOrderMarkovChain, ThirdOrderMarkovChain
from tqdm import tqdm

# 读取机场代码和城市的对应关系
with open('../lstm_markov/airport_dict.json', 'r',
          encoding='utf-8') as f:
    airport_to_city = json.load(f)


# 将机场代码转换为城市
def airport_to_city_name(airport_code):
    return airport_to_city.get(airport_code, None)


# 创建一阶和二阶城市转移的马尔可夫链实例
first_order_city_chain = FirstOrderMarkovChain()
second_order_city_chain = SecondOrderMarkovChain()
third_order_city_chain = ThirdOrderMarkovChain()


# 读取文件并构建城市转移的马尔可夫链
def read_and_process_files_for_city(file_paths, first_order_chain, second_order_chain):
    line_nums = 0
    airports_lens = 0
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                line_nums += 1
                airports = re.findall(r'\b[A-Z]{3}\b', line)
                airports_len = len(airports)
                airports_lens += airports_len
                cities = [airport_to_city_name(airport) for airport in airports]
                for i in range(1, len(cities)):
                    first_order_chain.add_transition(cities[i - 1], cities[i])
                    if i > 1:
                        second_order_chain.add_transition(cities[i - 2], cities[i - 1], cities[i])
                        if i > 2:
                            third_order_city_chain.add_transition(cities[i - 3], cities[i - 2], cities[i - 1], cities[i])

        print(f"Processed file: {file_path}")
    print('totol line:', line_nums)
    print('average airports:', airports_lens / line_nums)


def read_and_test_files_city(file_path, first_order_city_chain, second_order_city_chain, num_lines):
    first_total_predictions = 0
    first_correct_predictions = 0
    second_total_predictions = 0
    second_correct_predictions = 0
    third_total_predictions = 0
    third_correct_predictions = 0


    # 预编译正则表达式
    pattern = re.compile(r'\b[A-Z]{3}\b')

    # 使用tqdm创建进度条
    with tqdm(total=num_lines, desc="Processing") as pbar:
        with open(file_path, 'r') as file:
            lines_read = 0
            while lines_read < num_lines:
                line = next(file)
                pbar.update(1)  # 更新进度条
                lines_read += 1
                airports = pattern.findall(line)
                cities = [airport_to_city.get(airport, None) for airport in airports]  # 将机场代码转换为城市名称
                for i in range(1, len(cities)):
                    if cities[i-2] is None or cities[i - 1] is None or cities[i] is None :  # 检查城市名称是否有效
                        continue
                    # 一阶马尔可夫链预测
                    first_total_predictions += 1
                    if first_order_city_chain.transition(cities[i - 1]) == cities[i]:
                        first_correct_predictions += 1
                    # 二阶马尔可夫链预测
                    if i > 1:
                        second_total_predictions += 1
                        if second_order_city_chain.transition((cities[i - 2], cities[i - 1])) == cities[i]:
                            second_correct_predictions += 1
                        if i > 2:
                            third_total_predictions += 1
                            if third_order_city_chain.transition((cities[i - 3], cities[i - 2], cities[i - 1])) == cities[i]:
                                third_correct_predictions += 1

    accuracy_first_order = first_correct_predictions / first_total_predictions if first_total_predictions > 0 else 0
    accuracy_second_order = second_correct_predictions / second_total_predictions if second_total_predictions > 0 else 0
    accuracy_third_order = third_correct_predictions / third_total_predictions if third_total_predictions > 0 else 0

    print(f"First Order Markov Chain Accuracy: {accuracy_first_order:.4f}")
    print(f"Second Order Markov Chain Accuracy: {accuracy_second_order:.4f}")
    print(f"Third Order Markov Chain Accuracy: {accuracy_third_order:.4f}")


# 文件路径列表
file_paths = [
    '/doc/original_data/airports/db1b.public.201103.asc',
    '/doc/original_data/airports/db1b.public.201106.asc',
    '/doc/original_data/airports/db1b.public.201109.asc',
    # '../data/db1b.public.201112.asc',
    # 添加更多文件路径
]

# 读取文件并构建城市转移的马尔可夫链
read_and_process_files_for_city(file_paths, first_order_city_chain, second_order_city_chain)

# 测试准确率
test_path = '../data/original_data/db1b.public.201112.asc'
read_and_test_files_city(test_path, first_order_city_chain, second_order_city_chain, 10000)

# ——————————————————————桑基图数据——————————————————————————#
# # 预测并记录结果
# start_paths = [
#     '纽约 芝加哥',
#     '亚特兰大 芝加哥',
#     '旧金山 芝加哥',
#     '西雅图 芝加哥',
# ]
#
# def predict_and_record(start_paths, first_order_chain, second_order_chain, num_predictions, output_file_first_order, output_file_second_order):
#     results_first_order = []
#     results_second_order = []
#
#     for start_path in start_paths:
#         start_airports = start_path.split()
#         start_cities = [citys for citys in start_airports]
#         if len(start_cities) != 2:
#             continue
#
#         start, end = start_cities
#
#         # 一阶马尔可夫链预测
#         for _ in range(num_predictions):
#             prediction = first_order_chain.transition(end)
#             results_first_order.append(f"{start} {end} {prediction}")
#
#         # 二阶马尔可夫链预测
#         for _ in range(num_predictions):
#             if len(start_airports) < 2:
#                 continue
#             prediction = second_order_chain.transition((start, end))
#             results_second_order.append(f"{start} {end} {prediction}")
#
#     with open(output_file_first_order, 'w') as file:
#         for result in results_first_order:
#             file.write(f"{result}\n")
#
#     with open(output_file_second_order, 'w') as file:
#         for result in results_second_order:
#             file.write(f"{result}\n")
#
#
# predict_and_record(start_paths, first_order_city_chain, second_order_city_chain, 1000, '../data/predictions_first_order_city.txt', '../data/predictions_second_order_city.txt')

# #——————————————————————过滤桑基图数据——————————————————————————#
# consider_citys = [
#     '纽约',
#     '亚特兰大',
#     '旧金山',
#     '西雅图',
# ]
#
# input_paths = [
#     '../data/predictions_first_order_city.txt',
#     '../data/predictions_second_order_city.txt'
# ]
#
# output_paths = [
#     '../data/predictions_first_order_city_filter.txt',
#     '../data/predictions_second_order_city_filter.txt'
# ]
#
# def filter_data(input_path, output_path, consider_citys):
#     with open(input_path, 'r', encoding='GBK') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             cities = line.strip().split()
#             if len(cities) == 3 and  cities[2] in consider_citys:
#                 outfile.write(line)
#
# for input_path, output_path in zip(input_paths, output_paths):
#     filter_data(input_path, output_path, consider_citys)
