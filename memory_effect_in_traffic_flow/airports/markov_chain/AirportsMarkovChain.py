import re
from pre_process.markov_chain import FirstOrderMarkovChain, SecondOrderMarkovChain, ThirdOrderMarkovChain
from tqdm import tqdm


def read_and_process_files(file_paths, first_order_chain, second_order_chain, third_order_chain):
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            for line in file:
                airports = re.findall(r'\b[A-Z]{3}\b', line)
                for i in range(1, len(airports)):
                    first_order_chain.add_transition(airports[i - 1], airports[i])
                    if i > 1:
                        second_order_chain.add_transition(airports[i - 2], airports[i - 1], airports[i])
                        if i > 2:
                            third_order_chain.add_transition(airports[i - 3], airports[i - 2], airports[i - 1],
                                                             airports[i])
        print(f"Processed file: {file_path}")

# 计算 Macro Precision、Macro Recall 和 Macro F1
def calculate_metrics(tp, fp, fn):
    precision_list = []
    recall_list = []
    f1_list = []
    all_classes = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    for class_label in all_classes:
        tp_count = tp.get(class_label, 0)
        fp_count = fp.get(class_label, 0)
        fn_count = fn.get(class_label, 0)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    return macro_precision, macro_recall, macro_f1

def read_and_test_files(file_path, first_order_chain, second_order_chain, third_order_chain, num_lines):
    first_total_predictions = 0
    first_correct_predictions = 0
    second_total_predictions = 0
    second_correct_predictions = 0
    third_total_predictions = 0
    third_correct_predictions = 0

    # 用于存储每个类别的 TP、FP 和 FN
    first_order_tp = {}
    first_order_fp = {}
    first_order_fn = {}
    second_order_tp = {}
    second_order_fp = {}
    second_order_fn = {}
    third_order_tp = {}
    third_order_fp = {}
    third_order_fn = {}

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
                for i in range(2, len(airports)):
                    # 一阶马尔可夫链预测
                    first_total_predictions += 1
                    prediction = first_order_chain.transition(airports[i - 1])
                    if prediction == airports[i]:
                        first_correct_predictions += 1
                        # 更新 TP
                        first_order_tp[airports[i]] = first_order_tp.get(airports[i], 0) + 1
                    else:
                        # 更新 FP 和 FN
                        first_order_fp[prediction] = first_order_fp.get(prediction, 0) + 1
                        first_order_fn[airports[i]] = first_order_fn.get(airports[i], 0) + 1

                    # 二阶马尔可夫链预测
                    if i > 1:
                        second_total_predictions += 1
                        prediction = second_order_chain.transition((airports[i - 2], airports[i - 1]))
                        if prediction == airports[i]:
                            second_correct_predictions += 1
                            # 更新 TP
                            second_order_tp[airports[i]] = second_order_tp.get(airports[i], 0) + 1
                        else:
                            # 更新 FP 和 FN
                            second_order_fp[prediction] = second_order_fp.get(prediction, 0) + 1
                            second_order_fn[airports[i]] = second_order_fn.get(airports[i], 0) + 1

                    # 三阶马尔可夫链预测
                    if i > 2:
                        third_total_predictions += 1
                        prediction = third_order_chain.transition((airports[i - 3], airports[i - 2], airports[i - 1]))
                        if prediction == airports[i]:
                            third_correct_predictions += 1
                            # 更新 TP
                            third_order_tp[airports[i]] = third_order_tp.get(airports[i], 0) + 1
                        else:
                            # 更新 FP 和 FN
                            third_order_fp[prediction] = third_order_fp.get(prediction, 0) + 1
                            third_order_fn[airports[i]] = third_order_fn.get(airports[i], 0) + 1

    # 计算准确率
    accuracy_first_order = first_correct_predictions / first_total_predictions if first_total_predictions > 0 else 0
    accuracy_second_order = second_correct_predictions / second_total_predictions if second_total_predictions > 0 else 0
    accuracy_third_order = third_correct_predictions / third_total_predictions if third_total_predictions > 0 else 0

    first_order_macro_precision, first_order_macro_recall, first_order_macro_f1 = calculate_metrics(
        first_order_tp, first_order_fp, first_order_fn
    )
    second_order_macro_precision, second_order_macro_recall, second_order_macro_f1 = calculate_metrics(
        second_order_tp, second_order_fp, second_order_fn
    )
    third_order_macro_precision, third_order_macro_recall, third_order_macro_f1 = calculate_metrics(
        third_order_tp, third_order_fp, third_order_fn
    )

    # 输出结果
    print(f"First Order Markov Chain Accuracy: {accuracy_first_order:.4f}")
    print(f"First Order Markov Chain Macro Precision: {first_order_macro_precision:.4f}")
    print(f"First Order Markov Chain Macro Recall: {first_order_macro_recall:.4f}")
    print(f"First Order Markov Chain Macro F1: {first_order_macro_f1:.4f}")

    print(f"Second Order Markov Chain Accuracy: {accuracy_second_order:.4f}")
    print(f"Second Order Markov Chain Macro Precision: {second_order_macro_precision:.4f}")
    print(f"Second Order Markov Chain Macro Recall: {second_order_macro_recall:.4f}")
    print(f"Second Order Markov Chain Macro F1: {second_order_macro_f1:.4f}")

    print(f"Third Order Markov Chain Accuracy: {accuracy_third_order:.4f}")
    print(f"Third Order Markov Chain Macro Precision: {third_order_macro_precision:.4f}")
    print(f"Third Order Markov Chain Macro Recall: {third_order_macro_recall:.4f}")
    print(f"Third Order Markov Chain Macro F1: {third_order_macro_f1:.4f}")


# def read_and_test_files(file_path, first_order_chain, second_order_chain, third_order_chain, num_lines):
#     first_total_predictions = 0
#     first_correct_predictions = 0
#     second_total_predictions = 0
#     second_correct_predictions = 0
#     third_total_predictions = 0
#     third_correct_predictions = 0
#
#     # 预编译正则表达式
#     pattern = re.compile(r'\b[A-Z]{3}\b')
#
#     # 使用tqdm创建进度条
#     with tqdm(total=num_lines, desc="Processing") as pbar:
#         with open(file_path, 'r') as file:
#             lines_read = 0
#             while lines_read < num_lines:
#                 line = next(file)
#                 pbar.update(1)  # 更新进度条
#                 lines_read += 1
#                 airports = pattern.findall(line)
#                 for i in range(2, len(airports)):
#                     # 一阶马尔可夫链预测
#                     first_total_predictions += 1
#                     if first_order_chain.transition(airports[i - 1]) == airports[i]:
#                         first_correct_predictions += 1
#                     # 二阶马尔可夫链预测
#                     if i > 1:
#                         second_total_predictions += 1
#                         if second_order_chain.transition((airports[i - 2], airports[i - 1])) == airports[i]:
#                             second_correct_predictions += 1
#                     # 三阶马尔可夫链预测
#                     if i > 2:
#                         third_total_predictions += 1
#                         if third_order_chain.transition((airports[i - 3], airports[i - 2], airports[i - 1])) == \
#                                 airports[i]:
#                             third_correct_predictions += 1
#
#     accuracy_first_order = first_correct_predictions / first_total_predictions if first_total_predictions > 0 else 0
#     accuracy_second_order = second_correct_predictions / second_total_predictions if second_total_predictions > 0 else 0
#     accuracy_third_order = third_correct_predictions / third_total_predictions if third_total_predictions > 0 else 0
#
#     print(f"First Order Markov Chain Accuracy: {accuracy_first_order:.4f}")
#     print(f"Second Order Markov Chain Accuracy: {accuracy_second_order:.4f}")
#     print(f"Third Order Markov Chain Accuracy: {accuracy_third_order:.4f}")


def read_and_test_all_files(file_path, first_order_chain, second_order_chain, third_order_chain, num_lines):
    total_predictions = 0
    correct_predictions = 0
    first_order_predictions = 0
    first_order_correct_predictions = 0
    second_order_predictions = 0
    second_order_correct_predictions = 0
    third_order_predictions = 0
    third_order_correct_predictions = 0

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
                length = len(airports)
                if length == 2:
                    total_predictions += 1
                    first_order_predictions += 1
                    if first_order_chain.transition(airports[0]) == airports[1]:
                        first_order_correct_predictions += 1
                    if first_order_chain.transition(airports[1]) == airports[1]:
                        correct_predictions += 1
                elif length == 3:
                    total_predictions += 1
                    second_order_predictions += 1
                    if second_order_chain.transition((airports[0], airports[1])) == airports[2]:
                        second_order_correct_predictions += 1
                    if second_order_chain.transition((airports[0], airports[1])) == airports[2]:
                        correct_predictions += 1
                elif length == 4:
                    total_predictions += 1
                    third_order_predictions += 1
                    if third_order_chain.transition((airports[0], airports[1], airports[2])) == airports[3]:
                        third_order_correct_predictions += 1
                    if third_order_chain.transition((airports[0], airports[1], airports[2])) == airports[3]:
                        correct_predictions += 1
                else:
                    pass

    print(f"All Order Markov Chain Accuracy: {correct_predictions / total_predictions:.4f}")
    print(f"First Order Markov Chain Accuracy: {first_order_correct_predictions / first_order_predictions:.4f}")
    print(f"Second Order Markov Chain Accuracy: {second_order_correct_predictions / second_order_predictions:.4f}")
    print(f"Third Order Markov Chain Accuracy: {third_order_correct_predictions / third_order_predictions:.4f}")


# 创建一阶和二阶马尔可夫链实例
first_order_chain = FirstOrderMarkovChain()
second_order_chain = SecondOrderMarkovChain()
third_order_chain = ThirdOrderMarkovChain()

# 假设文件路径列表
file_paths = [
    '../data/original_data/db1b.public.201103.asc',
    '../data/original_data/db1b.public.201106.asc',
    '../data/original_data/db1b.public.201109.asc',
    # 添加更多文件路径
]
test_path = '../data/original_data/db1b.public.201203.asc'
# 读取多个文件并构建马尔可夫链
read_and_process_files(file_paths, first_order_chain, second_order_chain, third_order_chain)
read_and_test_files(test_path, first_order_chain, second_order_chain, third_order_chain, 10000)
# ——————————————————————预测测试——————————————————————————#
# # 假设 predict1 和 predict2 是从马尔可夫链得到的预测结果
# predict1 = first_order_chain.predict(10000)
# predict2 = second_order_chain.predict(10000)
#
# # 将预测结果转换为列表形式，如果它们不是列表
# predict1 = list(predict1)
# predict2 = list(predict2)
# with open('../data/1predictions10000.txt', 'w') as file:
#     for item in predict1:
#         file.write(f"{item}\n")
# with open('../data/2predictions10000.txt', 'w') as file:
#     for item in predict2:
#         file.write(f"{item}\n")

# ——————————————————————准确率测试——————————————————————————#
# print("开始测试准确率")
# 测试准确率
# read_and_test_files(test_path, first_order_chain, second_order_chain, third_order_chain, 10000)

# ——————————————————————桑基图数据——————————————————————————#
# def predict_and_record(start_paths, first_order_chain, second_order_chain, num_predictions, output_file_first_order,
#                        output_file_second_order):
#     results_first_order = []
#     results_second_order = []
#
#     for start_path in start_paths:
#         start_airports = start_path.split()
#         if len(start_airports) != 2:
#             continue
#
#         start, end = start_airports
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
# # # 假设起始路径列表
# # start_paths = [
# #     'ELP ATL',
# #     'AEX ATL',
# #     'DAY ATL',
# #     'ICT ATL',
# #     'JAN ATL',
# # ]
#
# start_paths = [
#     'ELP EWR',
#     'AEX EWR',
#     'DAY EWR',
#     'ICT EWR',
#     'JAN EWR',
# ]
#
# # 预测并记录结果
# predict_and_record(start_paths, first_order_chain, second_order_chain, 1000,
#                    '../data/predictions_first_order.txt', '../data/predictions_second_order.txt')
