import os
import pickle
import re
import random

import numpy as np
import torch
import json

from tqdm import tqdm

from LSTM import LSTMModel
import torch.nn.functional as F
from n_ordered_markov_chain import create_markov_chain
import pickle

from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('airport_dict.json', 'r',
          encoding='utf-8') as f:
    airport_to_city = json.load(f)



# 将机场代码映射到索引
airport_to_index = {airport: idx for idx, airport in enumerate(airport_to_city)}

# 模型参数
embedding_dim = 16
hidden_dim = 128
num_layers = 2
num_classes = len(airport_to_index)  # 465个不同的机场代码


def get_lstm_probabilities(model, input_sequence,temperature=0.5):
    model.eval()
    with torch.no_grad():
        output = model(input_sequence)
        probabilities = F.softmax(output/temperature, dim=-1)
    return probabilities


def combine_predictions(lstm_model, markov_chain, input_sequence, alpha):
    # 将输入序列转换为索引序列
    if isinstance(input_sequence, str):
        input_airports = input_sequence.split(" ")
    else:
        input_airports = input_sequence
    # print(input_airports, "input_airports")
    input_indices = [airport_to_index[airport] for airport in input_airports]

    # 将索引序列转换为张量
    input_tensor = torch.tensor([input_indices], device=device)

    # 获取 LSTM 模型的预测概率
    lstm_probabilities = get_lstm_probabilities(lstm_model, input_tensor)
    print("lstm_probabilities:", sum(lstm_probabilities))
    # 获取马尔可夫链的预测概率
    current_states = tuple(input_airports[:markov_chain.order])
    # print("current_state:", current_states)
    markov_probabilities = markov_chain.get_probabilities().get(current_states, {})
    # print("markov_probabilities:", markov_probabilities)

    # 将马尔可夫链的预测概率转换为张量
    markov_prob_tensor = torch.zeros(lstm_probabilities.shape, device=device)
    for state, prob in markov_probabilities.items():
        if state in airport_to_index:
            markov_prob_tensor[0, airport_to_index[state]] = prob

    # 结合 LSTM 和马尔可夫链的预测结果
    # print("lstm_probabilities:",lstm_probabilities)
    # print("markov_prob_tensor:",markov_prob_tensor)
    combined_probabilities = alpha * lstm_probabilities + (1 - alpha) * markov_prob_tensor
    # print("combined_probabilities:",combined_probabilities)
    return combined_probabilities



def sample_from_probabilities(probabilities):
    # 将概率分布转换为 numpy 数组
    prob_array = probabilities.cpu().numpy().flatten()
    # 确保概率值非负
    prob_array = np.maximum(prob_array, 1e-10)
    # 归一化概率分布
    prob_array /= prob_array.sum()
    # 根据概率分布随机选择状态
    # print("prob_array:",prob_array)
    sampled_state = np.random.choice(len(prob_array), p=prob_array)
    return sampled_state


def index_to_airport(index):
    for airport, idx in airport_to_index.items():
        if idx == index:
            return airport
    return None

def evaluate_model(model, markov_chain, train_data, alpha, test_times):
    correct_predictions = 0
    total_predictions = 0

    for i in tqdm(range(test_times), desc="Evaluating", unit="test"):
        # 使用取余数的方式循环获取输入序列
        input_sequence = train_data[i % len(train_data)][:-1]
        # print("input_sequence:",input_sequence)
        airports= input_sequence
        missing_airports = [airport for airport in airports if airport not in airport_to_index]
        if missing_airports:
            print(f"Skipping sequence at index {i} due to missing airports: {missing_airports}")
            continue  # 如果有机场名称不在映射中，跳过当前迭代
        combined_probabilities = combine_predictions(model, markov_chain, input_sequence, alpha)
        sampled_state = sample_from_probabilities(combined_probabilities)
        sampled_airport = index_to_airport(sampled_state)
        # 检查预测的机场是否与输入序列的最后一个机场相同
        if sampled_airport == input_sequence[-1]:
            correct_predictions += 1
        total_predictions += 1

    # 计算准确率
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def train_alpha(lstm_model, markov_chain, train_data_or_path, test_times=10000):
    if isinstance(train_data_or_path, str):
        with open(train_data_or_path, 'r') as f:
            train_data = f.readlines()
    else:
        train_data = train_data_or_path  # already a list

    best_accuracy = 0
    best_alpha = 0

    for alpha in np.arange(0, 1.1, 0.1):
        accuracy = evaluate_model(lstm_model, markov_chain, train_data, alpha, test_times)
        print(f"Alpha: {alpha:.1f}, Accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    print(f"Best alpha: {best_alpha:.1f}, Best accuracy: {best_accuracy:.4f}")
    return best_alpha

def train():
    order_to_sequences = defaultdict(list)
    sample_size=10000
    # 用来存储每个order对应的best_alpha
    best_alpha_file = 'best_alpha.json'

    # 如果文件存在，加载现有的best_alpha字典
    if os.path.exists(best_alpha_file):
        with open(best_alpha_file, 'r') as f:
            best_alphas = json.load(f)
    else:
        best_alphas = {}

     # Step 1: 构造滑动窗口序列（每个 order 对应一个序列列表）
    for x in range(3, 12):
        order = x - 1
        train_data_path = f'./sequence_split/202409/202409_{x}.txt'
        # Step 1: 构造训练样本，每条样本是一个长度为 order+1 的子序列
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"构造滑动窗口子序列样本 {order}", unit="line"):
                input_sequence = line.strip()
                # print("input_sequence", input_sequence)
                airports = input_sequence.split()
                # print('airports',airports,len(airports))
                if len(airports) <= order:
                    continue  # 无法预测

                for i in range(2,len(airports)):
                    window = airports[0:i]  # 长度为 order + 1 的窗口
                    order_to_sequences[len(window)].append(window)
        print(f"order {order} 构造完毕")

    for order in order_to_sequences:
        if len(order_to_sequences[order]) > sample_size:
            order_to_sequences[order] = random.sample(order_to_sequences[order], sample_size)

    print("样本生成完毕，开始训练 alpha...")
    for x in range(3, 12):
        order = x - 1
        lstm_model_path = f'./output/lstm_emb16_hid128_nl2/lstm{x}/lstm{x}_best.pth'
        markov_chain_path = f'./output/markov/markov{order}.pkl'
        # 加载马尔可夫链
        with open(markov_chain_path, 'rb') as f:
            data = pickle.load(f)
            MarkovChainClass = create_markov_chain(data['order'])
            markov_chain = MarkovChainClass()
            markov_chain.transitions = data['transitions']
            markov_chain.state_counts = data['state_counts']

        # 加载LSTM模型
        model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)
        model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        print("order:", order)
        best_alpha = train_alpha(model, markov_chain, order_to_sequences[order], test_times=1000)
        # 保存当前的best_alpha
        best_alphas[order] = best_alpha
        # 更新文件
        with open(best_alpha_file, 'w') as f:
            json.dump(best_alphas, f)

    # Step 2: 对每个 order 训练 best_alpha

    print("全部 alpha 训练完成 ✅")

def evaluate_model_with_counts(model, markov_chain, train_data_path, alpha):
    if isinstance(train_data_path, str):
        with open(train_data_path, 'r') as f:
            train_data = f.readlines()
    else:
        train_data = train_data_path  # already a list

    # 初始化每个类别的 TP、FP 和 FN
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)
    correct_predictions = 0
    prediction_num = min(len(train_data), 1000)  # 总预测数等于数据条数

    for i in tqdm(range(prediction_num), desc=f"Evaluating order {markov_chain.order}", unit="sequence"):
        if isinstance(train_data[i], str):
            input_sequence = train_data[i].strip()
            airports = input_sequence.split()
        elif isinstance(train_data[i], list) :
            # 如果 train_data 是列表，直接使用索引获取字符串
            input_sequence= train_data[i]
            airports = train_data[i]
        missing_airports = [airport for airport in airports if airport not in airport_to_index]
        if missing_airports:
            print(f"Skipping sequence at index {i} due to missing airports: {missing_airports}")
            continue

        combined_probabilities = combine_predictions(model, markov_chain, input_sequence, alpha)
        sampled_state = sample_from_probabilities(combined_probabilities)
        sampled_airport = index_to_airport(sampled_state)
        true_airport = airports[-1]

        if sampled_airport == true_airport:
            correct_predictions += 1
            tp_per_class[true_airport] += 1
        else:
            fp_per_class[sampled_airport] += 1
            fn_per_class[true_airport] += 1

    # 计算 Accuracy
    accuracy = correct_predictions / prediction_num if prediction_num > 0 else 0

    # 计算每个类别的 Precision、Recall 和 F1-score
    precisions = []
    recalls = []
    f1_scores = []
    for airport in set(tp_per_class.keys()) | set(fp_per_class.keys()) | set(fn_per_class.keys()):
        tp = tp_per_class[airport]
        fp = fp_per_class[airport]
        fn = fn_per_class[airport]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # 计算 Macro Precision、Macro Recall 和 Macro F1-score
    macro_precision = sum(precisions) / len(precisions) if precisions else 0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return accuracy, macro_precision, macro_recall, macro_f1, correct_predictions, prediction_num, tp_per_class, fp_per_class, fn_per_class
#
# def test1():
#     # 用来存储每个order对应的best_alpha
#     best_alpha_file = 'best_alpha_emb16_hid128_nl2.json'
#
#     # 如果文件存在，加载现有的best_alpha字典
#     if os.path.exists(best_alpha_file):
#         with open(best_alpha_file, 'r') as f:
#             best_alphas = json.load(f)
#     else:
#         print("Please train the model first.")
#         return
#
#     # 初始化用于存储各指标总和的变量
#     total_tp = defaultdict(int)
#     total_fp = defaultdict(int)
#     total_fn = defaultdict(int)
#     total_correct_predictions = 0
#     total_predictions = 0
#
#
#     for x in range(3, 12):  # 遍历所有order
#         order = x - 1
#         lstm_model_path = f'./output/lstm_emb16_hid128_nl2/lstm{x}/lstm{x}_best.pth'
#         markov_chain_path = f'./output/markov/markov{order}.pkl'
#         train_data_path = f'./sequence_split/202409/202409_{x}.txt'
#
#         # 加载马尔可夫链
#         with open(markov_chain_path, 'rb') as f:
#             data = pickle.load(f)
#             MarkovChainClass = create_markov_chain(data['order'])
#             markov_chain = MarkovChainClass()
#             markov_chain.transitions = data['transitions']
#             markov_chain.state_counts = data['state_counts']
#
#         # 加载LSTM模型
#         model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)
#         model.load_state_dict(torch.load(lstm_model_path, map_location=device))
#         print(f"Loading model for order {order}...")
#
#         # 获取当前order的最佳alpha
#         best_alpha = best_alphas[str(order)]  # 确保键是字符串类型
#
#         # 评估模型并获取准确率
#         accuracy, macro_precision, macro_recall, macro_f1, correct_predictions, prediction_num, tp_per_class, fp_per_class, fn_per_class = evaluate_model_with_counts(
#             model, markov_chain, train_data_path, best_alpha
#         )
#
#         print(
#             f"Order {order}, Accuracy: {accuracy:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")
#
#         # 累加每个类别的 TP、FP 和 FN
#         for airport in tp_per_class:
#             total_tp[airport] += tp_per_class[airport]
#         for airport in fp_per_class:
#             total_fp[airport] += fp_per_class[airport]
#         for airport in fn_per_class:
#             total_fn[airport] += fn_per_class[airport]
#
#         total_correct_predictions += correct_predictions
#         total_predictions += prediction_num
#
#     # 计算所有order的总准确率
#     overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
#     print(f"Overall accuracy across all orders (based on total predictions): {overall_accuracy:.4f}")
#
#     # 计算所有order的总宏准确率
#     overall_macro_precision = sum(total_tp[airport] / (total_tp[airport] + total_fp[airport]) if (total_tp[airport] +
#                                                                                                   total_fp[
#                                                                                                       airport]) > 0 else 0
#                                   for airport in total_tp) / len(total_tp)
#     print(f"Overall macro precision across all orders (based on total predictions): {overall_macro_precision:.4f}")
#
#     # 计算所有order的总召回率
#     overall_macro_recall = sum(total_tp[airport] / (total_tp[airport] + total_fn[airport]) if (total_tp[airport] +
#                                                                                                total_fn[
#                                                                                                    airport]) > 0 else 0
#                                for airport in total_tp) / len(total_tp)
#     print(f"Overall macro recall across all orders (based on total predictions): {overall_macro_recall:.4f}")
#
#     # 计算所有order的总F1值
#     f1_scores = []
#     for airport in total_tp:
#         denominator1 = total_tp[airport] + total_fp[airport]
#         denominator2 = total_tp[airport] + total_fn[airport]
#         if denominator1 > 0 and denominator2 > 0:
#             precision = total_tp[airport] / denominator1
#             recall = total_tp[airport] / denominator2
#             combined_denominator = precision + recall
#             if combined_denominator > 0:
#                 f1 = 2 * (precision * recall) / combined_denominator
#             else:
#                 f1 = 0
#         else:
#             f1 = 0
#         f1_scores.append(f1)
#
#     overall_macro_f1 = sum(f1_scores) / len(f1_scores)
#     print(f"Overall macro F1 across all orders (based on total predictions): {overall_macro_f1:.4f}")

def test():
    # 设置参数
    best_alpha_file = 'best_alpha.json'
    sample_size = 10000

    # 加载已保存的 best_alpha 值
    if os.path.exists(best_alpha_file):
        with open(best_alpha_file, 'r') as f:
            best_alphas = json.load(f)
    else:
        print("Please train the model first.")
        return

    # Step 1: 构造滑动窗口序列（每个 order 对应一个序列列表）
    print("构造滑动窗口样本用于测试...")
    order_to_sequences = defaultdict(list)

    for x in range(3, 12):
        order = x - 1
        train_data_path = f'./sequence_split/202412/202412_{x}.txt'
        # Step 1: 构造训练样本，每条样本是一个长度为 order+1 的子序列
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"构造滑动窗口子序列样本 {order}", unit="line"):
                input_sequence = line.strip()
                airports = input_sequence.split()
                if len(airports) <= order:
                    continue  # 无法预测

                for i in range(2,len(airports)):
                    window = airports[0:i]  # 长度为 order + 1 的窗口
                    order_to_sequences[len(window)].append(window)
        print(f"order {order} 构造完毕")

    for order in order_to_sequences:
        if len(order_to_sequences[order]) > sample_size:
            order_to_sequences[order] = random.sample(order_to_sequences[order], sample_size)

    print("样本生成完毕，开始训练 alpha...")

    # 限制样本数量
    for order in order_to_sequences:
        if len(order_to_sequences[order]) > sample_size:
            order_to_sequences[order] = random.sample(order_to_sequences[order], sample_size)

    # Step 2: 开始逐 order 模型测试
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    total_correct_predictions = 0
    total_predictions = 0

    for order, sequences in order_to_sequences.items():
        print(f"\nEvaluating order {order}...")

        lstm_model_path = f'./output/lstm_emb16_hid128_nl2/lstm{order+1}/lstm{order+1}_best.pth'
        markov_chain_path = f'./output/markov/markov{order}.pkl'

        if not os.path.exists(lstm_model_path) or not os.path.exists(markov_chain_path):
            print(f"Missing model files for order {order}, skipping...")
            continue

        # 加载模型
        model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)
        model.load_state_dict(torch.load(lstm_model_path, map_location=device))

        with open(markov_chain_path, 'rb') as f:
            data = pickle.load(f)
            MarkovChainClass = create_markov_chain(data['order'])
            markov_chain = MarkovChainClass()
            markov_chain.transitions = data['transitions']
            markov_chain.state_counts = data['state_counts']

        best_alpha = best_alphas[str(order)]

        # 执行评估
        accuracy, macro_precision, macro_recall, macro_f1, correct_predictions, prediction_num, tp_per_class, fp_per_class, fn_per_class = evaluate_model_with_counts(
            model, markov_chain, sequences, best_alpha
        )

        print(f"Order {order}, Accuracy: {accuracy:.4f}, Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")

        # 累积总量
        for airport in tp_per_class:
            total_tp[airport] += tp_per_class[airport]
        for airport in fp_per_class:
            total_fp[airport] += fp_per_class[airport]
        for airport in fn_per_class:
            total_fn[airport] += fn_per_class[airport]

        total_correct_predictions += correct_predictions
        total_predictions += prediction_num

    # 汇总所有 order 的指标
    overall_accuracy = total_correct_predictions / total_predictions if total_predictions > 0 else 0
    overall_macro_precision = sum(total_tp[ap] / (total_tp[ap] + total_fp[ap]) if (total_tp[ap] + total_fp[ap]) > 0 else 0 for ap in total_tp) / len(total_tp)
    overall_macro_recall = sum(total_tp[ap] / (total_tp[ap] + total_fn[ap]) if (total_tp[ap] + total_fn[ap]) > 0 else 0 for ap in total_tp) / len(total_tp)

    f1_scores = []
    for ap in total_tp:
        p = total_tp[ap] / (total_tp[ap] + total_fp[ap]) if (total_tp[ap] + total_fp[ap]) > 0 else 0
        r = total_tp[ap] / (total_tp[ap] + total_fn[ap]) if (total_tp[ap] + total_fn[ap]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
    overall_macro_f1 = sum(f1_scores) / len(f1_scores)

    print("\n=== Overall Evaluation ===")
    print(f"Overall Accuracy:       {overall_accuracy:.4f}")
    print(f"Overall Macro Precision:{overall_macro_precision:.4f}")
    print(f"Overall Macro Recall:   {overall_macro_recall:.4f}")
    print(f"Overall Macro F1:       {overall_macro_f1:.4f}")


if __name__ == "__main__":
    train()
    test()
