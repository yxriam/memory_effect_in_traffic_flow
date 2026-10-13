import os.path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from LSTM import LSTMModel
from dataloader.Dataset import AirportDataset, OrderSequenceDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, test_dataloader,airport_to_index, max_samples=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    airport_index_to_code = {idx: airport for airport, idx in airport_to_index.items()}
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing Batches"):
            batch = batch.to(device)
            inputs = batch[:, :-1].to(device)
            targets = batch[:, -1].to(device)

            outputs = model(inputs)
            preds = torch.argmax(torch.sigmoid(outputs), dim=1).cpu().numpy()
            labels = targets.cpu().numpy()
            for i in range(len(inputs)):
                # 转换 inputs 为机场三字码
                input_codes = [airport_index_to_code[idx.item()] for idx in inputs[i]]
            # 转换 targets 为机场三字码
            target_code = airport_index_to_code[labels[i]]
            # 转换 preds 为机场三字码
            pred_code = airport_index_to_code[preds[i]]

            print(f"  original Codes: {input_codes} {target_code}")
            print(f"  Predicted Code: {pred_code}")


def test_model(model, test_dataloader, max_samples=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    all_preds = []
    all_labels = []
    sample_count = 0
    model = model.to(device)
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing Batches"):
            inputs,targets = batch #解包
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = torch.argmax(torch.sigmoid(outputs), dim=1).cpu().numpy()
            labels = targets.cpu().numpy()

            for pred, label in zip(preds, labels):
                all_preds.append(pred)
                all_labels.append(label)
                sample_count += 1
                if sample_count >= max_samples:
                    break
            if sample_count >= max_samples:
                break
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # print("labels:",all_labels.shape,"preds:",all_preds.shape)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, precision, recall, f1


def prepare_data():
    # Step 1: 构造滑动窗口序列（每个 order 对应一个序列列表）
    print("构造滑动窗口样本用于测试...")
    order_to_sequences = defaultdict(list)

    for x in range(3, 12):
        order = x - 1
        train_data_path = f'./sequence_split/202409/202409_{x}.txt'
        # Step 1: 构造训练样本，每条样本是一个长度为 order+1 的子序列
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"构造滑动窗口子序列样本 {order}", unit="line"):
                input_sequence = line.strip()
                airports = input_sequence.split()
                if len(airports) <= order:
                    continue  # 无法预测

                for i in range(2, len(airports)):
                    window = airports[0:i]  # 长度为 order + 1 的窗口
                    order_to_sequences[len(window)].append(window)
        print(f"order {order} 构造完毕")

    print("样本生成完毕，开始训练 alpha...")
    return order_to_sequences


def example():
    log_file = open('test_results_original.log', 'w')
    for x in range(3, 11):
        seq_length = x-1
        # file_paths = [
        #     f"../data/sequences_split/202412/202412_{x}.txt"]
        # print(f'____________start load test data for seq_length {seq_length}_____________')
        # 将机场代码映射到索引
        with open('../data/mid_product/airport_dict.json', 'r',
                    encoding='utf-8') as f:
            airport_to_city = json.load(f)
        airport_to_index = {airport: idx for idx, airport in enumerate(airport_to_city)}


        # 定义正则表达式模式来匹配三个英文字母的机场名称
        pattern = re.compile(r'\b[A-Z]{3}\b')

        # 创建数据集和数据加载器
        dataset = AirportDataset(file_paths, airport_to_index, pattern, x)

        # 创建数据加载器
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # 模型参数
        embedding_dim = 16
        hidden_dim = 64
        num_layers = 2
        num_classes = len(airport_to_index)  # 465个不同的机场代码

        # 创建模型
        model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)

        # 加载最佳模型
        model_path = f'./output/lstmuffman/lstm{seq_length}/lstm{seq_length}_best.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 测试模型
        # predict(model, test_dataloader, airport_to_index)
        accuracy, precision, recall, f1 = test_model(model, test_dataloader)
        log_line = f'Test Results for seq_length {seq_length}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}\n'
        log_file.write(log_line)
        print(log_line.strip())

    log_file.close()

def example_last():
    log_file = open('test_results_all_202412.log', 'w')
    order_to_sequences = prepare_data()
    for x in range(3, 11):
        seq_length = x-1
        with open('airport_dict.json', 'r',
                    encoding='utf-8') as f:
            airport_to_city = json.load(f)
        airport_to_index = {airport: idx for idx, airport in enumerate(airport_to_city)}

        # 创建数据集和数据加载器
        dataset = OrderSequenceDataset(order_to_sequences[x], airport_to_index)

        # 创建数据加载器
        test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # 模型参数
        embedding_dim = 16
        hidden_dim = 128
        num_layers = 2
        num_classes = len(airport_to_index)  # 465个不同的机场代码

        # 创建模型
        model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)

        # 加载最佳模型
        model_path = f'output/lstm_emb{embedding_dim}_hid{hidden_dim}_nl{num_layers}/lstm{seq_length}/lstm{seq_length}_best.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 测试模型
        # predict(model, test_dataloader, airport_to_index)
        accuracy, precision, recall, f1 = test_model(model, test_dataloader)
        log_line = f'Test Results for seq_length {seq_length}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}\n'
        log_file.write(log_line)
        print(log_line.strip())

    log_file.close()





if __name__ == "__main__":
    example_last()
