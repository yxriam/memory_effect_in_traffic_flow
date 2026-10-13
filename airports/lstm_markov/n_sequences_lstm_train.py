import os.path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from LSTM import LSTMModel
from dataloader.Dataset import AirportDataset,OrderSequenceDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_lstm_model(seq_length, file_paths, model_output_dir):
    # 加载机场到城市的映射
    with open(f'./airport_dict.json', 'r',
              encoding='utf-8') as f:
        airport_to_city = json.load(f)

    # 将机场代码映射到索引
    airport_to_index = {airport: idx for idx, airport in enumerate(airport_to_city)}

    # 定义正则表达式模式来匹配三个英文字母的机场名称
    pattern = re.compile(r'\b[A-Z]{3}\b')

    # 创建数据集和数据加载器
    dataset = AirportDataset(file_paths, airport_to_index, pattern, seq_length+1)
    # dataset = OrderSequenceDataset(order_to_sequences[idx], airport_to_index)
    # 将数据集分为训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    # 模型参数
    embedding_dim = 16
    hidden_dim = 128
    num_layers = 2
    num_classes = len(airport_to_index)  # 465个不同的机场代码

    # 创建模型
    model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 30
    best_val_loss = float('inf')
    writer = SummaryWriter(os.path.join(model_output_dir, 'logs'))

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc="Training Batches"):
            batch = batch.to(device)
            # print(batch.shape, "batch shape")
            inputs = batch[:, :-1].to(device)
            # print(inputs.shape, "inputs shape")
            targets = batch[:, -1].to(device)
            # print(targets.shape, "targets shape")
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[:, :-1].to(device)
                targets = batch[:, -1].to(device)
                outputs = model(inputs).to(device)
                loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_output_dir, f'lstm{seq_length}_best.pth'))
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        torch.save(model.state_dict(), os.path.join(model_output_dir, f'lstm{seq_length}_epoch_{epoch + 1}.pth'))

        if no_improvement_epochs >= 5:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for 5 epochs.")
            break

    writer.close()
    torch.save(model.state_dict(), os.path.join(model_output_dir, f'lstm{seq_length}_final.pth'))
    print(f"Model saved to {model_output_dir}")

def train_lstm_model2(seq_length, sequence, model_output_dir):
    # 加载机场到城市的映射
    with open(f'./airport_dict.json', 'r',
              encoding='utf-8') as f:
        airport_to_city = json.load(f)

    # 将机场代码映射到索引
    airport_to_index = {airport: idx for idx, airport in enumerate(airport_to_city)}

    # 定义正则表达式模式来匹配三个英文字母的机场名称
    pattern = re.compile(r'\b[A-Z]{3}\b')

    # 创建数据集和数据加载器
    # dataset = AirportDataset(file_paths, airport_to_index, pattern, seq_length+1)
    dataset = OrderSequenceDataset(sequence, airport_to_index)
    # 将数据集分为训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    # 模型参数
    embedding_dim = 16
    hidden_dim = 128
    num_layers = 2
    num_classes = len(airport_to_index)  # 465个不同的机场代码

    # 创建模型
    model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 30
    best_val_loss = float('inf')
    writer = SummaryWriter(os.path.join(model_output_dir, 'logs'))

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc="Training Batches"):
            inputs, targets = batch  # 解包
            # print(batch.shape, "batch shape")
            inputs = inputs.to(device)
            # print(inputs.shape, "inputs shape")
            targets = targets.to(device)
            # print(targets.shape, "targets shape")
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs,targets= batch  # 解包
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs).to(device)
                loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_output_dir, f'lstm{seq_length}_best.pth'))
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        torch.save(model.state_dict(), os.path.join(model_output_dir, f'lstm{seq_length}_epoch_{epoch + 1}.pth'))

        if no_improvement_epochs >= 5:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for 5 epochs.")
            break

    writer.close()
    torch.save(model.state_dict(), os.path.join(model_output_dir, f'lstm{seq_length}_final.pth'))
    print(f"Model saved to {model_output_dir}")

def  prepare_data():
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

def example_last():
    # 使用示例
    for x in range(3,14):
        seq_length = x-1
        file_paths = [
            f'./sequence_split/202409/202409_{x}.txt',
            # 添加更多文件路径
        ]
        model_output_dir = f'./output/lstm_emb16_hid128_nl2/lstm{x-1}'
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        # print(seq_length)
        train_lstm_model(seq_length, file_paths, model_output_dir)

def example_all(order_to_sequences):
    # 使用示例
    for x in range(3,14):
        seq_length = x-1
        model_output_dir = f'./output/lstm_emb16_hid128_nl2_all/lstm{x-1}'
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        # print(seq_length)
        train_lstm_model2(seq_length, order_to_sequences[x], model_output_dir)
if __name__ == "__main__":
    order_to_sequences = prepare_data()
    example_all(order_to_sequences)
