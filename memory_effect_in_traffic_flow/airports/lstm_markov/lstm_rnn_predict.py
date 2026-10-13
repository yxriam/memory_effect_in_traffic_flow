import os
import torch
import torch.nn as nn
import json
import re
from torch.utils.data import DataLoader
import torch.optim as optim
from model.LSTM import LSTMModel
from model.RNN import RNNModel
from dataloader.Dataset import AirportDataset
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 配置 =====
dict_path = './airport_dict.json'
file_paths = [
    './sequence_split/202409/202409_3.txt',
]
seq_length = 3
embedding_dim = 36
hidden_dim = 32
num_layers = 5
num_epochs = 30
batch_size = 32
lr = 0.001

# ===== 数据准备 =====
with open(dict_path, 'r', encoding='utf-8') as f:
    airport_to_city = json.load(f)
airport_to_index = {airport: idx for idx, airport in enumerate(airport_to_city)}
index_to_airport = {idx: airport for airport, idx in airport_to_index.items()}
num_classes = len(airport_to_index)
pattern = re.compile(r'\b[A-Z]{3}\b')

dataset = AirportDataset(file_paths, airport_to_index, pattern, seq_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# ===== 训练循环 =====
def train_one_model(model, model_name):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    out_dir = f'./output/{model_name.lower()}'
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, -1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, -1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, num_classes), targets.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'[{model_name}] Epoch {epoch+1}: Training Loss={loss.item():.4f}, Validation Loss={avg_val_loss:.4f}')
        # 保存最优
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, f'{model_name.lower()}_seq3_best.pth'))
        torch.save(model.state_dict(), os.path.join(out_dir, f'{model_name.lower()}_seq3_last{epoch+1}.pth'))
    print(f'[{model_name}] Training Done. Best Val Loss: {best_val_loss:.4f}')
    torch.save(model.state_dict(), os.path.join(out_dir, f'{model_name.lower()}_seq3_final.pth'))

# 初始化模型
lstm_model = LSTMModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)
rnn_model = RNNModel(num_classes, embedding_dim, hidden_dim, num_layers).to(device)

# 训练LSTM和RNN
train_one_model(lstm_model, "LSTM")
train_one_model(rnn_model, "RNN")


# ===== 连续1000步预测函数 =====
def continuous_predict(model, seed_seq, airport_to_index, index_to_airport, steps=1000):
    model.eval()
    input_idxs = [airport_to_index[code] for code in seed_seq]
    results = input_idxs.copy()
    with torch.no_grad():
        for _ in range(steps):
            inp = torch.tensor([results[-seq_length:]], dtype=torch.long, device=device)  # shape: (1, seq_length)
            out = model(inp)
            next_idx = torch.argmax(out, dim=1).item()
            results.append(next_idx)
    return [index_to_airport[idx] for idx in results]

# ===== 返回率计算函数 =====
def return_rate(pred_seq, n):
    """
    n=2: 两步返回率。即当前位置与前n步是否相同
    n=3: 三步返回率。
    """
    total = len(pred_seq) - n
    count = sum(1 for i in range(n, len(pred_seq)) if pred_seq[i] == pred_seq[i-n])
    return count / total if total > 0 else 0

# ===== 进行1000步预测并统计 =====
def evaluate_model(model, model_name):
    # 选一组真实起始种子
    # 用你自己的真实序列起始，例如["PEK","SHA","CAN"]，这里举例用数据集第0条
    seed_seq = [index_to_airport[idx.item()] for idx in dataset[0][:seq_length]]
    print(f"[{model_name}] Seed sequence: {seed_seq}")
    pred_seq = continuous_predict(model, seed_seq, airport_to_index, index_to_airport, steps=1000)
    two_step = return_rate(pred_seq, 2)
    three_step = return_rate(pred_seq, 3)
    print(f"[{model_name}] 1000步预测 两步返回率: {two_step:.4f}，三步返回率: {three_step:.4f}")
    # 可以保存预测序列
    with open(f'./output/{model_name.lower()}_seq3_1000pred.txt', 'w') as f:
        f.write(' '.join(pred_seq))
    return two_step, three_step

# # ===== 加载最优权重进行1000步预测与返回率评测 =====
lstm_model.load_state_dict(torch.load('../output/lstm/lstm_seq3_best.pth', map_location=device))
rnn_model.load_state_dict(torch.load('../output/rnn/rnn_seq3_best.pth', map_location=device))

evaluate_model(lstm_model, "LSTM")
evaluate_model(rnn_model, "RNN")


# ===== 计算两步返回率和三步返回率 =====
def calculate_return_rates(sequence: list) -> dict:
    """
    计算两步返回率和三步返回率
    :param sequence: 输入序列（list，可以是机场代码字符串列表）
    :return: 字典 { 'two_step_return': float, 'three_step_return': float }
    """
    n = len(sequence)
    if n < 3:
        return {'two_step_return': 0.0, 'three_step_return': 0.0}

    # 两步返回率：隔一个位置相同的数量 / 总数量
    two_step_count = sum(1 for i in range(n - 2) if sequence[i] == sequence[i + 2])
    two_step_rate = two_step_count / (n - 2)

    # 三步返回率：隔两个位置相同的数量 / 总数量
    three_step_count = sum(1 for i in range(n - 3) if sequence[i] == sequence[i + 3])
    three_step_rate = three_step_count / (n - 3)

    return {
        'two_step_return': two_step_rate,
        'three_step_return': three_step_rate
    }

with open("./output/lstm_seq3_1000pred.txt", "r", encoding="utf-8") as f:
    content = f.read()
# 按空格分割
sequence = content.strip().split()
rates = calculate_return_rates(sequence)
print("两步返回率:", rates['two_step_return'])
print("三步返回率:", rates['three_step_return'])