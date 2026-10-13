import torch
import torch.nn as nn

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        # x = self.fc(x)
        x = self.fc(x[:, -1, :]) # 最后一个时间步的隐藏状态
        return x