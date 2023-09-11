import torch.nn as nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 2

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播通过LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出并通过全连接层
        out = self.fc(out[:, -1, :])
        return out
