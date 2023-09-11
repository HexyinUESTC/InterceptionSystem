import torch.cuda
import torch.nn as nn
from TreeConvolution.tcnn import BinaryTreeConv, TreeLayerNorm
from TreeConvolution.tcnn import TreeActivation, DynamicPooling
from TreeConvolution.util import prepare_trees


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]


def right_child(x):
    if len(x) != 3:
        return None
    return x[2]


def features(x):
    return x[0]


class BaoNet(nn.Module):
    def __init__(self, in_channels):
        super(BaoNet, self).__init__()
        self.__in_channels = in_channels
        if torch.cuda.is_available():
            self.__cuda = True
        else:
            self.__cuda = False

        self.tree_conv = nn.Sequential(
            BinaryTreeConv(self.__in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        # self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        #
        # self.linear = nn.Sequential(
        #     nn.Linear(32, 16),
        #     nn.LeakyReLU(),
        #     nn.Linear(16, 1)
        # )

    def in_channels(self):
        return self.__in_channels
        
    def forward(self, x):
        trees = prepare_trees(x, features, left_child, right_child,
                              cuda=self.__cuda)
        return self.tree_conv(trees)
        # lstm_input = self.tree_conv(trees)
        # h0 = torch.zeros(2, lstm_input.size(0), 32)
        # c0 = torch.zeros(2, lstm_input.size(0), 32)
        #
        # first_shape = lstm_input.shape[0]
        # lstm_input = lstm_input.view(first_shape, 1, 64)
        # lstm_output, _ = self.lstm(lstm_input, (h0, c0))
        #
        # # lstm_output, _ = self.lstm(out)
        # output = self.linear(lstm_output[:, -1, :])  # Take the last time step's output
        # return output
