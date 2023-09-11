import json
import os
# import sys
import torch
import torch.nn as nn
import torch.optim as optim
import FileRead
import Bao_net
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from featurize import TreeFeaturizer
from torch.utils.data import DataLoader

# import torch.utils.data as data

# 获取文件路径，从命令行中获取
train_file_path = "../data_about_plan/train"
train_label = "../data_about_label/train/train_difference"
test_file_path = "../data_about_plan/test"
test_label = "../data_about_label/test/test_difference"

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.tensor(targets)
    return trees, targets


def _inv_log1p(x):
    return np.exp(x) - 1


class HeXueYu:
    def __init__(self):
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()
        self.__net = None
        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        self.__tree_transform = TreeFeaturizer()
        self.__in_channels = None
        self.__n = 0

    def train(self, labels):
        batch_trees = []
        batch_labels = []

        # 加载数据
        for idx in range(0, len(labels)):
            train_json_filename = os.path.join(train_file_path, f"{idx}.json")
            with open(train_json_filename, 'r') as f:
                train_json_data = json.load(f)
            plan_tree = train_json_data[0]
            train_label = np.array(labels[idx])

            batch_trees.append(plan_tree)
            batch_labels.append(train_label)
        # 数据预处理
        plan_tree = [json.loads(x) if isinstance(x, str) else x for x in batch_trees]
        self.__tree_transform.fit(plan_tree)
        batch_trees = self.__tree_transform.transform(plan_tree)

        if isinstance(batch_labels, list):
            batch_labels = np.array(batch_labels)
        batch_labels = self.__pipeline.fit_transform(batch_labels.reshape(-1, 1)).astype(np.float32)

        pairs = list(zip(batch_trees, batch_labels))
        dataset = DataLoader(pairs,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=collate)

        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break
        self.__in_channels = in_channels
        self.__net = Bao_net.BaoNet(self.__in_channels)

        # 定义优化器以及损失函数
        criterion = nn.BCEWithLogitsLoss()  # 更换损失函数为BCELoss后准确率变为33.1%
        optimizer = optim.Adam(self.__net.parameters())

        losses = []
        for epoch in range(80):
            loss_accum = 0
            for x, y in dataset:
                # if CUDA:
                #     y = y.cuda()
                y = y.to(device)
                y_pred = self.__net(x)
                loss = criterion(y_pred, y)
                loss_accum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
            losses.append(loss_accum)
            if epoch % 15 == 0:
                print("Epoch", epoch, "training loss:", loss_accum)


    def test(self, test_labels):
        self.__net.eval()  # 设置为评估模式

        correct = 0
        total = 0

        for idx in range(len(test_labels)):
            test_json_filename = os.path.join(test_file_path, f"{idx}.json")
            with open(test_json_filename, 'r') as f:
                test_json_data = json.load(f)

            test_plan_tree = test_json_data
            plan_tree = [json.loads(x) if isinstance(x, str) else x for x in test_plan_tree]
            featurize = TreeFeaturizer()
            featurize.fit(plan_tree)
            test_tree = featurize.transform(test_plan_tree)

            label = test_labels[idx]

            output = self.__net(test_tree)
            predicted = torch.sigmoid(output)  # 使用sigmoid函数将输出转换为0到1之间的值
            predicted_label = (predicted > 0.45).float()  # 使用0.5作为阈值

            total += 1
            correct += (predicted_label == label).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    hexueyu = HeXueYu()

    # 训练过程
    hexueyu.train(FileRead.read_labels(train_label))
    hexueyu.test(FileRead.read_labels(test_label))
