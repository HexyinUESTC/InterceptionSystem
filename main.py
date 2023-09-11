import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import Net
from Use_BaoModel import FileRead
import Vectorization

# 获取文件路径，从命令行中获取
train_file_path = "data_about_plan/train"
train_label = "data_about_label/train/train_difference"
test_file_path = "data_about_plan/test"
test_label = "data_about_label/test/test_difference"

device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# 训练
def train(labels):
    num_epochs = 20
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for start_idx in range(0, len(labels), batch_size):
            end_idx = start_idx + batch_size
            batch_trees = []
            batch_labels = []

            for idx in range(start_idx, min(end_idx, len(labels))):
                train_json_filename = os.path.join(train_file_path, f"{idx}.json")
                with open(train_json_filename, 'r') as f:
                    train_json_data = json.load(f)

                train_plan_tree = train_json_data[0].get("Plan")
                feature_vector = Vectorization.construct_feature_vector(train_plan_tree)
                label = labels[idx]

                batch_trees.append(feature_vector)
                batch_labels.append(label)

            batch_trees = torch.stack(batch_trees)
            # print(batch_trees.shape)
            first_shape = batch_trees.shape[0]
            batch_trees = batch_trees.view(first_shape, 1, 30)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).unsqueeze(1)

            outputs = model(batch_trees)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")


def test(test_labels):
    model.eval()  # 设置为评估模式

    correct = 0
    total = 0

    for idx in range(len(test_labels)):
        test_json_filename = os.path.join(test_file_path, f"{idx}.json")
        with open(test_json_filename, 'r') as f:
            test_json_data = json.load(f)

        test_plan_tree = test_json_data[0].get("Plan")
        # feature_vector = Vectorization.construct_feature_vector(test_plan_tree).unsqueeze(0)
        feature_vector = Vectorization.construct_feature_vector(test_plan_tree)
        feature_vector = feature_vector.view(1, 1, 30)
        label = test_labels[idx]

        output = model(feature_vector.to(device))
        predicted = torch.sigmoid(output)  # 使用sigmoid函数将输出转换为0到1之间的值
        predicted_label = (predicted > 0.45).float()  # 设置阈值阈值

        total += 1
        correct += (predicted_label == label).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    model = Net.NeuralNetwork(30, 64).to(device)

    # 定义损失函数跟优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    batch_size = 20

    # 训练过程
    train(FileRead.read_labels(train_label))
    test(FileRead.read_labels(test_label))
