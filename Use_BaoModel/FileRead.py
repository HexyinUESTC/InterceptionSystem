import sys
import json


def read_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            label = int(line.strip())  # 假设标签是整数类型
            labels.append(label)
    return labels


# 读取 JSON 文件
def read_json_file(plan_path):
    with open(plan_path, 'r') as f:
        read_json_data = json.load(f)
    return read_json_data
