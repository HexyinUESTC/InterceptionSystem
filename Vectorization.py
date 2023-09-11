import torch

# 递归构建查询计划树的特征向量
def construct_feature_vector(query_plan_tree):
    def encode_sub_plan(sub_plan):
        encoded = []
        node_type_id = int(sub_plan.get("Node Type ID", 0))
        encoded.append(node_type_id)
        encoded.append((sub_plan.get("Total Cost", 0) / 1000000))
        encoded.append((sub_plan.get("Plan Rows", 0)))

        for sub_sub_plan in sub_plan.get("Plans", []):
            encoded.extend(encode_sub_plan(sub_sub_plan))
        return encoded

    encoded_sequence = []
    encoded_sequence.extend(encode_sub_plan(query_plan_tree))

    MAX_LENGTH = 30  # 例如
    if len(encoded_sequence) < MAX_LENGTH:
        encoded_sequence += [0] * (MAX_LENGTH - len(encoded_sequence))
    elif len(encoded_sequence) > MAX_LENGTH:
        encoded_sequence = encoded_sequence[:MAX_LENGTH]
    return torch.tensor(encoded_sequence, dtype=torch.float32)
