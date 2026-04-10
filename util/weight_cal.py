import torch

ckpt = torch.load('./checkpoints/dccrn/best_model.pth', map_location='cpu')
model_state = ckpt['model_state_dict']   # 提取模型参数

total_params = 0
for name, param in model_state.items():
    numel = param.numel()
    total_params += numel
    # 可选：打印每层信息，找出参数大户
    print(f"{name}: {tuple(param.shape)} -> {numel:,}")

print(f"模型总参数量: {total_params:,}")