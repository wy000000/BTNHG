import set_parent_dir
import torch

A = torch.tensor([[1, 2], [3, 4]])  # [2, 2]
B = torch.tensor([5, 6])            # [2] → 广播为 [1, 2] → 再广播为 [2, 2]
C = A + B                          # 结果: [[6, 8], [8, 10]]

print(f"C: {C}")
