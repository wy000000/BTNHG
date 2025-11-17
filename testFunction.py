import torch
import numpy as np

def compare_TwoX_InLine(x1, x2):
	# 遍历比较
	all_exist = True
	for i in range(x1.size(0)):
		# 当前元素
		xi = x1[i]

		# 构造掩码：忽略 NaN 的位置
		mask = ~torch.isnan(xi)

		# 在 x2 中逐行比较，只比较非 NaN 的位置
		match = torch.any(torch.all(x2[:, mask] == xi[mask], dim=1))

		if not match:
			print(f"索引 {i} 不存在于 train_loader2 中")
			print("内容:", xi)
			all_exist = False
			break

	if all_exist:
		print("✅ 所有元素都存在于 train_loader2 中（忽略 NaN）")
		return True
	else:
		return False

def compare_ignore_nan(x1, x2):
    """
    比较两个张量/数组，忽略 NaN。
    如果发现第一个不相等的位置，则打印索引和值，并返回 False。
    如果全部相等，则返回 True。
    如果形状不同，直接返回 False。
    """
    # 转换为 numpy 以统一处理
    if isinstance(x1, torch.Tensor):
        x1 = x1.cpu().numpy()
    if isinstance(x2, torch.Tensor):
        x2 = x2.cpu().numpy()
    
    # 如果形状不同，直接返回 False
    if x1.shape != x2.shape:
        print(f"形状不同: x1.shape={x1.shape}, x2.shape={x2.shape}")
        return False
    
    # 遍历比较
    it = np.nditer([x1, x2], flags=['multi_index'])
    for a, b in it:
        idx = it.multi_index
        # 忽略 NaN
        if np.isnan(a) and np.isnan(b):
            continue
        if a != b:
            print(f"第一个不相等的位置: 索引={idx}")
            print(f"x1:", a)
            print(f"x2:", b)
            return False
    return True