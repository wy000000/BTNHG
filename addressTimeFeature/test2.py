import numpy as np

# 示例数据：4 行 3 列
filtered_sample = np.array([
    [1, 1, 100],
    [1, 5, 200],
    [1, 6, 300],
    [1, 8, 400]
])

# 执行语句
diffs = filtered_sample[1:, 1] - filtered_sample[:-1, 1]

# 输出结果
print("diffs =", diffs)

mask = diffs != 1
print("mask =", mask)
# 将满足条件的行的第0个元素（索引0）设置为差值
filtered_sample[1:, 0][mask] = diffs[mask]
print("filtered_sample =", filtered_sample)  
