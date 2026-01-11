# 获取当前文件所在目录的父目录
import os
import sys

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# print(f"current_file: {current_file}")

# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file)
# print(f"current_dir: {current_dir}")

# 获取当前目录的父目录（项目根目录）
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
# print(f"parent_dir: {parent_dir}")

sys.path.append(parent_dir)
