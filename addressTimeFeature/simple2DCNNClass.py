"""向后兼容别名：2-D CNN 基线已迁移到 ``CNN2D.py``。

保留本文件是为了让旧脚本/旧 notebook 中的
``from simple2DCNNClass import simple2DCNNClass`` 继续可用。
"""
from CNN2D import simple2DCNNClass  # noqa: F401

__all__ = ["simple2DCNNClass"]
