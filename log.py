import pandas as pd
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import numpy as np

class TrainingLogger:
    def __init__(self):
        # logs[method][phase] = list of dicts
        self.logs = defaultdict(lambda: defaultdict(list))
    
    def log(self, method_name, phase, epoch, **metrics):
        """
        记录日志
        method_name: 方法名 (如 'HAN', 'GCN')
        phase: 阶段 ('train', 'val', 'test')
        epoch: 当前轮数
        **metrics: 各种指标 (如 loss, acc, f1, precision, recall 等)
        """
        # 验证 phase 参数有效性
        if phase not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid phase: {phase}. Must be one of ['train', 'val', 'test']")
        
        # 验证 epoch 参数有效性
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"Epoch must be a non-negative integer, got {epoch}")
        
        # 创建日志条目
        entry = {"epoch": epoch}
        entry.update(metrics)
        
        self.logs[method_name][phase].append(entry)
    
    def to_dataframe(self, method_name):
        """
        将某个方法的日志转为 DataFrame，方便分析
        """
        dfs = {}
        for phase, entries in self.logs[method_name].items():
            dfs[phase] = pd.DataFrame(entries)
        return dfs
    
    def compare_methods(self):
        """
        返回所有方法的日志，方便对比
        """
        result = {}
        for method, phases in self.logs.items():
            result[method] = {phase: pd.DataFrame(entries) for phase, entries in phases.items()}
        return result
    
    def save_logs(self, filepath):
        """
        将日志保存到文件
        filepath: 保存路径
        """
        # 转换 defaultdict 为普通 dict 以便序列化
        serializable_logs = {}
        for method, phases in self.logs.items():
            serializable_logs[method] = {}
            for phase, entries in phases.items():
                serializable_logs[method][phase] = entries
        
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_logs, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_logs(cls, filepath):
        """
        从文件加载日志
        filepath: 日志文件路径
        """
        logger = cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            serializable_logs = json.load(f)
        
        # 转换回 defaultdict 格式
        for method, phases in serializable_logs.items():
            for phase, entries in phases.items():
                for entry in entries:
                    # 提取 epoch 和其他指标
                    epoch = entry.pop('epoch', 0)
                    logger.log(method, phase, epoch, **entry)
        
        return logger
    
    def get_best_metrics(self, method_name, phase, metric, mode='min'):
        """
        获取指定方法在指定阶段的最佳指标值
        method_name: 方法名
        phase: 阶段
        metric: 指标名称
        mode: 'min' 表示寻找最小值, 'max' 表示寻找最大值
        """
        if method_name not in self.logs or phase not in self.logs[method_name]:
            raise ValueError(f"No logs found for method '{method_name}' and phase '{phase}'")
        
        df = pd.DataFrame(self.logs[method_name][phase])
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in logs")
        
        if mode == 'min':
            best_value = df[metric].min()
            best_epoch = df[df[metric] == best_value]['epoch'].iloc[0]
        else:  # mode == 'max'
            best_value = df[metric].max()
            best_epoch = df[df[metric] == best_value]['epoch'].iloc[0]
        
        return best_value, best_epoch
    
    def generate_summary(self):
        """
        生成所有方法的性能摘要
        """
        summary = {}
        
        for method in self.logs:
            summary[method] = {}
            
            for phase in self.logs[method]:
                df = pd.DataFrame(self.logs[method][phase])
                metrics = [col for col in df.columns if col != 'epoch']
                
                phase_summary = {}
                for metric in metrics:
                    phase_summary[f'{metric}_min'] = df[metric].min()
                    phase_summary[f'{metric}_max'] = df[metric].max()
                    phase_summary[f'{metric}_mean'] = df[metric].mean()
                    phase_summary[f'{metric}_std'] = df[metric].std()
                    
                    # 找到最佳值对应的轮数
                    if metric == 'loss':
                        best_idx = df[metric].idxmin()
                    else:
                        best_idx = df[metric].idxmax()
                    
                    phase_summary[f'{metric}_best_epoch'] = df['epoch'].iloc[best_idx]
                
            summary[method][phase] = phase_summary
        
        return summary
    
    def print_summary(self):
        """
        打印性能摘要
        """
        summary = self.generate_summary()
        
        for method, phases in summary.items():
            print(f"\n=== {method} ===")
            
            for phase, metrics in phases.items():
                print(f"  --- {phase} ---")
                
                # 按指标分组显示
                metric_groups = {}
                for key, value in metrics.items():
                    metric_name = key.split('_')[0]
                    if metric_name not in metric_groups:
                        metric_groups[metric_name] = {}
                    metric_groups[metric_name][key] = value
                
                for metric_name, values in metric_groups.items():
                    print(f"    {metric_name}:")
                    print(f"      Min: {values.get(f'{metric_name}_min', 'N/A'):.6f}")
                    print(f"      Max: {values.get(f'{metric_name}_max', 'N/A'):.6f}")
                    print(f"      Mean: {values.get(f'{metric_name}_mean', 'N/A'):.6f}")
                    print(f"      Std: {values.get(f'{metric_name}_std', 'N/A'):.6f}")
                    print(f"      Best at epoch: {values.get(f'{metric_name}_best_epoch', 'N/A')}")
    
    def should_stop_early(self, method_name, phase, metric, patience=5, mode='min'):
        """
        根据早停机制判断是否应该停止训练
        method_name: 方法名
        phase: 阶段
        metric: 指标名称
        patience: 耐心值，表示连续多少轮没有改善就停止
        mode: 'min' 表示指标越小越好, 'max' 表示指标越大越好
        """
        if method_name not in self.logs or phase not in self.logs[method_name]:
            return False
        
        df = pd.DataFrame(self.logs[method_name][phase])
        
        if metric not in df.columns or len(df) <= patience:
            return False
        
        # 按 epoch 排序
        df = df.sort_values('epoch')
        
        # 获取最近 patience+1 轮的指标
        recent_metrics = df[metric].tail(patience + 1).values
        
        if mode == 'min':
            # 检查是否最近 patience 轮都没有比最早的那轮好
            return not any(recent_metrics[i] < recent_metrics[0] for i in range(1, len(recent_metrics)))
        else:  # mode == 'max'
            # 检查是否最近 patience 轮都没有比最早的那轮好
            return not any(recent_metrics[i] > recent_metrics[0] for i in range(1, len(recent_metrics)))
    
    def to_latex_table(self, methods=None, phases=None, metrics=None, output_file=None):
        """
        将日志导出为 LaTeX 表格
        methods: 要包含的方法列表 (默认全部方法)
        phases: 要包含的阶段列表 (默认全部阶段)
        metrics: 要包含的指标列表 (默认全部指标)
        output_file: 输出文件路径 (None 表示返回字符串)
        """
        if methods is None:
            methods = list(self.logs.keys())
        
        # 收集所有阶段和指标
        all_phases = set()
        all_metrics = set()
        
        for method in methods:
            if method in self.logs:
                for phase in self.logs[method]:
                    all_phases.add(phase)
                    df = pd.DataFrame(self.logs[method][phase])
                    for col in df.columns:
                        if col != 'epoch':
                            all_metrics.add(col)
        
        if phases is None:
            phases = list(all_phases)
        
        if metrics is None:
            metrics = list(all_metrics)
        
        # 生成 LaTeX 表格
        header = "\\begin{table}[h]\n\\centering\\caption{Model Performance Comparison}"
        header += "\\begin{tabular}{l"
        header += "c" * (len(phases) * len(metrics)) + "}"
        header += "\\toprule\n"
        
        # 添加列标题
        col_titles = ["Method"]
        for phase in phases:
            for metric in metrics:
                col_titles.append(f"{phase}-{metric}")
        
        header += " u0026 ".join(col_titles) + " \\\\n\\midrule\n"
        
        # 添加数据行
        rows = []
        for method in methods:
            row = [method]
            
            for phase in phases:
                for metric in metrics:
                    if method in self.logs and phase in self.logs[method]:
                        df = pd.DataFrame(self.logs[method][phase])
                        if metric in df.columns:
                            # 获取最佳值
                            if metric == 'loss':
                                value = df[metric].min()
                            else:
                                value = df[metric].max()
                            row.append(f"{value:.4f}")
                        else:
                            row.append("-")
                    else:
                        row.append("-")
            
            rows.append(" u0026 ".join(row) + " \\\\")
        
        # 构建完整表格
        latex_table = header + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}"
        
        # 输出到文件或返回字符串
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_table)
            return f"LaTeX table saved to {output_file}"
        else:
            return latex_table

def plot_training_curves(logger, methods=None, metric="loss", phase="val", 
                         smooth=False, save_path=None, show=True):
    """
    绘制训练曲线对比
    logger: TrainingLogger 实例
    methods: 要对比的方法列表 (默认全部方法)
    metric: 'loss' 或 'acc' 或其他已记录的指标
    phase: 'train' 或 'val' 或 'test'
    smooth: 是否对曲线进行平滑处理
    save_path: 保存图像的路径 (None 表示不保存)
    show: 是否显示图像
    """
    if methods is None:
        methods = list(logger.logs.keys())
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab10', len(methods))  # 获取颜色映射
    
    for i, method in enumerate(methods):
        if phase not in logger.logs[method]:
            print(f"Warning: {method} has no {phase} logs")
            continue
        
        df = pd.DataFrame(logger.logs[method][phase])
        
        # 检查指标是否存在
        if metric not in df.columns:
            print(f"Warning: {method} has no {metric} logs")
            continue
        
        # 平滑处理
        if smooth and len(df) > 10:
            # 使用移动平均进行平滑
            window_size = min(5, len(df) // 10)  # 窗口大小自适应
            values = df[metric].rolling(window=window_size, min_periods=1).mean()
            label = f"{method}-{phase} (smoothed)"
        else:
            values = df[metric]
            label = f"{method}-{phase}"
        
        # 绘制曲线
        plt.plot(df["epoch"], values, label=label, color=colors(i))
        
        # 标记最佳值
        if metric == "loss":
            best_idx = values.idxmin()
        else:
            best_idx = values.idxmax()
        
        plt.scatter(df["epoch"][best_idx], values.iloc[best_idx], 
                   color=colors(i), marker='*', s=100)
        plt.annotate(f"Best: {values.iloc[best_idx]:.4f}",
                    (df["epoch"][best_idx], values.iloc[best_idx]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{phase.capitalize()} {metric} Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)  # 只保留一行网格线设置
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # 显示图像
    if show:
        plt.show()
    else:
        plt.close()  # 不显示时关闭图像，释放资源


# 仅在直接运行此脚本时执行示例代码
if __name__ == "__main__":
    # 创建日志记录器
    logger = TrainingLogger()
    
    # 记录日志（支持多种指标）
    logger.log('HAN', 'train', epoch=1, loss=0.65, acc=0.72, f1=0.68)
    logger.log('HAN', 'val', epoch=1, loss=0.62, acc=0.75, f1=0.71)
    logger.log('GCN', 'train', epoch=1, loss=0.68, acc=0.70, f1=0.65)
    logger.log('GCN', 'val', epoch=1, loss=0.66, acc=0.72, f1=0.68)
    
    # 绘制平滑的训练曲线并保存
    plot_training_curves(logger, metric='loss', phase='val', smooth=True, 
                         save_path='results/val_loss_comparison.png')
    
    # 检查早停条件
    if logger.should_stop_early('HAN', 'val', 'loss', patience=5):
        print("Early stopping triggered!")
    
    # 打印性能摘要
    logger.print_summary()
    
    # 保存日志到文件
    logger.save_logs('results/training_logs.json')
    
    # 导出为LaTeX表格
    latex_table = logger.to_latex_table(output_file='results/performance_table.tex')