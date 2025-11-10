from BTNHGV2Loader import loadData
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import HGTConv, Linear
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# 超参数配置
class Config:
    def __init__(self):
        # 模型参数
        self.hidden_dim = 64  # 隐藏层维度
        self.out_dim = 14  # 输出维度（分类数量）
        self.num_heads = 4  # 注意力头数
        self.dropout = 0.5  # Dropout比率
        self.layer_norm = True  # 是否使用层归一化
        self.leaky_relu_slope = 0.2  # LeakyReLU斜率
        
        # 训练参数
        self.lr = 0.005  # 学习率
        self.weight_decay = 5e-4  # 权重衰减
        self.epochs = 200  # 最大训练轮数
        self.early_stopping_patience = 15  # 早停耐心值
        
        # 其他参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_interval = 10  # 日志打印间隔

# 异构图注意力网络层
class HANLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, metadata, num_heads=4, dropout=0.5, layer_norm=True):
        super(HANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        
        # 为每种节点类型定义线性变换层
        self.node_lin = {}
        for node_type in self.node_types:
            self.node_lin[node_type] = Linear(in_dim, out_dim * num_heads)
        self.node_lin = torch.nn.ModuleDict(self.node_lin)
        
        # 为每种边类型定义注意力层
        self.convs = {}
        for edge_type in self.edge_types:
            src_type, rel_type, dst_type = edge_type
            self.convs[edge_type] = GATConv(
                (out_dim * num_heads, out_dim * num_heads),
                out_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False
            )
        self.convs = torch.nn.ModuleDict(self.convs)
        
        # 层归一化
        if self.layer_norm:
            self.norms = {}
            for node_type in self.node_types:
                self.norms[node_type] = torch.nn.LayerNorm(out_dim)
            self.norms = torch.nn.ModuleDict(self.norms)
        
        # 权重参数用于组合不同类型的边
        self.edge_type_weights = Parameter(torch.Tensor(len(self.edge_types), 1))
        torch.nn.init.xavier_uniform_(self.edge_type_weights.data)

    def forward(self, x_dict, edge_index_dict):
        # 节点特征的线性变换
        x_transformed = {}
        for node_type, x in x_dict.items():
            x_transformed[node_type] = self.node_lin[node_type](x).relu()
            x_transformed[node_type] = F.dropout(x_transformed[node_type], p=self.dropout, training=self.training)
        
        # 对每种边类型应用GAT卷积
        edge_type_outs = []
        for i, edge_type in enumerate(self.edge_types):
            src_type, rel_type, dst_type = edge_type
            if edge_type in edge_index_dict:
                # 只处理存在的边类型
                out = self.convs[edge_type](
                    (x_transformed[src_type], x_transformed[dst_type]),
                    edge_index_dict[edge_type]
                )
                # 应用边类型权重
                out = out * self.edge_type_weights[i]
                edge_type_outs.append(out)
        
        # 聚合不同边类型的输出
        out_dict = {}
        for node_type in self.node_types:
            # 收集所有包含当前节点类型作为目标的边的输出
            relevant_outs = []
            for i, edge_type in enumerate(self.edge_types):
                _, _, dst_type = edge_type
                if dst_type == node_type and edge_type in edge_index_dict:
                    relevant_outs.append(edge_type_outs[i] if len(edge_type_outs) > i else 0)
            
            if relevant_outs:
                out = sum(relevant_outs)
                if self.layer_norm:
                    out = self.norms[node_type](out)
                out_dict[node_type] = F.elu(out)
            else:
                # 如果没有入边，使用原始特征
                out_dict[node_type] = x_transformed[node_type].mean(dim=1, keepdim=True) if len(x_transformed[node_type].shape) > 1 else x_transformed[node_type]
        
        return out_dict

# HAN模型
class HAN(torch.nn.Module):
    def __init__(self, metadata, in_dim_dict, config):
        super(HAN, self).__init__()
        self.config = config
        self.in_dim_dict = in_dim_dict
        
        # 为每种节点类型定义输入投影层
        self.input_proj = {}
        for node_type, in_dim in in_dim_dict.items():
            self.input_proj[node_type] = Linear(in_dim, config.hidden_dim)
        self.input_proj = torch.nn.ModuleDict(self.input_proj)
        
        # 定义HAN层
        self.layer1 = HANLayer(
            config.hidden_dim, 
            config.hidden_dim, 
            metadata,
            num_heads=config.num_heads,
            dropout=config.dropout,
            layer_norm=config.layer_norm
        )
        
        self.layer2 = HANLayer(
            config.hidden_dim, 
            config.hidden_dim, 
            metadata,
            num_heads=config.num_heads,
            dropout=config.dropout,
            layer_norm=config.layer_norm
        )
        
        # 输出层，只关心address节点的分类
        self.out_proj = Linear(config.hidden_dim, config.out_dim)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        # 输入投影
        x_proj = {}
        for node_type, x in x_dict.items():
            x_proj[node_type] = self.input_proj[node_type](x).relu()
        
        # HAN层
        x_out = self.layer1(x_proj, edge_index_dict)
        x_out = self.layer2(x_out, edge_index_dict)
        
        # 输出层，只返回address节点的预测
        return self.out_proj(x_out['address'])

# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 训练函数
def train(model, data, optimizer, config, train_mask, val_mask):
    model.train()
    optimizer.zero_grad()
    
    out = model(data)
    loss = F.cross_entropy(out[train_mask], data['address'].y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    # 计算训练指标
    pred = out[train_mask].argmax(dim=1)
    acc = (pred == data['address'].y[train_mask]).sum() / train_mask.sum()
    
    return loss.item(), acc.item()

# 评估函数
def evaluate(model, data, mask):
    model.eval()
    
    with torch.no_grad():
        out = model(data)
        loss = F.cross_entropy(out[mask], data['address'].y[mask])
        
        pred = out[mask].argmax(dim=1)
        acc = (pred == data['address'].y[mask]).sum() / mask.sum()
        
        # 计算其他指标
        y_true = data['address'].y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
    
    return loss.item(), acc.item(), f1, precision, recall

# 主函数示例
def main(data_loader_function):
    # 初始化配置
    config = Config()
    
    # 加载数据（这里假设您有一个数据加载函数）
    data = data_loader_function()
    data = data.to(config.device)
    
    # 准备训练和验证掩码
    # 假设address节点的y中-1表示没有标签
    has_label = data['address'].y != -1
    # 随机分割训练集和验证集
    num_labeled = has_label.sum().item()
    train_idx = torch.randperm(num_labeled)[:int(0.8 * num_labeled)]
    val_idx = torch.randperm(num_labeled)[int(0.8 * num_labeled):]
    
    train_mask = torch.zeros_like(has_label, dtype=torch.bool)
    val_mask = torch.zeros_like(has_label, dtype=torch.bool)
    
    labeled_indices = torch.nonzero(has_label).squeeze()
    train_mask[labeled_indices[train_idx]] = True
    val_mask[labeled_indices[val_idx]] = True
    
    # 设置数据的训练和验证掩码
    data['address'].train_mask = train_mask
    data['address'].val_mask = val_mask
    
    # 获取每种节点类型的输入维度
    in_dim_dict = {}
    for node_type in data.node_types:
        # 假设特征矩阵是二维的，第一维是节点数，第二维是特征数
        in_dim_dict[node_type] = data[node_type].x.size(1) if hasattr(data[node_type], 'x') and data[node_type].x is not None else 1
    
    # 初始化模型
    metadata = (data.node_types, data.edge_types)
    model = HAN(metadata, in_dim_dict, config).to(config.device)
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
    
    # 训练循环
    best_val_acc = 0
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train(model, data, optimizer, config, train_mask, val_mask)
        
        # 验证
        val_loss, val_acc, val_f1, val_precision, val_recall = evaluate(model, data, val_mask)
        
        # 更新最佳验证精度
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # 打印日志
        if epoch % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '\
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {elapsed:.2f}s')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    # 在测试集上评估（如果有测试集）
    # test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(model, data, test_mask)
    # print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
    
    print(f'Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}')
    
    return model

if __name__ == "__main__":
    # 这里需要您提供数据加载函数
    # 假设您有一个名为load_data的函数来加载BTNHGV2Loader.py中的HeteroData数据
    # model = main(load_data)
    pass