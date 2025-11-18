#pip install torch torchvision torchaudio
#pip install torch-geometric
print("start")
import time
# def time_import(import_statement, description):
#     print(f"{description}")
#     start_time = time.time()
#     exec(import_statement)
#     end_time = time.time()
#     elapsed = (end_time - start_time) * 1000  # 转换为毫秒
#     print(f"{description}: {elapsed:.4f} 毫秒")
# print("start import")
# time_import('import torch', '导入torch')
# time_import('from torch_geometric.data import Data', '从torch_geometric.data导入Data')
# time_import('from torch_geometric.loader import DataLoader', '从torch_geometric.loader导入DataLoader')
# time_import('from torch_geometric.nn import GCNConv, global_mean_pool', '从torch_geometric.nn导入GCNConv和global_mean_pool')
# time_import('import torch.nn.functional as F', '导入torch.nn.functional并命名为F')
print("start import")
import torch
from torch_geometric.data import Data # 从torch_geometric.data导入Data类，用于表示图数据
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 检查CUDA是否可用，若可用则使用CUDA设备，否则使用CPU设备
print(f"使用设备: {device}")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


#构造图数据集（3个图形）
# 三角形图（3节点，3边）
triangle = Data(
    x=torch.eye(3, dtype=torch.float),  # 3节点，3维特征（one-hot编码）
    # x=torch.tensor([
    #     [1.0, 3],  # 节点0的特征: 原始特征+附加特征。第一个特征没用。
    #     [1.0, 3],  # 节点1的特征: 原始特征+附加特征
    #     [1.0, 3]   # 节点2的特征: 原始特征+附加特征
    # ]),  # 每个节点2维特征
    edge_index=torch.tensor([[0, 1, 1, 2, 2, 0],
                             [1, 0, 2, 1, 0, 2]], dtype=torch.long), # 3边，2维边索引
    y=torch.tensor([0])  # 类别0
)

# 正方形图（4节点，4边）
square = Data(
    x=torch.eye(4, dtype=torch.float),  #one-hot编码
    # x=torch.tensor([
    #     [1.0, 4],  # 节点0的特征: 原始特征+附加特征
    #     [1.0, 4],
    #     [1.0, 4],
    #     [1.0, 4]
    # ]),  
    edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                             [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long),
    y=torch.tensor([1])  # 类别1
)

# 五边形图（5节点，5边）
pentagon = Data(
    x=torch.eye(5, dtype=torch.float),  #one-hot编码
    # x=torch.tensor([
    #     [1.0, 5],  # 节点0的特征: 原始特征+附加特征
    #     [1.0, 5],
    #     [1.0, 5],
    #     [1.0, 5],
    #     [1.0, 5]
    # ]),  
    edge_index=torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
    ], dtype=torch.long),
    y=torch.tensor([2])  # 类别2
)

print("construct dataset")
# 构建数据集
# dataset = [triangle, square, pentagon]
# loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 定义一个函数，用0填充节点特征
def pad_node_features(data, target_dim=5):
    # 原始特征维度
    orig_dim = data.x.size(1) # 原始特征维度
    if orig_dim < target_dim:
        pad_size = target_dim - orig_dim
        padding = torch.zeros((data.x.size(0), pad_size), dtype=torch.float) # 用0填充缺失维度
        data.x = torch.cat([data.x, padding], dim=1) # 填充节点特征
    return data

# 定义一个函数，添加节点掩码
# def add_node_mask(data):    
#     data.node_mask = torch.ones(data.x.size(0), dtype=torch.bool)
#     return data

max_dim = 5
# dataset = [pad_node_features(add_node_mask(graph), target_dim=max_dim)
dataset = [pad_node_features(graph, target_dim=max_dim) # 填充节点特征
            for graph in [triangle, square, pentagon]]
loader = DataLoader(dataset, batch_size=1, shuffle=True)
#输出各图的节点特征和掩码
for graph in dataset:
    print(f"类别: {graph.y.item()}, 节点特征: {graph.x}")#, 节点掩码: {graph.node_mask}")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


#定义图神经网络模型（GCN）
class GNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 16)  # 输入层
        self.conv2 = GCNConv(16, 32) # 隐藏层
        self.fc = nn.Linear(32, 3)  # 输出3类

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index)) # 输入层激活
        x = F.relu(self.conv2(x, edge_index)) # 隐藏层激活
        x = global_mean_pool(x, batch)  # 图级聚合
        return self.fc(x) # 输出层

print("start train")
#训练模型
model = GNN().to(device)  # 将模型移动到CUDA设备
optimizer = torch.optim.Adam(model.parameters(), lr=0.03) # 优化器
criterion = nn.CrossEntropyLoss() # 损失函数

# 早停机制参数设置
patience = 10  # 当连续10个epoch损失不再改善时停止训练
best_loss = float('inf')
counter = 0
total_loss = 0

# 记录训练开始时间
train_start_time = time.time()

for_time = time.time()
for epoch in range(500):
    epoch_start_time = time.time()  # 记录每个epoch开始时间
    total_loss = 0
    for data in loader:
        # print(f"类别: {data.y.item()}, 特征: {data.x}")#打印data的类别和特征数据
        data = data.to(device)  # 将数据移动到CUDA设备
        # data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)  # 单图处理，确保batch也在CUDA上
        # print(data.y.cpu())  # 将CUDA张量转换为CPU张量后再打印
        out = model(data) # 前向传播
        loss = criterion(out, data.y) # 计算损失
        optimizer.zero_grad() # 清除之前的梯度
        loss.backward() # 计算梯度
        optimizer.step() # 更新模型参数
        total_loss += loss.item() # 累加每个batch的损失
    
    epoch_end_time = time.time()  # 记录每个epoch结束时间    
    elapsed_time = epoch_end_time - train_start_time  # 计算从训练开始到当前epoch结束的累计已用时间
    
    # 早停机制检查
    # if total_loss < best_loss:
    #     best_loss = total_loss
    #     counter = 0
    #     # 保存最佳模型
    #     torch.save(model.state_dict(), 'best_model.pth')
    # else:
    #     counter += 1
        
    if epoch % 50 == 0:
        for_time = epoch_end_time - for_time  
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, for Time: {for_time:.4f}s, 已用时间: {elapsed_time:.4f}s")
        for_time = time.time()
    # 检查是否需要早停
    # if counter >= patience:
    #     print(f"Early stopping at epoch {epoch}")
    #     break
    if total_loss<0.01:
        break
for_time = time.time()-for_time
print(f"Epoch {epoch}, Loss: {total_loss:.4f}, for Time: {for_time:.4f}s, 已用时间: {elapsed_time:.4f}s")

# 记录训练结束时间并计算总耗时
train_end_time = time.time()
total_train_time = train_end_time - train_start_time
print(f"总训练时间: {total_train_time:.4f}秒")
print(f"平均每个epoch耗时: {total_train_time/(epoch+1):.4f}秒")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")


# 加载最佳模型
# model.load_state_dict(torch.load('best_model.pth', map_location=device))

# 预测代码增强版本
print("start test")
#输出预测结果
model.eval()
correct = 0
with torch.no_grad():
    for data in loader:
        data = data.to(device)  # 将测试数据也移动到CUDA设备
        # data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device) # 单图处理，确保batch也在CUDA上
        # 获取模型原始输出（logits）
        logits = model(data) # 前向传播
        # 应用softmax获取置信度
        probabilities = F.softmax(logits, dim=1) # 对logits应用softmax，得到每个类别的置信度
        # 获取预测类别
        pred = logits.argmax(dim=1) # 获取logits中最大值的索引，即预测类别
        # 获取预测类别的置信度
        confidence = probabilities[0, pred.item()].item() # 对应预测类别的置信度
        print(f"预测类别: {pred.cpu().item()}, 实际类别: {data.y.cpu().item()}, 置信度: {confidence:.4f}")
        if pred.item() == data.y.item():
            correct += 1
    # 计算准确率
    accuracy = correct / len(dataset)
    print(f"测试准确率: {accuracy:.4f}")
print(f"当前时间: {time.strftime('%m-%d %H:%M:%S', time.localtime())}")