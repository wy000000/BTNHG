from BTNHGV2Loader import loadData
import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.loader import NeighborLoader

# =========================
# 超参数集中管理
# =========================
params = {
	"hidden_channels": 64,   # 隐藏层维度
	"out_channels": 32,      # 输出层维度
	"num_heads": 3,          # 注意力头数
	"dropout": 0.4,          # Dropout比例
	"lr": 0.005,             # 学习率
	"weight_decay": 1e-4,    # L2正则
	"epochs": 200,           # 最大训练轮数
	"patience": 20           # 早停容忍度
}
data = loadData(path=r"D:\BTNHG\BTNHGV2")
# =========================
# 定义 HAN 模型
# =========================
num_classes = int(data['address'].y.max().item()) + 1

class HAN(torch.nn.Module):
	def __init__(self, metadata, hidden_channels, out_channels, num_heads, dropout, num_classes):
		super().__init__()
		self.conv1 = HANConv(
			in_channels=-1,
			out_channels=hidden_channels,
			heads=num_heads,
			metadata=metadata
		)
		self.conv2 = HANConv(
			in_channels=hidden_channels * num_heads,
			out_channels=out_channels,
			heads=num_heads,
			metadata=metadata
		)
		self.lin = torch.nn.Linear(out_channels * num_heads, num_classes)
		self.dropout = dropout

	def forward(self, x_dict, edge_index_dict):
		# 第一次卷积：对每种关系做注意力聚合
		x_dict = self.conv1(x_dict, edge_index_dict)
		x_dict = {k: F.relu(v) for k, v in x_dict.items()}
		# Dropout 防止过拟合
		x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
		# 第二次卷积
		x_dict = self.conv2(x_dict, edge_index_dict)
		x_dict = {k: F.relu(v) for k, v in x_dict.items()}
		# 只对 address 节点做分类
		return self.lin(x_dict["address"])


# =========================
# 数据准备
# =========================
metadata = (data.node_types, data.edge_types)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HAN(
	metadata=metadata,
	hidden_channels=params["hidden_channels"],
	out_channels=params["out_channels"],
	num_heads=params["num_heads"],
	dropout=params["dropout"],
	num_classes=num_classes
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
criterion = torch.nn.CrossEntropyLoss()

# 训练/验证划分
labels = data['address'].y
mask = labels >= 0
train_mask = mask.clone()
val_mask = mask.clone()
# 简单划分：前80%训练，后20%验证
num_train = int(train_mask.sum().item() * 0.8)
train_idx = torch.where(mask)[0][:num_train]
val_idx = torch.where(mask)[0][num_train:]
train_mask[:] = False
val_mask[:] = False
train_mask[train_idx] = True
val_mask[val_idx] = True

# =========================
# 训练循环 + 早停
# =========================
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(params["epochs"]):
	model.train()
	optimizer.zero_grad()
	out = model(data.x_dict, data.edge_index_dict)
	loss = criterion(out[train_mask], labels[train_mask])
	loss.backward()
	optimizer.step()
	
	# # 记录训练日志
	# logger.log("HAN", "train", epoch, loss=loss.item())

	# 验证
	model.eval()
	with torch.no_grad():
		val_out = model(data.x_dict, data.edge_index_dict)
		val_loss = criterion(val_out[val_mask], labels[val_mask])
		val_pred = val_out[val_mask].argmax(dim=1)
		val_acc = (val_pred == labels[val_mask]).sum().item() / val_mask.sum().item()

	# # 记录验证日志
	# logger.log("HAN", "val", epoch, loss=val_loss.item(), acc=val_acc)

	print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

	# 早停逻辑
	if val_loss < best_val_loss:
		best_val_loss = val_loss
		patience_counter = 0
		torch.save(model.state_dict(), "best_han_model.pt")
	else:
		patience_counter += 1
		if patience_counter >= params["patience"]:
			print("早停触发，停止训练")
			break

# =========================
# 加载最佳模型并测试
# =========================
model.load_state_dict(torch.load("best_han_model.pt"))
model.eval()
with torch.no_grad():
	out = model(data.x_dict, data.edge_index_dict)
	pred = out.argmax(dim=1)
	acc = (pred[val_mask] == labels[val_mask]).sum().item() / val_mask.sum().item()
	print(f"Final Val Accuracy: {acc:.4f}")
