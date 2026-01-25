import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_cosine_schedule_with_warmup

# ====== 1. 准备一个简单的数据集 ======
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ====== 2. 一个简单的模型 ======
model = nn.Sequential(
	nn.Linear(10, 32),
	nn.ReLU(),
	nn.Linear(32, 2)
)

# ====== 3. AdamW + weight decay ======
optimizer = torch.optim.AdamW(
	model.parameters(),
	lr=3e-4,
	weight_decay=1e-2
)

# ====== 4. warmup + cosine scheduler ======
num_epochs = 10
total_steps = len(loader) * num_epochs
warmup_steps = int(total_steps * 0.1)   # 前 10% steps warmup

scheduler = get_cosine_schedule_with_warmup(
	optimizer,
	num_warmup_steps=warmup_steps,
	num_training_steps=total_steps
)

# ====== 5. 训练循环 ======
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
	for batch_x, batch_y in loader:
		logits = model(batch_x)
		loss = criterion(logits, batch_y)

		loss.backward()
		optimizer.step()
		scheduler.step()     # 每个 step 更新学习率
		optimizer.zero_grad()

	print(f"Epoch {epoch+1}, lr={scheduler.get_last_lr()[0]:.6f}, loss={loss.item():.4f}")
