from sklearn.model_selection import StratifiedKFold
import numpy as np

X = np.arange(10).reshape((10, 1))
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(F"X={X}")
print(F"y={y}")
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
	print(f"Fold {fold+1}")
	print("Train indices:", train_idx)
	print("Test indices:", test_idx)
