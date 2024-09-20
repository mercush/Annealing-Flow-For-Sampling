import torch
from scipy.io import mmread
import numpy as np

# Load training data
X_train_sparse = mmread('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/X_train_20k.mtx')  # Load sparse matrix
X_train_dense = X_train_sparse.toarray()  # Convert to dense
y_train = np.loadtxt('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/y_train_20k.txt')  # Load labels

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Load testing data
X_test_sparse = mmread('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/X_test_20k.mtx')  # Load sparse matrix
X_test_dense = X_test_sparse.toarray()  # Convert to dense
y_test = np.loadtxt('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/y_test_20k.txt')  # Load labels

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print(X_train_tensor)
print(X_train_tensor.shape)
print(y_train_tensor.shape)
print(X_test_tensor.shape)
print(y_test_tensor.shape)