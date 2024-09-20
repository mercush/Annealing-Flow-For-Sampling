from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
file_path = '/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/covtype.libsvm.binary.scale'
X, y = load_svmlight_file(file_path)

# Randomly select 20,000 samples
n_samples = 20000
total_samples = X.shape[0]
random_indices = np.random.choice(total_samples, n_samples, replace=False)
X_subset = X[random_indices]
y_subset = y[random_indices]
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

from scipy.io import mmwrite
mmwrite('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/X_train_20k.mtx', X_train)
np.savetxt('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/y_train_20k.txt', y_train)

mmwrite('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/X_test_20k.mtx', X_test)
np.savetxt('/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow/Bayesian_Logistic/dataset/y_test_20k.txt', y_test)
