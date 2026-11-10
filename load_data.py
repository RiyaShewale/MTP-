
import numpy as np
from pathlib import Path

# Dataset folder
path = Path(__file__).resolve().parent / "data" / "UCI HAR Dataset"

# Load training data
X_train = np.loadtxt(
    path / "train" / "X_train.txt",
    dtype=np.float32
)

y_train = np.loadtxt(
    path / "train" / "y_train.txt",
    dtype=np.int64
)

# Load testing data
X_test = np.loadtxt(
    path / "test" / "X_test.txt",
    dtype=np.float32
)

y_test = np.loadtxt(
    path / "test" / "y_test.txt",
    dtype=np.int64
)

# Convert labels from 1-6 to 0-5
y_train = y_train - 1
y_test = y_test - 1

# Print dataset dimensions
print("Training data:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing data:", X_test.shape)
print("Testing labels:", y_test.shape)

# Check the first sample
print("\nFirst 10 features:", X_train[0][:10])
print("Activity label:", y_train[0])

print("\nDataset loaded successfully!")
