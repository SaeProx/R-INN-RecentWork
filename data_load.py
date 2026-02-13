import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class MicrowaveDataset:
    def __init__(self, x_path='dataset_500DOE_aid/dataset_x.npy', y_path='dataset_500DOE_aid/dataset_y.npy', test_split=0.2):
        # 1. Load Data
        print(f"Loading data from {x_path} and {y_path}...")
        raw_x = np.load(x_path)
        raw_y = np.load(y_path)

        # 2. Transpose if necessary
        # PyTorch expects (N_samples, N_features)
        if raw_x.shape[0] < raw_x.shape[1]:
            print(f"Transposing X from {raw_x.shape} to {raw_x.T.shape}")
            raw_x = raw_x.T

        if raw_y.shape[0] < raw_y.shape[1]:
            print(f"Transposing Y from {raw_y.shape} to {raw_y.T.shape}")
            raw_y = raw_y.T

        self.n_samples = raw_x.shape[0]
        self.x_dim = raw_x.shape[1]  # Should be 5
        self.y_dim = raw_y.shape[1]  # Should be 202

        print(f"Dataset Stats: {self.n_samples} samples.")
        print(f"X Dim: {self.x_dim}, Y Dim: {self.y_dim}")

        # 3. Normalization (Standard Scaling: Mean 0, Std 1)
        # This is CRITICAL for Invertible Neural Networks
        self.x_mean = raw_x.mean(axis=0)
        self.x_std = raw_x.std(axis=0) + 1e-6  # Avoid div by zero

        self.y_mean = raw_y.mean(axis=0)
        self.y_std = raw_y.std(axis=0) + 1e-6

        x_norm = (raw_x - self.x_mean) / self.x_std
        y_norm = (raw_y - self.y_mean) / self.y_std

        # 4. Convert to Tensor
        self.x_data = torch.FloatTensor(x_norm)
        self.y_data = torch.FloatTensor(y_norm)

        # 5. Split Train/Test
        n_test = int(self.n_samples * test_split)
        n_train = self.n_samples - n_test

        self.train_x = self.x_data[:n_train]
        self.train_y = self.y_data[:n_train]
        self.test_x = self.x_data[n_train:]
        self.test_y = self.y_data[n_train:]

        print(f"Split: {n_train} Train, {n_test} Test")

    def get_train_loader(self, batch_size=32):
        ds = TensorDataset(self.train_x, self.train_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_test_data(self):
        return self.test_x, self.test_y

    def denormalize_x(self, x_tensor):
        """Convert network output back to real physical parameters"""
        x_np = x_tensor.detach().cpu().numpy()
        return x_np * self.x_std + self.x_mean

    def denormalize_y(self, y_tensor):
        """Convert network output back to real S-parameters"""
        y_np = y_tensor.detach().cpu().numpy()
        return y_np * self.y_std + self.y_mean


# Helper to load names
def load_names(path='dataset_300/x_names.txt'):
    with open(path, 'r') as f:
        line = f.readline()
        # Remove whitespace and split by comma
        names = [n.strip() for n in line.split(',')]
    return names