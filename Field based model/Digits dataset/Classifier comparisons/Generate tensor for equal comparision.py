import torch
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def random_unitary_tensor(n, d):
    #n = no spatial inputs
    #d = no of time
    tensor_slices = []
    p = 0
    while p < d:
        random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        scatter_matrix, _ = np.linalg.qr(random_matrix)
        for j in range(n):
            p += 1
            if p > d: break
            tensor_slices.append(scatter_matrix[j])
    tensor = np.array(tensor_slices).reshape((d, n))
    return tensor


digits = load_digits()
X = digits.images  # shape (n_samples, 8, 8)
y = digits.target  # shape (n_samples,)
n_samples, h, w = X.shape
n_pixels = h * w  # = 64
scale_factor = 5
n_pixels_new = n_pixels * scale_factor
n_time = 200

scatter_np = random_unitary_tensor(n_pixels_new, n_time)  # New scatter matrix for the scaled input
scatter = torch.tensor(scatter_np, dtype=torch.cfloat)

with open('saved_scatterer.pkl', 'wb') as file:
    pkl.dump(scatter, file)



