import torch
import torch.fft
from matplotlib import pyplot as plt
import numpy as np

def random_unitary_tensor(n, d):
    tensor_slices = []
    p = 0
    while p < d:
        random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        scatter_matrix, _ = np.linalg.qr(random_matrix)
        for j in range(n):
            p += 1
            if p > d: break
            spatial_slice = scatter_matrix[j]
            tensor_slices.append(spatial_slice)
    return np.array(tensor_slices).reshape((n, d))

def forward(E_in, v, complex_weights):
    E_in = torch.tensor(E_in, dtype=torch.cfloat)

    # Step 2: row-wise scaling
    E_scaled = E_in * v.view(-1, 1)

    

    # Step 3: FFT row-wise
    E_fft = torch.fft.fft(E_scaled, dim=1)

    # Step 4: column-wise dot product
    s = torch.einsum('ij,ij->j', E_fft.T, complex_weights)

    # Step 5: IFFT
    s_ifft = torch.fft.ifft(s)

    # Step 6: real part, chunk, abs, sum
    real = s_ifft.real
    n_chunks = 10
    chunks = torch.chunk(real, n_chunks)
    chunk_sums = [torch.sum(torch.sqrt(chunk**2 + 1e-20)) for chunk in chunks]
    
    return torch.stack(chunk_sums)

# --- Inputs ---
n_time = 100
n_space = 200
input_freq = 5
t = np.linspace(-3, 3, n_time)
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)
image_slm = np.ones(n_space)
modulated_input = np.outer(image_slm, time_domain_waveform)

# Learnable vector
v = torch.tensor(np.random.uniform(0, 1, n_space), requires_grad=True, dtype=torch.float32)

# Random unitary complex weights
scatter_np = random_unitary_tensor(n_time, n_space)
scatter = torch.tensor(scatter_np, dtype=torch.cfloat)

# Target output (simple test target)
target = torch.zeros(10)
target[6] = 1.0

# Optimizer + loss
optimizer = torch.optim.Adam([v], lr=0.1)
loss_fn = torch.nn.MSELoss()

# --- Training loop ---
for step in range(100):
    optimizer.zero_grad()
    output = forward(modulated_input, v, scatter)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()



# Final output visualization
plt.plot(output.detach().numpy())
plt.title("Final Chunk Sums After Training")
plt.show()


