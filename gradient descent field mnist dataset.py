import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split



def guassian(length, total_indexes, center_index=1):
    x = np.linspace(0, length - 1, length)
    mean = x[center_index * length // total_indexes]
    std_dev = length / 20
    gaussian_array = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    gaussian_array /= np.max(gaussian_array) # normalisation condition
    return gaussian_array


# Generate scattering matrix (complex weights)
def random_unitary_tensor(n, d):
    #n = no spatial inputs
    #d = no of time
    tensor_slices = []
    p = 0
    add_dispersion = guassian(n,d)
    while p < d:
        random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        scatter_matrix, _ = np.linalg.qr(random_matrix)
        for j in range(n):
            p += 1
            if p > d: break
            tensor_slices.append(scatter_matrix[j] * add_dispersion[p])
    tensor = np.array(tensor_slices).reshape((n, d))
    plt.imshow(tensor.real)
    plt.show()
    return tensor

# ----- Dataset Scaling -----
# Hyperparameters
scale_factor = 1  # Replicate each image for more degrees of freedom
batch_size = 32

n_pixels = 28 * 28
n_pixels_new = n_pixels * scale_factor  # after replication
n_time = 200

# Time waveform input
# ----- Scattering Simulation Setup -----
t = np.linspace(-3, 3, n_time)
input_freq = 5
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)


freqs = np.fft.fftfreq(len(time_domain_waveform), d=1/n_time) 
freq_sigma = 30  # Width of Gaussian in frequency domain
f_0 = 25
gaussian_envelope = np.exp(-(freqs-f_0)**2 / (2 * freq_sigma**2))
# Add noise to the Gaussian envelope
noise = np.random.normal(0, 0.01, len(freqs))
noisy_gaussian_envelope = gaussian_envelope + noise
noisy_gaussian_envelope = torch.tensor(noisy_gaussian_envelope, dtype=torch.cfloat)
plt.plot(noisy_gaussian_envelope)
plt.show()

# Generate new scattering matrix
scatter_np = random_unitary_tensor(n_pixels_new,n_time)
scatter = torch.tensor(scatter_np, dtype=torch.cfloat)


# Transform: convert to tensor & flatten image to 1D vector (28x28 = 784)
transform = transforms.Compose([
    transforms.ToTensor(),         # (1, 28, 28)
    transforms.Lambda(lambda x: x.view(-1)),  # Flatten to (784,)
])

# Download and load MNIST
mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Split into training and validation
train_size = int(0.8 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size
train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

# Create test dataset
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)





# ----- Forward Simulation -----
def forward(modulated_input, v, complex_weights):
    E_in = torch.tensor(modulated_input, dtype=torch.cfloat)  # (n_pixels, n_time)
    E_scaled = E_in * v.view(-1, 1)  # element-wise amplitude modulation
    E_fft = torch.fft.fft(E_scaled, dim=1) * noisy_gaussian_envelope
    s = torch.einsum('ij,ij->j', E_fft, complex_weights)
    s_ifft = torch.fft.ifft(s)
    real = s_ifft.real
    chunks = torch.chunk(real, 10)
    #Option 1: take the sum of each bin
    #chunk_sums = [torch.sum(torch.sqrt(chunk**2 + 1e-8)) for chunk in chunks] 
    #Option 2: take the max of each bin
    chunk_sums = [torch.max(torch.sqrt(chunk**2 + 1e-20)) for chunk in chunks] 

    return torch.stack(chunk_sums)

# ----- Classifier -----
class ScatteringClassifier(nn.Module):
    def __init__(self, n_pixels, scatter_matrix):
        super().__init__()
        self.v = nn.Parameter(torch.randn(n_pixels))  # learnable spatial profile
        self.scatter_matrix = scatter_matrix
        self.linear = nn.Linear(10, 10)  # from 10 chunks to 10 classes
        self.waterfall = []

    def forward(self, x_batch):  # x_batch shape: (batch_size, n_pixels)
        batch_outputs = []
        for x in x_batch:
            modulated_input = np.outer(x, time_domain_waveform)
            chunk_output = forward(modulated_input, self.v, self.scatter_matrix)
            batch_outputs.append(chunk_output)
        batch_tensor = torch.stack(batch_outputs)  # (batch_size, 10)
        return self.linear(batch_tensor)
    
    # def add_water(self):
    #     x = X_train_scaled[0]
    #     modulated_input = np.outer(x, time_domain_waveform)
    #     output = forward(modulated_input, self.v, self.scatter_matrix)
    #     self.waterfall.append(output.detach())

    # def plot_waterfall(self):
    #     image = np.array(self.waterfall)
    #     image = image.reshape((image.size//10, 10))
    #     plt.imshow(image)
    #     plt.show()
    #     image = np.array(X_train[0])
    #     image = image.reshape((8,8))
    #     plt.imshow(image)
    #     plt.show()


# ----- Training Setup -----
model = ScatteringClassifier(n_pixels_new, scatter)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()


# ----- Training Loop with Scaled Data -----
n_epochs = 10
batch_size = 32



for epoch in range(n_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        # Scale input by replicating vector
        batch_x_scaled = batch_x.repeat(1, scale_factor)  # shape: (batch, n_pixels_new)

        optimizer.zero_grad()
        preds = model(batch_x_scaled)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            val_x_scaled = val_x.repeat(1, scale_factor)
            preds = model(val_x_scaled)
            correct += (preds.argmax(1) == val_y).sum().item()
            total += val_y.size(0)

    print(f"Epoch {epoch+1}: Val Accuracy = {correct / total:.4f}")

#    model.add_water()
#model.plot_waterfall()


# ----- Prepare Test Data (scaled like train/val) -----
X_test, y_test = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
X_test_scaled = X_test.repeat(1, scale_factor)  # shape: (num_test_samples, n_pixels_new)

X_test_tensor = X_test_scaled.to(torch.float32)
y_test_tensor = y_test.to(torch.long)

# ----- Generate Predictions -----
model.eval()
all_preds = []
with torch.no_grad():
    for i in range(0, X_test_tensor.size(0), batch_size):
        x_batch = X_test_tensor[i:i+batch_size]
        preds = model(x_batch)
        all_preds.append(preds.argmax(dim=1))
y_pred = torch.cat(all_preds).cpu().numpy()
y_true = y_test_tensor.cpu().numpy()

# ----- Confusion Matrix -----
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])

# ----- Plot -----
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Confusion Matrix")
plt.show()


# ----- Verify that the distribution of bins for random SLM arrangements is even -----------
ones = np.ones(n_pixels * scale_factor)
mod_input = np.outer(ones, time_domain_waveform)#
sum_outputs = np.zeros(10)

for i in range(1000):
    v = torch.randn(n_pixels * scale_factor)
    sum_outputs += forward(mod_input, v, scatter).numpy()

plt.bar(np.linspace(0,9,10), sum_outputs)
plt.show()


