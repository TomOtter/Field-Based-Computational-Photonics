import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



# ----- Data: Load digits dataset (8x8 grayscale images) -----
digits = load_digits()
X = digits.images  # shape (n_samples, 8, 8)
y = digits.target  # shape (n_samples,)
n_samples, h, w = X.shape
n_pixels = h * w  # = 64

# Flatten images to vectors
X = X.reshape((n_samples, n_pixels)) / 16.0  # normalize to [0, 1]

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Scattering Simulation Setup -----
n_time = 150
input_freq = 5
t = np.linspace(-3, 3, n_time)
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)

def guassian(length, total_indexes):
    center_index=total_indexes//2
    x = np.linspace(0, length - 1, length)
    mean = x[center_index * length // total_indexes]
    std_dev = length / 5
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
            tensor_slices.append(scatter_matrix[j] * add_dispersion)
    tensor = np.array(tensor_slices).reshape((d, n))
    return tensor

# ----- Dataset Scaling -----
scale_factor = 10  # Let's replicate each image 3 times (adjust this to your needs)
X_train_scaled = np.tile(X_train, (1, scale_factor))  # Replicate each image n times
X_val_scaled = np.tile(X_val, (1, scale_factor))  # Same for validation data

n_pixels_new = n_pixels * scale_factor  # New input size after scaling

# ----- Adjust the Scattering Matrix -----
scatter_np = random_unitary_tensor(n_pixels_new, n_time)  # New scatter matrix for the scaled input
scatter = torch.tensor(scatter_np, dtype=torch.cfloat)


# ----- Forward Simulation -----
def forward(modulated_input, v, phases, complex_weights, correction):
    E_in = torch.tensor(modulated_input, dtype=torch.cfloat)  # (n_pixels, n_time)
    E_scaled = E_in * v.view(-1, 1)  # element-wise amplitude modulation
    E_scaled = E_scaled * (torch.exp(1j * phases)).view(-1,1) #element wise phase modulation
    #E_scaled = E_in * (torch.exp(1j * phases)).view(-1,1) #For no amplitude modulation, just phase
    E_fft = torch.fft.fft(E_scaled, dim=1)
    s = torch.einsum('ik,ki->i', complex_weights, E_fft)
    s_ifft = torch.fft.ifft(s)
    real = s_ifft.real
    real = real / correction
    chunks = torch.chunk(real, 10)
    chunk_sums = [torch.sum(torch.sqrt(chunk**2 + 1e-20)) for chunk in chunks]
    return torch.stack(chunk_sums)

# ----- Classifier -----
class ScatteringClassifier(nn.Module):
    def __init__(self, n_pixels,n_time, scatter_matrix):
        super().__init__()
        self.v = nn.Parameter(torch.rand(n_pixels))  # learnable spatial profile
        self.phase = nn.Parameter(torch.rand(n_pixels) * (2 * torch.pi) )
        self.scatter_matrix = scatter_matrix
        self.linear = nn.Linear(10, 10)  # from 10 chunks to 10 classes
        self.waterfall = []



        # ----- Verify that the distribution output is evenly distributed -----------
        ones = np.ones(n_pixels)
        mod_input = np.outer(ones, time_domain_waveform)#
        sum_outputs = np.zeros(n_time)
        sum_bins = np.zeros(10)
        phases = torch.zeros(n_pixels)

        for i in range(10000):
            v = torch.rand(n_pixels)

            E_in = torch.tensor(mod_input, dtype=torch.cfloat)  # (n_pixels, n_time)
            E_scaled = E_in * v.view(-1, 1)  # element-wise amplitude modulation
            E_scaled = E_scaled * (torch.exp(1j * phases)).view(-1,1) #element wise phase modulation
            #E_scaled = E_in * (torch.exp(1j * phases)).view(-1,1) #For no amplitude modulation, just phase
            E_fft = torch.fft.fft(E_scaled, dim=1)
            s = torch.einsum('ik,ki->i', self.scatter_matrix, E_fft)
            s_ifft = torch.fft.ifft(s)
            real = s_ifft.real
            chunks = torch.chunk(real, 10)
            chunk_sums = [torch.sum(torch.sqrt(chunk**2 + 1e-20)) for chunk in chunks]
            sum_outputs += abs(real.numpy())
            sum_bins += chunk_sums

        # plt.bar(np.linspace(0,n_time,n_time), sum_outputs)
        # plt.xlabel("Bin")
        # plt.ylabel("Sum of bin")
        # plt.show()

        # plt.bar(np.linspace(0,9,10), sum_bins)
        # plt.show()

        #define correction to be used to account for uneven function
        self.correction =  torch.from_numpy(sum_outputs / max(sum_outputs)).to(torch.cfloat).real

        # plt.bar(np.linspace(0,n_time,n_time), sum_outputs / self.correction)
        # plt.show()

    def forward(self, x_batch):  # x_batch shape: (batch_size, n_pixels)
        batch_outputs = []
        for x in x_batch:
            modulated_input = np.outer(x, time_domain_waveform)
            chunk_output = forward(modulated_input, self.v, self.phase, self.scatter_matrix, self.correction)
            batch_outputs.append(chunk_output)
        batch_tensor = torch.stack(batch_outputs)  # (batch_size, 10)
        return self.linear(batch_tensor)
    
    def add_water(self):
        x = X_train_scaled[0]
        modulated_input = np.outer(x, time_domain_waveform)
        output = forward(modulated_input, self.v, self.phase, self.scatter_matrix, self.correction)
        self.waterfall.append(output.detach())

    def plot_waterfall(self):
        image = np.array(self.waterfall)
        image = image.reshape((image.size//10, 10))
        plt.imshow(image)
        plt.show()
        image = np.array(X_train[0])
        image = image.reshape((8,8))
        plt.imshow(image)
        plt.show()


# ----- Training Setup -----
model = ScatteringClassifier(n_pixels=n_pixels * scale_factor,n_time = n_time, scatter_matrix=scatter)
optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay= 0.000001) #000001
loss_fn = nn.CrossEntropyLoss()

# ----- Convert Scaled Data to PyTorch tensors -----
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# ----- Training Loop with Scaled Data -----
n_epochs = 40 #40
batch_size = 32

for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        x_batch = X_train_tensor[indices]
        y_batch = y_train_tensor[indices]

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.v.clamp_(0.0)

        

    # Validation accuracy
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_accuracy = (val_preds.argmax(dim=1) == y_val_tensor).float().mean().item()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}")

    model.add_water()
model.plot_waterfall()

print(model.v)

# ----- Prepare Test Data (scaled like train/val) -----
X_test_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_val, dtype=torch.long)

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

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")  # e.g. 94.50%


# ----- Confusion Matrix -----
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])

# ----- Plot -----
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.2%}')
plt.tight_layout()
plt.show()


