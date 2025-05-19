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
n_time = 200
input_freq = 5
t = np.linspace(-3, 3, n_time)
time_domain_waveform = np.exp(1j * input_freq * t) * np.exp(-500 * input_freq**2 * t**2)

# Generate scattering matrix (complex weights)
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

# ----- Dataset Scaling -----
scale_factor = 2  # Let's replicate each image n times (adjust this to your needs)
X_train_scaled = np.tile(X_train, (1, scale_factor))  # Replicate each image n times
X_val_scaled = np.tile(X_val, (1, scale_factor))  # Same for validation data

n_pixels_new = n_pixels * scale_factor  # New input size after scaling

# ----- Adjust the Scattering Matrix -----
scatter_np = random_unitary_tensor(n_pixels_new, n_time)  # New scatter matrix for the scaled input
scatter = torch.tensor(scatter_np, dtype=torch.cfloat)


# ----- Forward Simulation -----
def forward(modulated_input, v, complex_weights):
    E_in = torch.tensor(modulated_input, dtype=torch.cfloat)  # (n_pixels, n_time)
    E_scaled = E_in * v.view(-1, 1)  # element-wise amplitude modulation
    E_fft = torch.fft.fft(E_scaled, dim=1)
    s = torch.einsum('ij,ij->j', E_fft.T, complex_weights)
    s_ifft = torch.fft.ifft(s)
    real = s_ifft.real
    chunks = torch.chunk(real, 10)
    chunk_sums = [torch.sum(torch.sqrt(chunk**2 + 1e-20)) for chunk in chunks]
    return torch.stack(chunk_sums)

# ----- Classifier -----
class ScatteringClassifier(nn.Module):
    def __init__(self, n_pixels, scatter_matrix):
        super().__init__()
        self.v = nn.Parameter(torch.rand(n_pixels))  # learnable spatial profile
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


lr = 0.1 #lr = 0.1 for most results in report
# ----- Training Setup -----
model = ScatteringClassifier(n_pixels=n_pixels * scale_factor, scatter_matrix=scatter)
optimizer = optim.Adam(model.parameters(), lr=lr) 
loss_fn = nn.CrossEntropyLoss()

# ----- Convert Scaled Data to PyTorch tensors -----
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# ----- Training Loop with Scaled Data -----
n_epochs = 40
batch_size = 32

train_accuracies = []

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
        train_accuracies.append(val_accuracy)
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Accuracy: {val_accuracy:.4f}")


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
plt.title(f"Confusion Matrix ({accuracy:.2%} accurate)")
plt.savefig(str("Scale factor = " + str(scale_factor) + " ntime = " + str(n_time) +  " Confusion matrix"))
plt.clf()


plt.plot(range(1, n_epochs + 1), train_accuracies, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs Epoch')
plt.grid(True)
plt.savefig(str("Scale factor = " + str(scale_factor) + " ntime = " + str(n_time) + " Training vs Epoch"))
plt.clf()



# ----- Verify that the distribution of bins for random SLM arrangements is even -----------
ones = np.ones(n_pixels * scale_factor)
mod_input = np.outer(ones, time_domain_waveform)#
sum_outputs = np.zeros(10)

for i in range(1000):
    v = torch.rand(n_pixels * scale_factor)
    sum_outputs += forward(mod_input, v, scatter).numpy()

plt.bar(np.linspace(0,9,10), sum_outputs)
plt.xlabel("Bin")
plt.ylabel("Sum of bin")
plt.show()
