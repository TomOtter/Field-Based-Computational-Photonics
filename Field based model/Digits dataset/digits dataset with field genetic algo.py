import numpy as np
import scipy.fftpack as fft
import scipy
from multiprocessing.dummy import Pool, Array
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import time
from scipy.signal import butter, filtfilt, hilbert
import multiprocessing
import cProfile

plt.rcParams['image.cmap'] = 'jet'


n_time = 100
target_position = 0
pi = np.pi
no_freq = 100

def guassian(length, center_index, total_indexes):
    x = np.linspace(0, length - 1, length)
    mean = x[center_index * length // total_indexes]
    std_dev = length / 4
    gaussian_array = np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    gaussian_array /= np.max(gaussian_array) # normalisation condition
    return gaussian_array

def random_unitary_tensor(n,d):

    tensor_slices = []
    add_dispersion = guassian(n,target_position,d)

    checkpoint = d / 10  # Calculate 10% intervals
    next_checkpoint = checkpoint  # Track next checkpoint
    p = 0
    while p<d:
        if p >= next_checkpoint:
            print(f"Progress: {round((p / d) * 100)}% completed")
            next_checkpoint += checkpoint 
        random_matrix = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        scatter_matrix, _ = np.linalg.qr(random_matrix) 
        for j in range(n):
            p+=1
            if p > d: break
            
            spatial_slice = scatter_matrix[j] * add_dispersion[p]
            tensor_slices.append(spatial_slice)
    print("Done - Initialisation")
    graphic = np.real(np.array(tensor_slices).reshape((n,d)))
    plt.imshow(graphic)
    plt.show()
    return tensor_slices

def fast_tensor(n,d):
    tensor_slices = []
    checkpoint = d / 10  # Calculate 10% intervals
    next_checkpoint = checkpoint  # Track next checkpoint
    for i in range(d):
        if i >= next_checkpoint:
            print(f"Progress: {round((i / d) * 100)}% completed")
            next_checkpoint += checkpoint 
        spatial_slice = np.random.randn(n) + 1j * np.random.randn(n)
        tensor_slices.append(spatial_slice)
    return tensor_slices

# Softmax function for classification
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilize for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return x / np.sum(x)

# Cross-entropy loss
def cross_entropy_loss(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

# One-hot encode labels
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def bad_forward_pass(input):
    output_after_scatterer = part_forward(input)

    # Additional step for waveform - convert output to bins
    bins = np.array_split(output_after_scatterer, targets) # split into ten bins
    summmed_bins = []
    for bin in bins:
        summmed_bins.append(np.sum(bin))
    transformed_output = np.array(summmed_bins)

    return abs(transformed_output)

def xforward_pass(input):
    output_after_scatterer = part_forward(input)
    output_after_scatterer = abs(output_after_scatterer.real)
    expectation = 0
    total = np.sum(output_after_scatterer)
    for i in range(len(output_after_scatterer)):
        expectation += i * output_after_scatterer[i] / total
    expectation = expectation /100
       
    x = np.linspace(0,2, targets)
    output = np.exp(-(x - expectation)**2/2**2)

    return output

def good_forward_pass(input):
    output_after_scatterer = part_forward(input)

    # Additional step for waveform - convert output to bins
    bins = np.array_split(np.real(output_after_scatterer), targets) # split into bins
    max_bins = []
    for bin in bins:
        max_bins.append(max(bin))
    transformed_output = np.array(max_bins)
    return transformed_output
    


def filter_forward_pass(input):
    fs = len(input)
    rectified = np.abs(input)
    nyq = 0.5 * fs
    cutoff = 50
    normal_cutoff = cutoff / nyq
    order = 5
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, rectified)
    bins = np.array_split(filtered_data, targets) # split into bins
    sum_bins = []
    for bin in bins:
        sum_bins.append(np.sum(bin))
    transformed_output = np.array(sum_bins)
    return transformed_output

def forward_pass(input):
    output_after_scatterer = part_forward(input)
    rectified = abs(np.real(output_after_scatterer))
    bins = np.array_split(rectified, targets) # split into bins
    sum_bins = []
    for bin in bins: 
        sum_bins.append(np.sum(bin))
    transformed_output = np.array(sum_bins)
    return transformed_output


def part_forward(input):

    scatter_matrix = np.frombuffer(scatter_matrix_array).reshape(n_time,input_size)


    modulated_input = np.outer(input, time_domain_waveform)
    freq_input = fft.fft(modulated_input) * noisy_gaussian_envelope
    scatter = []
    start_index = 0
    for i in range(n_time):
        scatter.append(scatter_matrix[i] @ freq_input[:,i])
    scattered_output = np.stack(scatter, axis=-1)
    output_after_scatterer = fft.ifft(scattered_output)
    return np.multiply(output_after_scatterer, correction_map)


def loss_of_specimen(specimen):
    slm_array = specimen
    total_loss = 0
    total_error = 0
    correct_predictions = 0
    true_positives = np.zeros(targets)
    false_negatives = np.zeros(targets)
    # Loop through the training samples
    for i in range(num_samples):
        # Step 1: Encode input and modulate with SLM
        slm_modulation = X_train[i] * slm_array
        # Step 2: Apply transmission matrix
        transformed_output = forward_pass(slm_modulation)

        # Step 3: Classify using softmax
        predictions = softmax(transformed_output.reshape(1, -1))
        
        # Step 4: Compute loss
        loss = cross_entropy_loss(predictions, y_train_onehot[i].reshape(1, -1))
        total_loss += loss
        
        # Step 5: Compute error
        error = np.sum(predictions - y_train_onehot[i].reshape(1, -1))
        total_error += error

        # Track accuracy
        if np.argmax(predictions) == y_train[i]:
            correct_predictions += 1
            true_positives[y_train[i]] += 1
        else:
            false_negatives[y_train[i]] += 1

    return correct_predictions, total_loss

def evaluate_batch(batch):
    return [loss_of_specimen(ind) for ind in batch] 


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]



# Main training loop
def train_system(X_train, y_train, num_epochs, learning_rate, popsize):
    waterfall = []

    population = []
    for _ in range(popsize):
        specimen = np.random.uniform(0, 1, input_size)
        population.append(specimen)

    
        
    fitness = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        batch_size = 10
        batches = list(chunked(population, batch_size))

        with Pool() as pool:
            results = pool.map(evaluate_batch, batches)

        flattened_results = [res for sublist in results for res in sublist]    
        fitness, total_losses = zip(*flattened_results)
        
        
        #make new pop
        top_indices = np.argsort(fitness)[-4:]
        top_individuals = []
        for index in top_indices:
            top_individuals.append(population[index])
    
        best_indices = np.argsort(fitness)[-6:]
        best_individuals = []
        for index in best_indices:
            best_individuals.append(population[index])
        

        new_population = []
        for z in range(popsize):
            if z < 5 & z>0 :
                idx1, idx2 = np.random.choice(len(top_individuals), 2, replace=False)
                parent1, parent2 = top_individuals[idx1], top_individuals[idx2] 
            else:
                idx1, idx2 = np.random.choice(len(best_individuals), 2, replace=False)
                parent1, parent2 = best_individuals[idx1], best_individuals[idx2] 
            
            child = (parent1 + parent2) / 2  # Crossover (averaging weights)
            # Mutation (randomly perturb some weights)
            if np.random.rand() < 0.7:   #Mutation rate
                child += np.random.randn(input_size) * learning_rate # Small random noise
            child = np.clip(child, 0, 1) 
            if z == 0:
                child = best_individuals[-1]

            new_population.append(child)


        population = np.array(new_population)

        # Report epoch results
        print(f"Time elaspsed = {time.time() - start_time}")
        accuracy = max(fitness) / num_samples
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {min(total_losses):.7f}, Accuracy: {accuracy:.7f}")
        waterfall.append(forward_pass(best_individuals[-1] * X_train[0]))
    bestindex = np.argmax(fitness)
    best_slm = population[bestindex]
    waterfall = np.array(waterfall)
    plot = waterfall.reshape((num_epochs,targets))
    plt.imshow(plot)
    plt.show()
    plt.imshow(best_slm.reshape((scale_factor * 8,8)))
    plt.show()
    plt.imshow(X_train[0].reshape((scale_factor * 8, 8)))
    plt.show()

    return best_slm

# Terahertz numbers for input signal (Computer likes bigger numebers so not using)
# input_freq = 5e12
# # pulse_width = 100e-15
# t = np.linspace(-30e-12, 30e-12, n_time)
# time_domain_waveform = np.exp(1j*input_freq*t) * np.exp(-5*input_freq**2*t**2)
# freqs = np.fft.fftfreq(len(time_domain_waveform), d=1/n_time) 


input_freq = 5
# pulse_width = 5e-3
t = np.linspace(-3, 3, n_time)
time_domain_waveform = np.exp(1j*input_freq*t) * np.exp(-500*input_freq**2*t**2)
freqs = np.fft.fftfreq(len(time_domain_waveform), d=1/n_time) 


freq_sigma = 30  # Width of Gaussian in frequency domain
f_0 = 25
gaussian_envelope = np.exp(-(freqs-f_0)**2 / (2 * freq_sigma**2))
# Add noise to the Gaussian envelope
noise = np.random.normal(0, 0.01, len(freqs))
noisy_gaussian_envelope = gaussian_envelope + noise



scale_factor = 5
input_size = 64 * scale_factor




digits = datasets.load_digits()
y = digits.target
X = digits.images

selected_digits = [0,1,2,3,4]
mask = np.isin(y, selected_digits)

X_selected, y_selected = X[mask], y[mask]
n_samples = len(X_selected)
data = X_selected.reshape((n_samples, -1))
targets = len(selected_digits)

# input_size = 13 * scale_factor
# wine = datasets.load_wine()
# targets = 3
# n_samples = wine.data.shape[1]
# data = wine.data.reshape((wine.data.shape[0], -1))
# mapping = {old: new for old, new in enumerate(wine.target)}


# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(data, y_selected, test_size=0.8, shuffle=True, stratify=y_selected)
# X_train, X_test, y_train, y_test = train_test_split(data, wine.target, test_size=0.2, shuffle=True, stratify=wine.target)

X_expanded = np.tile(X_train, (1, scale_factor)) 
X_test_expanded =  np.tile(X_test, (1, scale_factor)) 

# Initial guess for the optimization of the first half of the SLM modulation

scatter_matrix_data = np.array(random_unitary_tensor(input_size,no_freq))

scatter_matrix_flat = scatter_matrix_data.flatten()
scatter_matrix_array = Array('d', scatter_matrix_flat) 



correction_map = np.ones(n_time)
sum_transformed_output = np.zeros(n_time)
for _ in range(1000):
    specimen = np.random.uniform(0, 1, input_size)
    sum_transformed_output += abs(part_forward(specimen))
plt.bar(np.linspace(0,n_time,n_time),sum_transformed_output)
plt.show()
normalised_sum_transformed_output = sum_transformed_output / max(sum_transformed_output)
correction_map = 1 / normalised_sum_transformed_output

learning_rate = 10e-2 # 2e-2
num_epochs = 30
popsize = 30

test_input = np.random.uniform(0, 1, input_size)
output = part_forward(test_input)
print("Integral of output = ",np.sum(output),"Integral of absoulte output", np.sum(abs(output)))
# plt.plot(output)
# plt.show()

X_train = X_expanded
num_samples, input_size = X_train.shape
num_classes = targets
# One-hot encode the labels
y_train_onehot = one_hot_encode(y_train, num_classes)


trained_slm = train_system(X_expanded, y_train, num_epochs, learning_rate, popsize=popsize)




def test_system(X_test, y_test, slm_array):
    predictions = []
    for i in range(X_test.shape[0]):
        modulated_input = X_test_expanded[i] * slm_array
        transformed_output = forward_pass(modulated_input)

        predictions.append(np.argmax(softmax(transformed_output.reshape(1, -1))))
    return predictions


#scatter based off less matricies in tensor
#tomorrow job
#start_index = 0
# for i in range(no_freq):
#     for j in range(start_index + n_time//no_freq):
#         scatter.append(scatter_matrix[i] @ freq_input[:,j])
#     start_index += n_time//no_freq



# Predict and evaluate
test_y = y_test[:300]

y_pred = test_system(X_test_expanded[:300], test_y, trained_slm)

print("Unique labels in y_true:", np.unique(test_y))
print("Unique labels in y_pred:", np.unique(y_pred))



accuracy = accuracy_score(test_y, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(test_y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(targets), yticklabels=range(targets))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
