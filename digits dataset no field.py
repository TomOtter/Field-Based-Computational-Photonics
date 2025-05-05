import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns




# Generate a random transmission matrix
def generate_transmission_matrix(input_size, output_size):
    return np.random.random((output_size, input_size))

# Softmax function for classification
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilize for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(predictions, labels):
    return -np.mean(np.sum(labels * np.log(predictions + 1e-9), axis=1))

# One-hot encode labels
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

# Feedback process to update the SLM array
def update_slm(slm, gradient, learning_rate):
    return slm - learning_rate * gradient

# Main training loop
def train_system(X_train, y_train, transmission_matrix, slm_array, num_epochs, learning_rate):
    num_samples, input_size = X_train.shape
    output_size = transmission_matrix.shape[0]
    num_classes = np.max(y_train) + 1

    # One-hot encode the labels
    y_train_onehot = one_hot_encode(y_train, num_classes)

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        if epoch > num_epochs * 9/10:
            learning_rate *= 0.99

        # Loop through the training samples
        for i in range(num_samples):
            # Step 1: Encode input and modulate with SLM
            modulated_input = X_train[i] * slm_array
            
            # Step 2: Apply transmission matrix
            transformed_output = transmission_matrix @ modulated_input
            transformed_output = abs(transformed_output**2)
            
            # Step 3: Classify using softmax
            predictions = softmax(transformed_output.reshape(1, -1))
            
            # Step 4: Compute loss
            loss = cross_entropy_loss(predictions, y_train_onehot[i].reshape(1, -1))
            total_loss += loss
            
            # Step 5: Compute gradient for SLM
            error = predictions - y_train_onehot[i].reshape(1, -1)
            gradient = (transmission_matrix.T @ error.flatten()) * X_train[i]
            # gradient /= np.linalg.norm(gradient) + 1e-9

            
            # Step 6: Update SLM array and constrain values
            slm_array = update_slm(slm_array, gradient, learning_rate)
            slm_array = np.clip(slm_array, 0, 1)  # Constrain SLM to [0, 1]
            
            # Track accuracy
            if np.argmax(predictions) == y_train[i]:
                correct_predictions += 1
        
        # Report epoch results
        accuracy = correct_predictions / num_samples
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    plt.plot(transformed_output)
    plt.show()
    plt.imshow(digits.images[i],cmap = plt.cm.gray_r , interpolation = 'nearest')
    plt.show()
    return slm_array

scale_factor = 16
input_size = 64 * scale_factor
output_size = 10

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=True)

X_expanded = np.tile(X_train, (1, scale_factor)) 
X_test_expanded =  np.tile(X_test, (1, scale_factor)) 

# Initialize system
transmission_matrix = generate_transmission_matrix(input_size, output_size)
slm_array = np.ones(input_size)  # Start with an all-ones SLM array
learning_rate = 0.00002
num_epochs = 1000

# Train the system
trained_slm = train_system(X_expanded, y_train, transmission_matrix, slm_array, num_epochs, learning_rate)

# Test the system
def test_system(X_test, y_test, transmission_matrix, slm_array):
    predictions = []
    for i in range(X_test.shape[0]):
        modulated_input = X_test_expanded[i] * slm_array
        transformed_output = transmission_matrix @ modulated_input
        predictions.append(np.argmax(softmax(transformed_output.reshape(1, -1))))
    return predictions

# Predict and evaluate
test_y = y_test[:500]

y_pred = test_system(X_test_expanded[:500], test_y, transmission_matrix, trained_slm)



accuracy = accuracy_score(test_y, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot confusion matrix
conf_matrix = confusion_matrix(test_y, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()