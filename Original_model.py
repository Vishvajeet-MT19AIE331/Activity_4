import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
# Define model parameters
input_size = X.shape[1]
hidden_size = 50
num_classes = len(set(y))
learning_rate = 0.1
batch_size = 32
num_epochs = 10

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Create PyTorch datasets and dataloaders for train and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Define the ReLU activation function
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    # Forward pass through the neural network.
    def forward(self, x):
        # Pass the input through the first fully connected layer
        out = self.fc1(x)
        # Apply the ReLU activation function
        out = self.relu(out)
        # Pass the output through the second fully connected layer
        out = self.fc2(out)
        return out

    # Define functions for training
    def train(self, train_loader, criterion, optimizer, num_epochs):
        # Iterate over each epoch
        for epoch in range(num_epochs):
            # Iterate over each batch in the training data
            for inputs, labels in train_loader:
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass: compute predicted outputs by passing inputs to the model
                outputs = self(inputs)
                # Calculate the loss
                loss = criterion(outputs, labels)
                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # Update the parameters
                optimizer.step()

    # Method for prediction
    def predict(self, inputs):
        with torch.no_grad():
            # Forward pass through the model
            outputs = self(inputs)
            # Get the predicted class labels
            _, predicted = torch.max(outputs, 1)
        return predicted

# Train the model
model = NeuralNetwork(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model.train(train_loader, criterion, optimizer, num_epochs)

# Convert X_test to PyTorch tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Predict using the trained model
y_pred = model.predict(X_test_tensor)

# Print predictions and accuracy
y_pred_np = np.array(y_pred)

# Calculate accuracy
correct = sum(y_test == y_pred_np)
accuracy = correct / len(y_test)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred_np))
 
