import numpy as np

# store accuracy and loss for each epoch
train_losses1 = []
val_losses1 = []
train_accuracies1 = []
val_accuracies1 = []

def load_idx(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        magic_number, num_items = np.frombuffer(f.read(8), dtype='>u4')

        # Handle different data types
        if magic_number == 2051:  # Image data
            num_rows, num_cols = np.frombuffer(f.read(8), dtype='>u4')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        elif magic_number == 2049:  # Label data
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number: {magic_number}")

    return data

def load_mnist():
    X_train = load_idx('train-images.idx3-ubyte')
    y_train = load_idx('train-labels.idx1-ubyte')
    X_test = load_idx('t10k-images.idx3-ubyte')
    y_test = load_idx('t10k-labels.idx1-ubyte')
    return X_train, y_train, X_test, y_test

def one_hot_encode(y):
    num_classes = np.max(y) + 1
    matrix = np.eye(num_classes, dtype=int)[y]
    return matrix

def preprocess_data(X, y):
    X_normalized = X / 255.0
    num_samples = X.shape[0]
    X_flattened = X_normalized.reshape(num_samples, -1)
    y_encoded = one_hot_encode(y)
    return X_flattened, y_encoded

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_hidden = np.random.randn(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_output = np.random.randn(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))
    return weights_hidden, bias_hidden, weights_output, bias_output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def forward_pass(X, weights_hidden, bias_hidden, weights_output, bias_output):
    hidden_output = sigmoid(np.dot(X, weights_hidden) + bias_hidden)
    final_output = softmax(np.dot(hidden_output, weights_output) + bias_output)
    return hidden_output, final_output

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def backward_pass(X, y, hidden_output, final_output, weights_hidden, weights_output, learning_rate):
    output_error = y - final_output
    output_delta = output_error * final_output * (1 - final_output)
    hidden_error = np.dot(output_delta, weights_output.T)
    hidden_delta = hidden_error * hidden_output * (1 - hidden_output)
    weights_hidden += learning_rate * np.dot(X.T, hidden_delta)
    weights_output += learning_rate * np.dot(hidden_output.T, output_delta)
    return weights_hidden, weights_output

def predict(X, weights_hidden, bias_hidden, weights_output, bias_output):
    _, final_output = forward_pass(X, weights_hidden, bias_hidden, weights_output, bias_output)
    predictions = np.argmax(final_output, axis=1)
    return predictions

def calculate_accuracy(X, y, weights_hidden, bias_hidden, weights_output, bias_output):
    predictions = predict(X, weights_hidden, bias_hidden, weights_output, bias_output)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == true_labels)
    return accuracy

def train_model(X_train, y_train, X_test, y_test, hidden_size, learning_rate, epochs):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    weights_hidden, bias_hidden, weights_output, bias_output = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        for i in range(X_train.shape[0]):  # SGD: process one sample at a time
            xi = X_train[i:i+1]
            yi = y_train[i:i+1]
            hidden_output, final_output = forward_pass(xi, weights_hidden, bias_hidden, weights_output, bias_output)
            loss = mse(yi, final_output)
            train_losses1.append(loss)
            weights_hidden, weights_output = backward_pass(xi, yi, hidden_output, final_output, weights_hidden, weights_output, learning_rate)
        
        train_accuracy = calculate_accuracy(X_train, y_train, weights_hidden, bias_hidden, weights_output, bias_output)
        train_accuracies1.append(train_accuracy)
        
        # Calculate final output for validation set
        _, val_final_output = forward_pass(X_test, weights_hidden, bias_hidden, weights_output, bias_output)
        val_loss = mse(y_test, val_final_output)
        val_losses1.append(val_loss)
        val_accuracy = calculate_accuracy(X_test, y_test, weights_hidden, bias_hidden, weights_output, bias_output)
        val_losses1.append(val_accuracy)

        print(f'Epoch: {epoch+1}, Training Loss: {loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    return weights_hidden, bias_hidden, weights_output, bias_output

X_train, y_train, X_test, y_test = load_mnist()
X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)

hidden_size = 128
learning_rate = 0.01
epochs = 100

weights_hidden, bias_hidden, weights_output, bias_output = train_model(X_train, y_train, X_test, y_test, hidden_size, learning_rate, epochs)


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.decoder = nn.Linear(hidden_size, input_size)  # linear activation on output layer

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the MNIST dataset
X_train, _, X_test, _ = load_mnist()

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28*28) / 255.
X_test = X_test.reshape(-1, 28*28) / 255.

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

# Create DataLoader
train_data = DataLoader(TensorDataset(X_train, X_train), batch_size=64, shuffle=True)

# Create the model
model = Autoencoder(784, 128)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):  # 100 epochs
    for data in train_data:
        img, _ = data
        img = Variable(img)
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')





import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Load the MNIST dataset
X_train, y_train, X_test, y_test = load_mnist()

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28*28) / 255.
X_test = X_test.reshape(-1, 28*28) / 255.

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

# One-hot encode the labels
y_train_one_hot = torch.from_numpy(one_hot_encode(y_train)).float()
y_test_one_hot = torch.from_numpy(one_hot_encode(y_test)).float()

# Create DataLoader
train_data = DataLoader(TensorDataset(X_train, y_train_one_hot), batch_size=64, shuffle=True)
val_data = DataLoader(TensorDataset(X_test, y_test_one_hot), batch_size=64, shuffle=True)

# Create the model
model = MLP(784, 128, 10)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Initialize lists to store losses and accuracies
train_losses3 = []
val_losses3 = []
train_accuracies3 = []
val_accuracies3 = []

# Train the model
for epoch in range(100):  # 100 epochs
    train_loss = 0
    train_correct = 0
    model.train()
    for data, target in train_data:
        data, target = Variable(data), Variable(target)
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Calculate training loss and accuracy
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        _, true = torch.max(target.data, 1)
        train_correct += (predicted == true).sum().item()

    train_losses3.append(train_loss / len(train_data))
    train_accuracies3.append(train_correct / len(X_train))

    val_loss = 0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_data:
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = criterion(output, target)
            # Calculate validation loss and accuracy
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            _, true = torch.max(target.data, 1)
            val_correct += (predicted == true).sum().item()

    val_losses3.append(val_loss / len(val_data))
    val_accuracies3.append(val_correct / len(X_test))

    print(f'Epoch: {epoch+1}, Training Loss: {train_losses3[-1]:.4f}, Training Accuracy: {train_accuracies3[-1]:.4f}, Validation Loss: {val_losses3[-1]:.4f}, Validation Accuracy: {val_accuracies3[-1]:.4f}')




import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

# Load the MNIST dataset
X_train, y_train, X_test, y_test = load_mnist()

# Normalize the data and convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float() / 255
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float() / 255
y_test = torch.from_numpy(y_test).long()

# Convert labels to one-hot encoding
y_train_one_hot = torch.nn.functional.one_hot(y_train, 10).float()
y_test_one_hot = torch.nn.functional.one_hot(y_test, 10).float()

# Create DataLoader
train_data = DataLoader(TensorDataset(X_train, y_train_one_hot), batch_size=64, shuffle=True)
val_data = DataLoader(TensorDataset(X_test, y_test_one_hot), batch_size=64, shuffle=True)

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Train the autoencoder
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.01)

for epoch in range(100):  # 100 epochs
    for data, _ in train_data:
        data = Variable(data.view(-1, 28*28))
        # Forward pass
        output = autoencoder(data)
        loss = criterion(output, data)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Extract the weights from the autoencoder
autoencoder_weights = list(autoencoder.encoder.children())[0].weight.data, list(autoencoder.encoder.children())[2].weight.data

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),  # Adjusted to match the autoencoder's encoder output
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 10),  # Adjusted to match the MLP's second layer output
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# Initialize the MLP with the weights from the autoencoder
mlp = MLP()
mlp.layer1[0].weight.data = autoencoder_weights[0]
mlp.layer2[0].weight.data = autoencoder_weights[1]  # No need to transpose

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)

# Train the MLP
train_losses4, val_losses4, train_accuracies4, val_accuracies4 = [], [], [], []
for epoch in range(100):  # 100 epochs
    train_loss, val_loss, train_correct, val_correct = 0, 0, 0, 0
    mlp.train()
    for data, target in train_data:
        data = Variable(data.view(-1, 28*28))
        target = Variable(target)
        # Forward pass
        output = mlp(data)
        loss = criterion(output, target)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        _, actual = torch.max(target.data, 1)
        train_correct += (predicted == actual).sum().item()
    train_losses4.append(train_loss/len(train_data))
    train_accuracies4.append(train_correct/len(X_train))

    # Validate the model
    mlp.eval()
    with torch.no_grad():
        for data, target in val_data:
            data = Variable(data.view(-1, 28*28))
            target = Variable(target)
            output = mlp(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            _, actual = torch.max(target.data, 1)
            val_correct += (predicted == actual).sum().item()
        val_losses4.append(val_loss/len(val_data))
        val_accuracies4.append(val_correct/len(X_test))

    print(f'Epoch {epoch+1}, Train Loss: {train_losses4[-1]:.4f}, Val Loss: {val_losses4[-1]:.4f}, Train Acc: {train_accuracies4[-1]:.4f}, Val Acc: {val_accuracies4[-1]:.4f}')







import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses3, label='Model III Training Loss')
plt.plot(val_losses3, label='Model III Validation Loss')
plt.plot(train_losses4, label='Model IV Training Loss')
plt.plot(val_losses4, label='Model IV Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies3, label='Model III Training Accuracy')
plt.plot(val_accuracies3, label='Model III Validation Accuracy')
plt.plot(train_accuracies4, label='Model IV Training Accuracy')
plt.plot(val_accuracies4, label='Model IV Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Load the MNIST dataset
X_train, y_train, X_test, y_test = load_mnist()

# Convert numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

# Create DataLoader objects
train_data = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
val_data = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=64, shuffle=True)

# Initialize the LeNet model
lenet = LeNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lenet.parameters(), lr=0.01, momentum=0.9)

# Train the LeNet
train_losses5, val_losses5, train_accuracies5, val_accuracies5 = [], [], [], []
for epoch in range(100):  # 100 epochs
    train_loss, val_loss, train_correct, val_correct = 0, 0, 0, 0
    lenet.train()
    for data, target in train_data:
        data = Variable(data.view(-1, 1, 28, 28))  # Adjusted for 2D convolution
        target = Variable(target)  # Adjusted for CrossEntropyLoss
        # Forward pass
        output = lenet(data)
        loss = criterion(output, target)
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        train_correct += (predicted == target).sum().item()
    train_losses5.append(train_loss/len(train_data))
    train_accuracies5.append(train_correct/len(X_train))

    # Validate the model
    lenet.eval()
    with torch.no_grad():
        for data, target in val_data:
            data = Variable(data.view(-1, 1, 28, 28))  # Adjusted for 2D convolution
            target = Variable(target)  # Adjusted for CrossEntropyLoss
            output = lenet(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_correct += (predicted == target).sum().item()
        val_losses5.append(val_loss/len(val_data))
        val_accuracies5.append(val_correct/len(X_test))
        print(f'Epoch {epoch+1}, Train Loss: {train_losses5[-1]:.4f}, Val Loss: {val_losses5[-1]:.4f}, Train Acc: {train_accuracies5[-1]:.4f}, Val Acc: {val_accuracies5[-1]:.4f}')
        


import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure()
plt.plot(train_accuracies1, label='Model I Training Accuracy')
plt.plot(val_accuracies1, label='Model I Validation Accuracy')
plt.plot(train_accuracies5, label='Lenet Training Accuracy')
plt.plot(val_accuracies5, label='Lenet Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title("Lenet compared with Best of above (1-hidden layer neural network)")
plt.legend()

plt.show()



