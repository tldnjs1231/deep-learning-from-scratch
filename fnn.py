import torch
import torch.nn.functional as F # parameterless functions like (some) activation functions
import torchvision.datasets as datasets # standard datasets
import torchvision.transforms as transforms # transformations for data augmentation
from torch import optim # optimizers like SGD, Adam, etc.
from torch import nn # all neural network modules
from torch.utils.data import DataLoader # for easier dataset management by creating mini batches, etc.
from tqdm import tqdm # progress bar


class FNN(nn.Module):
    def __init__(self, input_size, num_classes):
        '''
        Define the layers of the network (two fully connected layers in this case)

        Parameters:
            input_size: the size of the input, 784 (28 x 28) in this case
            num_classes: the number of classes we want to predict, 10 (0 ~ 9) in this case

        '''
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50) # takes input_size, 784 nodes to 50 in this case
        self.fc2 = nn.Linear(50, num_classes) # 50 to num_classes, 10 in this case

    def forward(self, x):
        '''
        x here is the mnist images, run through the layers (fc1, fc2) created above
        Add a ReLU activation function in between, (since it has no parameters) use nn.functional

        '''
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        
        return out


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10


# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_datset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datset, batch_size=batch_size, shuffle=True)


# Initialize network
model = FNN(input_size=input_size, num_classes=num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        # Data to device
        data = data.to(device=device) # (N, C, H, W) = (64, 1, 28, 28)
        targets = targets.to(device=device)

        # Reshape data
        data = data.reshape(data.shape[0], -1) # (64, 784)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Train loss {loss:.6f}')


# Check train and test accuracy
def check_accuracy(loader, model):
    '''
    Check the accuracy of the trained model given a loader and a model
    
    '''
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            # Forward
            scores = model(x) # probabilities (ten per sample)
            _, predictions = scores.max(1) # max values, indices

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0) # the number of samples in the batch
    
    model.train()

    return (num_correct / num_samples) * 100


print(f'Train accuracy: {check_accuracy(train_loader, model):.2f}%')
print(f'Test accuracy: {check_accuracy(test_loader, model):.2f}%')


