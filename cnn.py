import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# 1) Define layers separately
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        '''
        Parameters:
            in_channels: the number of input channels, 1 in this case (gray-scale image)
            num_classes: the number of classes we want to predict, 10 (0 ~ 9) in this case

        '''
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # flatten before FC
        out = self.fc1(x)
        
        return out


# 2) Define layers using nn.Sequential
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        '''
        Parameters:
            in_channels: the number of input channels, 1 in this case (gray-scale image)
            num_classes: the number of classes we want to predict, 10 (0 ~ 9) in this case

        '''
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.shape[0], -1)
        out = self.fc1(x)
        
        return out


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4 # Karpathy's constant (empirical learning rate for Adam)
batch_size = 64
num_epochs = 10


# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_datset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_datset, batch_size=batch_size, shuffle=True)


# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):

        # Data to device
        data = data.to(device=device) # (N, C, H, W) = (64, 1, 28, 28)
        targets = targets.to(device=device)

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

    num_correct = 0
    num_samples = 0
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=device)
            y = y.to(device=device)

            # Forward
            scores = model(x) # probabilities (ten per sample)
            _, predictions = scores.max(1) # max values, indices

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0) # the number of samples in the batch
    
    model.train()

    return (num_correct / num_samples) * 100


print(f'Train accuracy: {check_accuracy(train_loader, model):.2f}%')
print(f'Test accuracy: {check_accuracy(test_loader, model):.2f}%')

