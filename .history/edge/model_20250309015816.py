import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Input: 1 channel (MNIST), Output: 10 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # Input: 10 channels, Output: 20 filters
        self.fc1 = nn.Linear(320, 50)  # Input: 20 * 4 * 8 = 320 (after pooling), Output: 50
        self.fc2 = nn.Linear(50, num_classes)  # Output: 10 (MNIST classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Conv1 -> ReLU -> MaxPool (2x2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Conv2 -> ReLU -> MaxPool (2x2)
        x = x.view(-1, 320)  # Flatten to 320
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        return self.fc2(x)  # FC2 (logits)

def train(net, trainloader, optimizer, epochs, device: str):
    criterion = nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device: str):
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            loss += criterion(output, labels).item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy